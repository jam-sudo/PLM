"""
PLM + LLM meta-learner.

Combines XGBoost PLM (structural) with LLM (pharmacological knowledge) via a trained meta-model.

Input features: PLM_pred, LLM_pred, drug descriptors, disagreement
Target: actual log_cd
Training: GroupKFold by inchikey (no leakage)
"""

import json, math, warnings, sys
import numpy as np
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

sys.path.insert(0, 'pipeline')
from llm_enriched_experiment import build_sample, CANONICAL_COND, XGB_PARAMS, smi_to_ik
from ho_diagnostic import build_training


def safe_arr(X):
    return np.where(np.isinf(X), np.nan, X).astype(np.float32)


def descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return [
        float(Descriptors.ExactMolWt(mol)),
        float(Descriptors.MolLogP(mol)),
        float(Descriptors.TPSA(mol)),
        int(Descriptors.NumHDonors(mol)),
        int(Descriptors.NumHAcceptors(mol)),
        int(Descriptors.NumRotatableBonds(mol)),
        int(Descriptors.NumAromaticRings(mol)),
        int(Chem.GetFormalCharge(mol)),
    ]


def aafe(err): return float(10 ** np.mean(np.abs(np.asarray(err))))


def main():
    print("Loading data...")
    with open('data/curated/tdc_adme_data.json') as f: tdc = json.load(f)
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    holdout_drugs = ho['holdout_drugs']
    ho_iks_14 = set(d['inchikey14'] for d in holdout_drugs)
    with open('data/curated/plm_dataset_v10_labels.json') as f: v10 = json.load(f)
    with open('data/llm_extracted/pk_llm_merged.json') as f: llm_data = json.load(f)

    # PLM training data
    X_tr_full, Y_tr_full, g_tr_full, W_tr_full, smi_tr_full = build_training(v10, llm_data, ho_iks_14, tdc)
    print(f"PLM training data: {len(Y_tr_full)} profiles")

    # LLM training predictions
    with open('/tmp/train_full.json') as f: train_full = json.load(f)
    llm_train_preds = {}
    for i in range(8):
        with open(f'/tmp/train_pred_{i}.json') as f:
            for item in json.load(f):
                if item.get('smiles') and item.get('predicted_cmax_ngml', 0) > 0:
                    llm_train_preds[item['smiles']] = item['predicted_cmax_ngml']

    # Map from ik to actual_log_cd + LLM prediction
    ik_to_actual = {}
    ik_to_llm_pred_lcd = {}
    ik_to_smi = {}
    for d in train_full:
        if d['smiles'] not in llm_train_preds: continue
        ik = d['ik']
        pred_lcd = math.log10(llm_train_preds[d['smiles']] / d['dose_mg'])
        ik_to_actual[ik] = d['actual_log_cd']
        ik_to_llm_pred_lcd[ik] = pred_lcd
        ik_to_smi[ik] = d['smiles']
    print(f"Training drugs with LLM predictions: {len(ik_to_actual)}")

    # Aggregate PLM training data to drug level (median per ik)
    # We'll use drug-level training to match LLM which is per-drug
    ik_to_plm_samples = {}
    for i, ik in enumerate(g_tr_full):
        if ik not in ik_to_actual: continue
        ik_to_plm_samples.setdefault(ik, []).append(i)

    # For each drug, take median log_cd
    meta_iks = []
    for ik, indices in ik_to_plm_samples.items():
        if ik not in ik_to_llm_pred_lcd: continue
        meta_iks.append(ik)
    print(f"Drugs with both PLM and LLM: {len(meta_iks)}")

    # Compute PLM OOF predictions (GroupKFold by ik) on PROFILE level, then aggregate
    print("\nComputing PLM OOF predictions (5-fold GroupKFold)...")
    gkf = GroupKFold(n_splits=5)
    oof_plm_profile = np.full(len(Y_tr_full), np.nan, dtype=np.float32)
    for tr_idx, va_idx in gkf.split(X_tr_full, Y_tr_full, g_tr_full):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(safe_arr(X_tr_full[tr_idx]), Y_tr_full[tr_idx], sample_weight=W_tr_full[tr_idx])
        oof_plm_profile[va_idx] = m.predict(safe_arr(X_tr_full[va_idx]))

    # Aggregate PLM predictions to drug level (median)
    ik_to_plm_pred_lcd = {}
    for ik, indices in ik_to_plm_samples.items():
        if ik not in ik_to_llm_pred_lcd: continue
        preds = oof_plm_profile[indices]
        preds = preds[~np.isnan(preds)]
        if len(preds) > 0:
            ik_to_plm_pred_lcd[ik] = float(np.median(preds))

    # Build training meta-feature matrix
    X_meta_tr, y_meta_tr, g_meta_tr = [], [], []
    for ik in meta_iks:
        if ik not in ik_to_plm_pred_lcd: continue
        desc = descriptors(ik_to_smi[ik])
        if desc is None: continue
        plm = ik_to_plm_pred_lcd[ik]
        lcd = ik_to_llm_pred_lcd[ik]
        features = [plm, lcd, plm - lcd, abs(plm - lcd)] + desc
        X_meta_tr.append(features)
        y_meta_tr.append(ik_to_actual[ik])
        g_meta_tr.append(ik)
    X_meta_tr = np.array(X_meta_tr)
    y_meta_tr = np.array(y_meta_tr)
    print(f"\nMeta-training data: N={len(X_meta_tr)}, features={X_meta_tr.shape[1]}")
    print(f"Target (actual_log_cd): mean={np.mean(y_meta_tr):+.3f}, std={np.std(y_meta_tr):.3f}")

    # Baseline: PLM-only AAFE
    plm_err = X_meta_tr[:, 0] - y_meta_tr
    print(f"\nTraining performance:")
    print(f"  PLM OOF (training-CV):  AAFE={aafe(plm_err):.3f}  bias={np.mean(plm_err):+.3f}")
    llm_err = X_meta_tr[:, 1] - y_meta_tr
    print(f"  LLM (training):         AAFE={aafe(llm_err):.3f}  bias={np.mean(llm_err):+.3f}")

    # CV evaluation of meta-learners
    print(f"\nMeta-learner CV (on training):")
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Ridge-10': Ridge(alpha=10.0),
        'Lasso-0.01': Lasso(alpha=0.01),
        'GBR': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    }
    g_meta_arr = np.array(g_meta_tr)
    for name, model in models.items():
        cv_preds = np.full(len(y_meta_tr), np.nan)
        for tr_i, va_i in gkf.split(X_meta_tr, y_meta_tr, groups=g_meta_arr):
            model.fit(X_meta_tr[tr_i], y_meta_tr[tr_i])
            cv_preds[va_i] = model.predict(X_meta_tr[va_i])
        cv_err = cv_preds - y_meta_tr
        print(f"  {name:<12s}: CV AAFE={aafe(cv_err):.3f}  bias={np.mean(cv_err):+.3f}")

    # Load HO data
    print(f"\n{'='*80}\nHO EVALUATION\n{'='*80}")
    # PLM HO predictions (train on full training, predict HO)
    m_plm_full = xgb.XGBRegressor(**XGB_PARAMS)
    m_plm_full.fit(safe_arr(X_tr_full), Y_tr_full, sample_weight=W_tr_full)

    # Build HO features
    with open('data/validation/llm_cot_results.json') as f: cot = json.load(f)
    rounds_data = {f'round_{r}': cot['rounds'][f'round_{r}']['preds'] for r in [1,2,3]}

    X_meta_ho = []
    ho_names_ordered = []
    ho_actuals = []
    for d in holdout_drugs:
        name = d['name']; smi = d.get('smiles'); dose = d.get('dose_mg'); cmax = d.get('cmax_obs_ngml')
        if not smi or not dose or not cmax: continue
        ik = d.get('inchikey14', '')
        # PLM prediction
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None: continue
        plm_pred = float(m_plm_full.predict(safe_arr(np.array([s])))[0])
        # LLM 3-round geomean
        llm_preds = []
        for r in [1,2,3]:
            if name in rounds_data[f'round_{r}']:
                llm_preds.append(math.log10(rounds_data[f'round_{r}'][name]/dose))
        if len(llm_preds) < 2: continue
        llm_pred = float(np.mean(llm_preds))
        desc = descriptors(smi)
        if desc is None: continue
        features = [plm_pred, llm_pred, plm_pred - llm_pred, abs(plm_pred - llm_pred)] + desc
        X_meta_ho.append(features)
        ho_names_ordered.append(name)
        ho_actuals.append(math.log10(cmax/dose))
    X_meta_ho = np.array(X_meta_ho)
    ho_actuals = np.array(ho_actuals)
    print(f"HO feature matrix: {X_meta_ho.shape}")

    # Baselines
    plm_ho_err = X_meta_ho[:, 0] - ho_actuals
    llm_ho_err = X_meta_ho[:, 1] - ho_actuals
    print(f"\n  PLM HO:  AAFE={aafe(plm_ho_err):.3f}  bias={np.mean(plm_ho_err):+.3f}")
    print(f"  LLM HO (geomean, uncal): AAFE={aafe(llm_ho_err):.3f}  bias={np.mean(llm_ho_err):+.3f}")
    print(f"  LLM HO + named cal:      AAFE={aafe(llm_ho_err - 0.022):.3f}  bias={np.mean(llm_ho_err - 0.022):+.3f}")

    # Train meta-learners on full training, predict HO
    print(f"\nMeta-learner HO predictions (trained on full training):")
    for name, model in models.items():
        model.fit(X_meta_tr, y_meta_tr)
        ho_preds = model.predict(X_meta_ho)
        ho_err = ho_preds - ho_actuals
        print(f"  {name:<12s}: AAFE={aafe(ho_err):.3f}  bias={np.mean(ho_err):+.3f}")

    # Save
    results = {
        'prior_best': 2.087,
        'candidates': {},
    }
    for name, model in models.items():
        model.fit(X_meta_tr, y_meta_tr)
        ho_preds = model.predict(X_meta_ho)
        ho_err = ho_preds - ho_actuals
        results['candidates'][name] = {
            'aafe': round(aafe(ho_err), 3),
            'bias': round(float(np.mean(ho_err)), 3),
        }
    with open('data/validation/meta_learner_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved data/validation/meta_learner_results.json")


if __name__ == '__main__':
    main()
