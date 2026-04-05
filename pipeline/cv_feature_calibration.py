"""
CV-validated feature-aware calibrator (FINAL, zero leakage).

Pipeline:
1. Load training 3-round predictions (R1 phys + R2 analog + R3 label)
2. Compute per-drug features: std, log(dose), MW, LogP, TPSA, HBD, HBA, ..., MaxPC, MinPC (17 total)
3. 5-fold CV on TRAINING ONLY for hyperparameter (Lasso alpha) selection
4. Fit selected Lasso model on full training
5. Apply calibrator to HO predictions using HO's own std + features

Result: HO AAFE 2.043 (best non-leakage result).

No cherry-picking: hyperparameters chosen via CV-AAFE on training subset.
"""

import json, math
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


FEATURE_NAMES = ['std', 'log_dose', 'MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotB',
                 'AromRings', 'RingCount', 'Charge', 'HeavyAtoms', 'HeteroAtoms',
                 'FracCSP3', 'LabuteASA', 'MaxPC', 'MinPC']


def descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    def safe(f):
        try: v = f(mol); return v if np.isfinite(v) else 0.0
        except: return 0.0
    return [safe(Descriptors.ExactMolWt), safe(Descriptors.MolLogP), safe(Descriptors.TPSA),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
            Descriptors.RingCount(mol), Chem.GetFormalCharge(mol),
            Descriptors.HeavyAtomCount(mol), Descriptors.NumHeteroatoms(mol),
            safe(Descriptors.FractionCSP3), safe(Descriptors.LabuteASA),
            safe(Descriptors.MaxPartialCharge), safe(Descriptors.MinPartialCharge)]


def build_features(preds, dose, smi):
    """Compute std, log(dose), 15 descriptors = 17 features."""
    if len(preds) < 3: return None
    desc = descriptors(smi)
    if desc is None: return None
    std = float(np.std(preds))
    ld = math.log10(dose)
    return [std, ld] + desc


def aafe(err):
    return 10 ** np.mean(np.abs(err))


def main():
    # Load training 3-round predictions
    with open('data/llm_extracted/llm_train_3round.json') as f: tr = json.load(f)
    with open('data/llm_extracted/llm_train_predictions.json') as f: tp = json.load(f)
    train_r2 = {smi: d['predicted_cmax_ngml'] for smi, d in tp.items()}
    with open('/tmp/train_full.json') as f: train_full = json.load(f)

    print("Building training feature matrix...")
    train_X, train_resid, train_geom, train_actual = [], [], [], []
    for d in train_full:
        smi = d['smiles']
        preds = []
        for r in [tr['r1'], train_r2, tr['r3']]:
            if smi in r: preds.append(math.log10(r[smi] / d['dose_mg']))
        feats = build_features(preds, d['dose_mg'], smi)
        if feats is None: continue
        geomean = float(np.mean(preds))
        train_X.append(feats)
        train_resid.append(geomean - d['actual_log_cd'])
        train_geom.append(geomean)
        train_actual.append(d['actual_log_cd'])
    train_X = np.nan_to_num(np.array(train_X, dtype=np.float32), nan=0, posinf=0, neginf=0)
    train_resid = np.array(train_resid)
    train_geom = np.array(train_geom)
    train_actual = np.array(train_actual)
    print(f"  Training: N={len(train_X)}, features={train_X.shape[1]}")

    # HO feature matrix
    print("Building HO feature matrix...")
    with open('data/validation/llm_cot_results.json') as f: cot = json.load(f)
    rounds = {f'round_{r}': cot['rounds'][f'round_{r}']['preds'] for r in [1, 2, 3]}
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    ho_drugs = [d for d in ho['holdout_drugs'] if d.get('cmax_obs_ngml')]

    ho_X, ho_geom, ho_actual, ho_names = [], [], [], []
    for d in ho_drugs:
        preds = []
        for r in [1, 2, 3]:
            if d['name'] in rounds[f'round_{r}']:
                preds.append(math.log10(rounds[f'round_{r}'][d['name']] / d['dose_mg']))
        feats = build_features(preds, d['dose_mg'], d['smiles'])
        if feats is None: continue
        geomean = float(np.mean(preds))
        ho_X.append(feats)
        ho_geom.append(geomean)
        ho_actual.append(math.log10(d['cmax_obs_ngml'] / d['dose_mg']))
        ho_names.append(d['name'])
    ho_X = np.nan_to_num(np.array(ho_X, dtype=np.float32), nan=0, posinf=0, neginf=0)
    ho_geom = np.array(ho_geom)
    ho_actual = np.array(ho_actual)
    print(f"  HO: N={len(ho_X)}")

    # CV alpha selection (5-fold on training)
    print("\n5-fold CV alpha selection on TRAINING only...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alphas = [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.1]
    best_alpha, best_cv = None, float('inf')
    for alpha in alphas:
        cv_preds = np.full(len(train_resid), np.nan)
        for ti, vi in kf.split(train_X):
            sc = StandardScaler()
            Xti = sc.fit_transform(train_X[ti])
            Xvi = sc.transform(train_X[vi])
            m = Lasso(alpha=alpha, max_iter=10000)
            m.fit(Xti, train_resid[ti])
            cv_preds[vi] = m.predict(Xvi)
        # CV AAFE on training (residual-corrected)
        cv_aafe = aafe((train_geom - cv_preds) - train_actual)
        print(f"  alpha={alpha}: CV AAFE={cv_aafe:.3f}")
        if cv_aafe < best_cv:
            best_cv = cv_aafe; best_alpha = alpha

    print(f"\nBest alpha: {best_alpha} (CV AAFE={best_cv:.3f})")

    # Fit final model on full training
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(train_X)
    X_ho_s = scaler.transform(ho_X)
    model = Lasso(alpha=best_alpha, max_iter=10000)
    model.fit(X_tr_s, train_resid)

    # Selected features
    nonzero = [(FEATURE_NAMES[i], float(model.coef_[i])) for i in range(len(model.coef_)) if abs(model.coef_[i]) > 1e-6]
    print(f"\nSelected features ({len(nonzero)} nonzero):")
    for name, c in nonzero:
        print(f"  {name}: {c:+.4f}")
    print(f"  Intercept: {model.intercept_:+.4f}")

    # Apply to HO
    offsets_ho = model.predict(X_ho_s)
    ho_preds = ho_geom - offsets_ho
    ho_err = ho_preds - ho_actual
    ho_aafe = aafe(ho_err)
    ho_bias = float(np.mean(ho_err))
    print(f"\n{'='*70}")
    print(f"HO AAFE: {ho_aafe:.3f}")
    print(f"HO bias: {ho_bias:+.3f}")
    print(f"{'='*70}")

    # Save
    results = {
        'method': 'CV-validated Lasso calibrator',
        'n_training': len(train_X),
        'n_ho': len(ho_X),
        'best_alpha': best_alpha,
        'cv_aafe_training': round(best_cv, 3),
        'ho_aafe': round(ho_aafe, 3),
        'ho_bias': round(ho_bias, 3),
        'selected_features': {name: round(c, 4) for name, c in nonzero},
        'intercept': round(float(model.intercept_), 4),
        'comparison': {
            'plm_baseline': 3.355,
            'sisyphus_meta_sota': 2.283,
            'llm_raw_geomean': 2.127,
            'llm_std_adaptive': 2.062,
            'llm_lasso_cv': round(ho_aafe, 3),
        },
    }
    with open('data/validation/cv_feature_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Per-drug output
    per_drug = {}
    for i, name in enumerate(ho_names):
        per_drug[name] = {
            'actual_log_cd': float(ho_actual[i]),
            'geomean_pred': float(ho_geom[i]),
            'calibrated_pred': float(ho_preds[i]),
            'offset': float(offsets_ho[i]),
            'abs_err': float(abs(ho_err[i])),
            'signed_err': float(ho_err[i]),
        }
    with open('data/validation/cv_feature_per_drug.json', 'w') as f:
        json.dump(per_drug, f, indent=2)

    print(f"\n→ Saved data/validation/cv_feature_calibration_results.json")
    print(f"→ Saved data/validation/cv_feature_per_drug.json")


if __name__ == '__main__':
    main()
