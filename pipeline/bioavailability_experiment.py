"""
Experiment: Add predicted bioavailability as feature to improve PLM.

Pre-registration:
  Hypothesis: Continuous F prediction reduces overprediction for low-F drugs
  Success: F<20% group AAFE 6.0 → <4.0, OR overall AAFE 3.355 → <3.2
  Method: Train F classifier on TDC data, add P(F>20%) as feature

Usage:
    python -m pipeline.bioavailability_experiment
"""

import json, math, warnings, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

sys.path.insert(0, 'pipeline')
from llm_enriched_experiment import (
    smi_to_ik, build_sample, CANONICAL_COND, normalize_condition, XGB_PARAMS,
    smiles_to_fp, smiles_to_physchem,
)
from ho_diagnostic import build_training, morgan_fp_2048


def train_f_predictor():
    """Train bioavailability classifier on TDC data. Returns model + CV AUC."""
    ba = pd.read_csv('data/bioavailability_ma.tab', sep='\t')
    print(f"TDC bioavailability: {len(ba)} drugs (F>20%: {ba['Y'].sum()}, F<20%: {(1-ba['Y']).sum()})")

    # Features: Morgan FP + physicochemical descriptors
    X_list, Y_list, valid_smi = [], [], []
    for _, row in ba.iterrows():
        smi = row['Drug']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
        desc = np.array([
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.ExactMolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumAromaticRings(mol),
        ], dtype=np.float32)
        X_list.append(np.concatenate([fp, desc]))
        Y_list.append(row['Y'])
        valid_smi.append(smi)

    X = np.array(X_list)
    Y = np.array(Y_list)
    print(f"Valid molecules: {len(X)} (class balance: {Y.mean():.2f})")

    # 5-fold CV to measure performance
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    for train_idx, test_idx in skf.split(X, Y):
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, random_state=42,
            eval_metric='logloss', verbosity=0,
        )
        m.fit(X[train_idx], Y[train_idx])
        probs = m.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(Y[test_idx], probs)
        cv_aucs.append(auc)

    print(f"F classifier CV AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")

    # Train final model on all data
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, random_state=42,
        eval_metric='logloss', verbosity=0,
    )
    model.fit(X, Y)

    return model, np.mean(cv_aucs)


def predict_f(model, smiles: str) -> float:
    """Predict P(F>20%) for a given SMILES. Returns probability [0, 1]."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
    desc = np.array([
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.ExactMolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumAromaticRings(mol),
    ], dtype=np.float32)
    x = np.concatenate([fp, desc]).reshape(1, -1)
    return float(model.predict_proba(x)[0, 1])


def build_sample_with_f(smi, dose, ik, conditions, tdc, f_model, use_conditions=True):
    """build_sample + predicted F probability as extra feature."""
    base = build_sample(smi, dose, ik, conditions, tdc, use_conditions)
    if base is None:
        return None
    f_prob = predict_f(f_model, smi)
    # Add F probability and log(F_proxy) as features
    # F_proxy: map probability to estimated F (rough: P * 0.6 + 0.05)
    f_proxy = f_prob * 0.6 + 0.05 if not np.isnan(f_prob) else np.nan
    log_f = math.log10(f_proxy) if not np.isnan(f_proxy) and f_proxy > 0 else np.nan
    return np.concatenate([base, [f_prob, log_f]])


def main():
    print("=" * 75)
    print("BIOAVAILABILITY FEATURE EXPERIMENT")
    print("=" * 75)
    print("Pre-registered success criteria:")
    print("  1. F<20% group AAFE: 6.018 → <4.0")
    print("  2. Overall AAFE: 3.355 → <3.2")
    print()

    # Step 1: Train F predictor
    print(">>> Step 1: Train bioavailability classifier")
    f_model, f_auc = train_f_predictor()
    print()

    # Step 2: Load PLM data
    print(">>> Step 2: Load PLM data")
    with open('data/curated/tdc_adme_data.json') as f:
        tdc = json.load(f)
    with open('data/validation/holdout_definition.json') as f:
        ho_data = json.load(f)
    holdout_drugs = ho_data['holdout_drugs']
    ho_iks = set(d['inchikey14'] for d in holdout_drugs)
    with open('data/curated/plm_dataset_v10_labels.json') as f:
        v10 = json.load(f)
    with open('data/llm_extracted/pk_llm_merged.json') as f:
        llm = json.load(f)

    # Step 3: Build training with F feature
    print(">>> Step 3: Build training sets")

    # Baseline (no F feature) — reproduce ho_diagnostic exactly
    X_base, Y_base, g_base, W_base, smi_base = build_training(v10, llm, ho_iks, tdc)
    X_base = np.where(np.isinf(X_base), np.nan, X_base)
    print(f"  Baseline: {len(Y_base)} samples, {len(set(g_base))} drugs, {X_base.shape[1]} features")

    # Expanded: add F features to each sample
    print("  Building F-augmented features...")
    X_f_list, Y_f_list, g_f_list, W_f_list = [], [], [], []
    for i in range(len(Y_base)):
        smi = smi_base[i]
        f_prob = predict_f(f_model, smi)
        f_proxy = f_prob * 0.6 + 0.05 if not np.isnan(f_prob) else np.nan
        log_f = math.log10(f_proxy) if not np.isnan(f_proxy) and f_proxy > 0 else np.nan
        x_augmented = np.concatenate([X_base[i], [f_prob, log_f]])
        X_f_list.append(x_augmented)
        Y_f_list.append(Y_base[i])
        g_f_list.append(g_base[i])
        W_f_list.append(W_base[i])

    X_f = np.array(X_f_list, dtype=np.float32)
    Y_f = np.array(Y_f_list, dtype=np.float32)
    g_f = np.array(g_f_list)
    W_f = np.array(W_f_list, dtype=np.float32)
    X_f = np.where(np.isinf(X_f), np.nan, X_f)
    print(f"  F-augmented: {len(Y_f)} samples, {X_f.shape[1]} features (+2)")

    # Step 4: Train and evaluate both
    print()
    print(">>> Step 4: Train and evaluate")

    results = {}
    for label, X_tr, Y_tr, W_tr, use_f in [
        ("baseline", X_base, Y_base, W_base, False),
        ("with_F_feature", X_f, Y_f, W_f, True),
    ]:
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr, Y_tr, sample_weight=W_tr)

        per_drug = []
        for d in holdout_drugs:
            smi, dose, cmax = d.get('smiles'), d.get('dose_mg'), d.get('cmax_obs_ngml')
            if not smi or not dose or dose <= 0 or not cmax or cmax <= 0:
                continue
            ik = d.get('inchikey14', '')

            if use_f:
                s = build_sample_with_f(smi, dose, ik, CANONICAL_COND, tdc, f_model, True)
            else:
                s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, True)
            if s is None:
                continue

            X_h = np.array([s], dtype=np.float32)
            X_h = np.where(np.isinf(X_h), np.nan, X_h)
            pred = float(m.predict(X_h)[0])
            actual = math.log10(cmax / dose)
            err = abs(pred - actual)

            # Get F group
            ba = tdc.get(ik, {}).get('bioavailability_binary')
            f_pred = predict_f(f_model, smi)

            per_drug.append({
                'name': d['name'],
                'actual': actual,
                'pred': pred,
                'abs_err': err,
                'signed_err': pred - actual,
                'fold_err': 10**err,
                'ba_binary': ba,
                'f_pred': round(f_pred, 4) if not np.isnan(f_pred) else None,
            })

        # Overall metrics
        errs = [d['abs_err'] for d in per_drug]
        aafe = 10**np.mean(errs)
        bias = np.mean([d['signed_err'] for d in per_drug])

        # Stratified by F
        f_high = [d for d in per_drug if d['ba_binary'] == 1.0]
        f_low = [d for d in per_drug if d['ba_binary'] == 0.0]
        f_unk = [d for d in per_drug if d['ba_binary'] is None]

        aafe_high = 10**np.mean([d['abs_err'] for d in f_high]) if f_high else None
        aafe_low = 10**np.mean([d['abs_err'] for d in f_low]) if f_low else None
        aafe_unk = 10**np.mean([d['abs_err'] for d in f_unk]) if f_unk else None

        print(f"\n  --- {label} ---")
        print(f"  Overall: AAFE={aafe:.3f}  bias={bias:+.3f}  N={len(per_drug)}")
        if f_high: print(f"  F>20%:   AAFE={aafe_high:.3f}  N={len(f_high)}")
        if f_low:  print(f"  F<20%:   AAFE={aafe_low:.3f}  N={len(f_low)}")
        if f_unk:  print(f"  F unkn:  AAFE={aafe_unk:.3f}  N={len(f_unk)}")

        results[label] = {
            'aafe': round(aafe, 3),
            'bias': round(bias, 4),
            'n': len(per_drug),
            'aafe_f_high': round(aafe_high, 3) if aafe_high else None,
            'aafe_f_low': round(aafe_low, 3) if aafe_low else None,
            'aafe_f_unknown': round(aafe_unk, 3) if aafe_unk else None,
            'n_f_high': len(f_high),
            'n_f_low': len(f_low),
            'n_f_unknown': len(f_unk),
        }

    # Step 5: Verdict
    print()
    print("=" * 75)
    print("VERDICT")
    print("=" * 75)
    b = results['baseline']
    f = results['with_F_feature']

    print(f"  Overall AAFE:  {b['aafe']:.3f} → {f['aafe']:.3f}  (delta: {f['aafe']-b['aafe']:+.3f})")
    print(f"  F<20% AAFE:    {b['aafe_f_low']:.3f} → {f['aafe_f_low']:.3f}  (delta: {f['aafe_f_low']-b['aafe_f_low']:+.3f})" if b['aafe_f_low'] and f['aafe_f_low'] else "")
    print(f"  Bias:          {b['bias']:+.3f} → {f['bias']:+.3f}")
    print()

    success_overall = f['aafe'] < 3.2
    success_flow = f['aafe_f_low'] is not None and f['aafe_f_low'] < 4.0
    print(f"  Criterion 1 (overall <3.2):  {'PASS' if success_overall else 'FAIL'}")
    print(f"  Criterion 2 (F<20% <4.0):    {'PASS' if success_flow else 'FAIL'}")

    # Save
    output = {
        'pre_registration': {
            'hypothesis': 'Predicted F feature reduces low-F overprediction',
            'success_criteria': ['overall AAFE < 3.2', 'F<20% AAFE < 4.0'],
        },
        'f_classifier_cv_auc': round(f_auc, 3),
        'results': results,
        'verdict': {
            'overall_improved': bool(success_overall),
            'f_low_improved': bool(success_flow),
        },
    }
    with open('data/validation/bioavailability_experiment_results.json', 'w') as f_out:
        json.dump(output, f_out, indent=2)
    print(f"\nSaved to data/validation/bioavailability_experiment_results.json")


if __name__ == "__main__":
    main()
