"""
Covariate Effect Predictor — SMILES + condition → PK fold-change.

Given a drug structure and a patient condition (e.g., "hepatic moderate"),
predicts the expected fold-change in Cmax/AUC relative to healthy reference.

Model: XGBoost on Morgan FP + RDKit descriptors + condition one-hot encoding.
Evaluation: Leave-one-drug-out cross-validation (drug-level split).

Usage:
  python pipeline/covariate_predictor.py              # train + evaluate
  python pipeline/covariate_predictor.py --predict     # predict holdout
"""

import json
import math
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not available")

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available")


# ─── Feature Engineering ─────────────────────────────────────────

CONDITION_TYPES = ['renal', 'hepatic', 'food', 'age', 'sex']
CONDITION_LEVELS = {
    'renal':   ['mild', 'moderate', 'severe', 'esrd', 'unspecified'],
    'hepatic': ['mild', 'moderate', 'severe', 'unspecified'],
    'food':    ['fed', 'high_fat', 'low_fat', 'unspecified'],
    'age':     ['elderly', 'pediatric', 'unspecified'],
    'sex':     ['female', 'male', 'unspecified'],
}

PK_PARAMS = ['cmax', 'auc', 'exposure', 'clearance']

FP_NBITS = 1024
FP_RADIUS = 2

DESCRIPTOR_FUNCS = [
    ('MW', Descriptors.ExactMolWt),
    ('LogP', Descriptors.MolLogP),
    ('TPSA', Descriptors.TPSA),
    ('HBD', Descriptors.NumHDonors),
    ('HBA', Descriptors.NumHAcceptors),
    ('RotB', Descriptors.NumRotatableBonds),
    ('AromRings', Descriptors.NumAromaticRings),
    ('RingCount', Descriptors.RingCount),
    ('HeavyAtoms', Descriptors.HeavyAtomCount),
    ('FracCSP3', Descriptors.FractionCSP3),
    ('LabuteASA', Descriptors.LabuteASA),
]


def compute_morgan_fp(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(FP_NBITS)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
    return np.array(fp)


def compute_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(DESCRIPTOR_FUNCS))
    desc = []
    for name, func in DESCRIPTOR_FUNCS:
        try:
            v = func(mol)
            desc.append(v if np.isfinite(v) else 0.0)
        except:
            desc.append(0.0)
    return np.array(desc)


def encode_condition(ctype: str, clevel: str) -> np.ndarray:
    """One-hot encode condition type + level."""
    # Type one-hot (5 dims)
    type_vec = np.zeros(len(CONDITION_TYPES))
    if ctype in CONDITION_TYPES:
        type_vec[CONDITION_TYPES.index(ctype)] = 1.0

    # Level one-hot (variable dims, max ~5 per type)
    # Flatten all levels into single vector
    all_levels = []
    for ct in CONDITION_TYPES:
        all_levels.extend([(ct, l) for l in CONDITION_LEVELS[ct]])

    level_vec = np.zeros(len(all_levels))
    target = (ctype, clevel)
    if target in all_levels:
        level_vec[all_levels.index(target)] = 1.0

    return np.concatenate([type_vec, level_vec])


def encode_pk_param(param: str) -> np.ndarray:
    """One-hot encode PK parameter."""
    param_lower = param.lower()
    vec = np.zeros(len(PK_PARAMS))
    for i, p in enumerate(PK_PARAMS):
        if p in param_lower:
            vec[i] = 1.0
            break
    return vec


def build_features(entry: Dict) -> np.ndarray:
    """Build full feature vector for one entry."""
    smiles = entry.get('smiles', '')
    fp = compute_morgan_fp(smiles)
    desc = compute_descriptors(smiles)
    cond = encode_condition(entry.get('condition_type', ''),
                            entry.get('condition_level', ''))
    pk = encode_pk_param(entry.get('pk_param', ''))
    return np.concatenate([fp, desc, cond, pk])


# ─── Data Loading ────────────────────────────────────────────────

def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Load covariate effects and build feature matrix."""
    with open('data/curated/covariate_effects_pk.json') as f:
        data = json.load(f)

    # Filter: need SMILES + valid fold-change
    valid = [e for e in data
             if e.get('smiles')
             and e.get('fold_change') is not None
             and 0.01 < e['fold_change'] < 20.0]

    print(f"Loaded {len(data)} effects, {len(valid)} valid with SMILES")

    X_list = []
    y_list = []
    groups = []  # NDA for drug-level split

    for e in valid:
        feat = build_features(e)
        X_list.append(feat)
        # Target: log2(fold_change) — centered at 0 for no change
        y_list.append(math.log2(e['fold_change']))
        groups.append(e['nda'])

    X = np.array(X_list)
    y = np.array(y_list)
    g = np.array(groups)

    return X, y, g, valid


# ─── Evaluation ──────────────────────────────────────────────────

def evaluate_logo(X, y, groups, model_class, **model_kwargs):
    """Leave-one-group-out (drug-level) cross-validation."""
    logo = LeaveOneGroupOut()
    unique_groups = np.unique(groups)

    y_pred = np.full_like(y, np.nan)
    fold_count = 0

    for train_idx, test_idx in logo.split(X, y, groups):
        model = model_class(**model_kwargs)
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
        fold_count += 1

    # Metrics on log2 scale
    valid_mask = ~np.isnan(y_pred)
    y_true = y[valid_mask]
    y_hat = y_pred[valid_mask]

    mae_log2 = mean_absolute_error(y_true, y_hat)
    r2 = r2_score(y_true, y_hat)

    # Metrics on fold-change scale
    fc_true = 2 ** y_true
    fc_pred = 2 ** y_hat
    fold_errors = np.maximum(fc_true / fc_pred, fc_pred / fc_true)
    aafe = 10 ** np.mean(np.log10(fold_errors))
    within_1_5 = np.mean(fold_errors <= 1.5) * 100
    within_2 = np.mean(fold_errors <= 2.0) * 100

    return {
        'n': int(valid_mask.sum()),
        'n_drugs': fold_count,
        'mae_log2': round(mae_log2, 4),
        'r2': round(r2, 4),
        'aafe': round(aafe, 3),
        'within_1.5x': round(within_1_5, 1),
        'within_2x': round(within_2, 1),
        'y_true': y_true,
        'y_pred': y_hat,
    }


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    if not HAS_RDKIT or not HAS_SKLEARN:
        print("Error: RDKit and scikit-learn required")
        return

    # Load data
    X, y, groups, entries = load_data()
    print(f"Feature matrix: {X.shape}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}] (log2 fold-change)")
    print(f"Unique drugs: {len(np.unique(groups))}")

    # Condition breakdown
    ctype_counts = Counter(e['condition_type'] for e in entries)
    print(f"\nCondition breakdown:")
    for ct, cnt in sorted(ctype_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct:15s}: {cnt}")

    # ── Baseline: always predict 1.0 (no change) ──
    print(f"\n{'='*60}")
    print("Baseline: predict fold_change = 1.0 (log2 = 0)")
    fc_true = 2 ** y
    fc_pred_baseline = np.ones_like(fc_true)
    fe_baseline = np.maximum(fc_true / fc_pred_baseline, fc_pred_baseline / fc_true)
    aafe_baseline = 10 ** np.mean(np.log10(fe_baseline))
    w2_baseline = np.mean(fe_baseline <= 2.0) * 100
    print(f"  AAFE = {aafe_baseline:.3f}, within 2x = {w2_baseline:.1f}%")

    # ── Baseline: predict condition-level mean ──
    print(f"\nBaseline: condition-level mean fold-change")
    cond_means = {}
    for e, yi in zip(entries, y):
        key = (e['condition_type'], e['condition_level'])
        if key not in cond_means:
            cond_means[key] = []
        cond_means[key].append(yi)
    cond_means = {k: np.mean(v) for k, v in cond_means.items()}

    y_pred_condmean = np.array([
        cond_means.get((e['condition_type'], e['condition_level']), 0.0)
        for e in entries
    ])
    fc_pred_cm = 2 ** y_pred_condmean
    fe_cm = np.maximum(fc_true / fc_pred_cm, fc_pred_cm / fc_true)
    aafe_cm = 10 ** np.mean(np.log10(fe_cm))
    w2_cm = np.mean(fe_cm <= 2.0) * 100
    print(f"  AAFE = {aafe_cm:.3f}, within 2x = {w2_cm:.1f}%")

    # ── Model: GBR with LOGO CV ──
    print(f"\n{'='*60}")
    print("Model: GradientBoosting + Leave-One-Drug-Out CV")
    gbr_results = evaluate_logo(
        X, y, groups,
        GradientBoostingRegressor,
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    print(f"  N={gbr_results['n']}, drugs={gbr_results['n_drugs']}")
    print(f"  MAE(log2) = {gbr_results['mae_log2']:.4f}")
    print(f"  R² = {gbr_results['r2']:.4f}")
    print(f"  AAFE = {gbr_results['aafe']:.3f}")
    print(f"  Within 1.5x = {gbr_results['within_1.5x']:.1f}%")
    print(f"  Within 2x = {gbr_results['within_2x']:.1f}%")

    # ── Model: Ridge (linear baseline) ──
    print(f"\nModel: Ridge + LOGO CV")
    ridge_results = evaluate_logo(
        X, y, groups,
        Ridge, alpha=1.0,
    )
    print(f"  AAFE = {ridge_results['aafe']:.3f}")
    print(f"  Within 2x = {ridge_results['within_2x']:.1f}%")
    print(f"  R² = {ridge_results['r2']:.4f}")

    # ── Model: Random Forest ──
    print(f"\nModel: RandomForest + LOGO CV")
    rf_results = evaluate_logo(
        X, y, groups,
        RandomForestRegressor,
        n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42,
    )
    print(f"  AAFE = {rf_results['aafe']:.3f}")
    print(f"  Within 2x = {rf_results['within_2x']:.1f}%")
    print(f"  R² = {rf_results['r2']:.4f}")

    # ── Summary Table ──
    print(f"\n{'='*60}")
    print(f"Summary (N={len(entries)}, drugs={len(np.unique(groups))})")
    print(f"{'='*60}")
    print(f"{'Model':30s} {'AAFE':>6s} {'2x%':>6s} {'R²':>6s}")
    print(f"{'-'*60}")
    print(f"{'Baseline (no change)':30s} {aafe_baseline:6.3f} {w2_baseline:5.1f}% {'--':>6s}")
    print(f"{'Condition-level mean':30s} {aafe_cm:6.3f} {w2_cm:5.1f}% {'--':>6s}")
    print(f"{'Ridge (linear)':30s} {ridge_results['aafe']:6.3f} "
          f"{ridge_results['within_2x']:5.1f}% {ridge_results['r2']:6.4f}")
    print(f"{'GradientBoosting':30s} {gbr_results['aafe']:6.3f} "
          f"{gbr_results['within_2x']:5.1f}% {gbr_results['r2']:6.4f}")
    print(f"{'RandomForest':30s} {rf_results['aafe']:6.3f} "
          f"{rf_results['within_2x']:5.1f}% {rf_results['r2']:6.4f}")

    # ── Per-condition analysis (best model) ──
    best = gbr_results if gbr_results['aafe'] <= rf_results['aafe'] else rf_results
    best_name = 'GBR' if gbr_results['aafe'] <= rf_results['aafe'] else 'RF'
    y_true_best = best['y_true']
    y_pred_best = best['y_pred']

    print(f"\nPer-condition breakdown (best: {best_name}):")
    for ctype in CONDITION_TYPES:
        mask = np.array([e['condition_type'] == ctype for e in entries])
        if mask.sum() == 0:
            continue
        yt = y_true_best[mask[:len(y_true_best)]] if mask.sum() <= len(y_true_best) else y_true_best
        yp = y_pred_best[mask[:len(y_pred_best)]] if mask.sum() <= len(y_pred_best) else y_pred_best
        if len(yt) == 0:
            continue
        fc_t = 2 ** yt
        fc_p = 2 ** yp
        fe = np.maximum(fc_t / fc_p, fc_p / fc_t)
        aafe_cond = 10 ** np.mean(np.log10(fe))
        w2 = np.mean(fe <= 2.0) * 100
        print(f"  {ctype:15s}: N={len(yt):3d}, AAFE={aafe_cond:.3f}, 2x={w2:.1f}%")

    # ── Save results ──
    results = {
        'n_entries': len(entries),
        'n_drugs': int(len(np.unique(groups))),
        'baselines': {
            'no_change': {'aafe': round(aafe_baseline, 3), 'within_2x': round(w2_baseline, 1)},
            'condition_mean': {'aafe': round(aafe_cm, 3), 'within_2x': round(w2_cm, 1)},
        },
        'models': {
            'ridge': {k: v for k, v in ridge_results.items() if k not in ('y_true', 'y_pred')},
            'gbr': {k: v for k, v in gbr_results.items() if k not in ('y_true', 'y_pred')},
            'rf': {k: v for k, v in rf_results.items() if k not in ('y_true', 'y_pred')},
        },
    }

    out_path = Path('data/validation/covariate_predictor_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
