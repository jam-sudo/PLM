"""
PLM Phase 1: XGBoost Multi-Output C-t Profile Prediction

Features: Morgan FP 2048 + [log10(dose), route_onehot, form_onehot, food_onehot]
Target: 13 timepoint log10(C/dose) values
Evaluation: Cmax AAFE on drug-level holdout

Usage:
    python models/train_xgboost.py
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import GroupKFold
import xgboost as xgb


GRID = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]


def load_dataset(path: str = "data/curated/plm_dataset_v0.4.json"):
    """Load and filter dataset for model training."""
    with open(path) as f:
        dataset = json.load(f)

    profiles = dataset["profiles"]
    valid = []
    for p in profiles:
        # Must have: SMILES, dose, grid concentrations
        if not p.get("smiles") or not p.get("dose_mg") or p["dose_mg"] <= 0:
            continue
        concs = p.get("concentrations_ngml", [])
        if not concs or not any(v is not None and v > 0 for v in concs):
            continue
        # Must have at least 3 non-None grid points
        n_valid = sum(1 for v in concs if v is not None and v > 0)
        if n_valid < 3:
            continue
        # Validate SMILES
        mol = Chem.MolFromSmiles(p["smiles"])
        if mol is None:
            continue
        valid.append(p)

    print(f"Loaded {len(valid)}/{len(profiles)} valid profiles")
    return valid


def compute_features(profile: dict) -> np.ndarray:
    """Compute feature vector for a single profile."""
    mol = Chem.MolFromSmiles(profile["smiles"])
    # Morgan fingerprint (2048 bits, radius 2)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.array(fp, dtype=np.float32)

    # Dose feature
    dose = math.log10(profile["dose_mg"])

    # Route one-hot (oral, IV, SC, IM, other)
    route = profile.get("route", "oral")
    route_vec = [0, 0, 0, 0, 0]
    route_map = {"oral": 0, "IV": 1, "SC": 2, "IM": 3}
    route_vec[route_map.get(route, 4)] = 1

    # Formulation one-hot (IR_tablet, IR_capsule, ER_tablet, solution, other)
    form = profile.get("formulation", "other")
    form_vec = [0, 0, 0, 0, 0]
    form_map = {"IR_tablet": 0, "IR_capsule": 1, "ER_tablet": 2, "solution": 3}
    form_vec[form_map.get(form, 4)] = 1

    # Food one-hot (fasted, fed, not_specified)
    food = profile.get("food_effect", "not_specified")
    food_vec = [0, 0, 0]
    food_map = {"fasted": 0, "fed": 1}
    food_vec[food_map.get(food, 2)] = 1

    # Concatenate all features
    features = np.concatenate([
        fp_array,
        [dose],
        route_vec,
        form_vec,
        food_vec,
    ])
    return features


def compute_target(profile: dict) -> np.ndarray:
    """Compute target vector: log10(C(t)/dose) at 13 timepoints."""
    concs = profile.get("concentrations_ngml", [])
    dose = profile["dose_mg"]

    target = np.full(13, np.nan)
    for i, c in enumerate(concs):
        if i < 13 and c is not None and c > 0 and dose > 0:
            target[i] = math.log10(c / dose)

    return target


def aafe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average Absolute Fold Error."""
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        return float("nan")
    return float(10 ** np.mean(np.abs(y_true[valid] - y_pred[valid])))


def cmax_from_predictions(log_c_dose: np.ndarray, dose: float) -> float:
    """Recover Cmax from log10(C/dose) predictions."""
    valid = np.isfinite(log_c_dose)
    if not np.any(valid):
        return 0.0
    return float(np.max(10 ** log_c_dose[valid]) * dose)


def train_and_evaluate(profiles: list):
    """Train XGBoost with drug-level GroupKFold, evaluate Cmax AAFE."""
    # Build feature matrix and targets
    X_list, Y_list, groups, doses, drug_names = [], [], [], [], []

    for p in profiles:
        x = compute_features(p)
        y = compute_target(p)
        X_list.append(x)
        Y_list.append(y)
        # Group by drug (InChIKey first 14 chars for grouping)
        mol = Chem.MolFromSmiles(p["smiles"])
        inchi = Chem.MolToInchi(mol) if mol else ""
        ik = Chem.InchiToInchiKey(inchi)[:14] if inchi else p["smiles"][:20]
        groups.append(ik)
        doses.append(p["dose_mg"])
        drug_names.append(p.get("drug_name", ""))

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    groups = np.array(groups)
    doses = np.array(doses)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target matrix: {Y.shape}")
    print(f"Unique drug groups: {len(set(groups))}")

    # Drug-level GroupKFold (5-fold)
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    all_true_cmax = []
    all_pred_cmax = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        doses_test = doses[test_idx]
        test_drugs = set(groups[test_idx])

        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        print(f"  Train: {len(train_idx)} profiles, Test: {len(test_idx)} profiles")
        print(f"  Test drugs: {len(test_drugs)}")

        # Train one XGBoost per timepoint
        fold_preds = np.full_like(Y_test, np.nan)

        for t in range(13):
            y_train_t = Y_train[:, t]
            y_test_t = Y_test[:, t]

            # Skip if too few valid training samples
            valid_train = np.isfinite(y_train_t)
            valid_test = np.isfinite(y_test_t)

            if np.sum(valid_train) < 10 or np.sum(valid_test) < 2:
                continue

            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.3,
                reg_alpha=1.0,
                reg_lambda=5.0,
                min_child_weight=5,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X_train[valid_train], y_train_t[valid_train])
            preds = model.predict(X_test)
            fold_preds[:, t] = preds

        # Compute Cmax for each test profile
        for i in range(len(test_idx)):
            true_cmax = cmax_from_predictions(Y_test[i], doses_test[i])
            pred_cmax = cmax_from_predictions(fold_preds[i], doses_test[i])
            if true_cmax > 0 and pred_cmax > 0:
                all_true_cmax.append(true_cmax)
                all_pred_cmax.append(pred_cmax)

        # Fold-level AAFE on timepoints
        fold_aafe = aafe(Y_test, fold_preds)
        print(f"  Timepoint AAFE: {fold_aafe:.3f}")

    # Overall Cmax AAFE
    true_cmax = np.array(all_true_cmax)
    pred_cmax = np.array(all_pred_cmax)

    if len(true_cmax) > 0:
        log_errors = np.abs(np.log10(pred_cmax / true_cmax))
        cmax_aafe = 10 ** np.mean(log_errors)
        pct_within_2fold = np.mean(log_errors < np.log10(2)) * 100
        pct_within_3fold = np.mean(log_errors < np.log10(3)) * 100
    else:
        cmax_aafe = float("nan")
        pct_within_2fold = 0
        pct_within_3fold = 0

    print(f"\n{'='*50}")
    print(f"=== OVERALL RESULTS ===")
    print(f"{'='*50}")
    print(f"Profiles evaluated: {len(true_cmax)}")
    print(f"Cmax AAFE: {cmax_aafe:.3f}")
    print(f"Cmax within 2-fold: {pct_within_2fold:.1f}%")
    print(f"Cmax within 3-fold: {pct_within_3fold:.1f}%")
    print(f"\nSisyphus baselines:")
    print(f"  Meta AAFE: 2.283")
    print(f"  ML AAFE:   2.336")
    print(f"  Engine:    3.416")

    results = {
        "n_profiles": len(profiles),
        "n_drugs": len(set(groups)),
        "n_folds": n_splits,
        "cmax_aafe": round(cmax_aafe, 3),
        "cmax_within_2fold_pct": round(pct_within_2fold, 1),
        "cmax_within_3fold_pct": round(pct_within_3fold, 1),
        "n_evaluated": len(true_cmax),
        "sisyphus_meta_aafe": 2.283,
        "sisyphus_ml_aafe": 2.336,
    }

    Path("models").mkdir(exist_ok=True)
    with open("models/xgboost_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    profiles = load_dataset()
    if len(profiles) < 20:
        print("Too few profiles for model training")
    else:
        train_and_evaluate(profiles)
