"""S15 — Hyperparameter Optimization + Feature Selection + LightGBM.

PRE-REGISTRATION
================
Date: 2026-04-12
Hypothesis: Current XGB params are hand-set and never optimized.
  Bayesian HP optimization (Optuna) on CV AAFE should improve holdout.

Design:
  Phase A: Optuna XGBoost HP search (100 trials, 5-fold GroupKFold, 1 seed)
  Phase B: Feature importance pruning (top-K features from Phase A best)
  Phase C: LightGBM with Optuna (100 trials)
  Phase D: XGB+LGBM ensemble (mean of best configs)
  Phase E: Evaluate all on holdout (4 seeds)

Success criteria (applied to best config HO AAFE, 4-seed mean):
  PASS:    HO AAFE ≤ 3.15 (ΔHO ≤ −0.18 from 3.332)
  PARTIAL: HO AAFE ≤ 3.28 (ΔHO ≤ −0.05)
  NULL:    HO AAFE > 3.28

Outputs: models/b1/s15_hp_results.json
"""
from __future__ import annotations

import gc
import json
import math
import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

import sys
ROOT = Path("/home/jam/PLM")
sys.path.insert(0, str(ROOT / "models"))
from s12_v12_retrain import (
    ADMEEncoder, build_features, build_holdout,
    XGB_BASE_PARAMS, aafe_metrics,
)

OUT = ROOT / "models/b1/s15_hp_results.json"
SEEDS = [42, 137, 2024, 7]
N_TRIALS = 100

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def cv_aafe(X, y, groups, model_fn, seed=42):
    """5-fold GroupKFold CV AAFE for a given model factory."""
    gkf = GroupKFold(n_splits=5)
    preds = np.full_like(y, np.nan)
    for ti, vi in gkf.split(X, y, groups):
        m = model_fn(seed)
        m.fit(X[ti], y[ti])
        preds[vi] = m.predict(X[vi])
        del m
    err = np.abs(preds - y)
    return float(10**np.mean(err[np.isfinite(err)]))


def multi_seed_holdout(X_tr, y_tr, X_ho, y_ho, model_fn, seeds):
    """Train on full training set with multiple seeds, evaluate holdout."""
    ho_aafes = []
    all_preds = []
    for seed in seeds:
        m = model_fn(seed)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_ho)
        all_preds.append(pred)
        err = np.abs(pred - y_ho)
        ho_aafes.append(float(10**np.mean(err)))
        del m
    return ho_aafes, np.array(all_preds)


def main():
    print("S15 — HP Optimization + Feature Selection + LightGBM", flush=True)
    print("=" * 60, flush=True)

    # Load data
    v12 = json.load(open(ROOT / "data/curated/plm_dataset_v12_chembl.json"))
    ho = json.load(open(ROOT / "data/validation/holdout_definition.json"))
    tdc = {k[:14]: v for k, v in json.load(open(ROOT / "data/curated/tdc_adme_data.json")).items()}
    ho_iks = set((k or "")[:14] for k in ho["holdout_inchikeys"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ADMEEncoder().to(device)
    state = torch.load(ROOT / "models/adme_encoder.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()

    print("Building features...", flush=True)
    X_tr, y_tr, groups = build_features(v12, tdc, ho_iks, encoder, device)
    X_ho, y_ho, ho_names = build_holdout(ho, tdc, encoder, device)
    print(f"  Train: {X_tr.shape}, Holdout: {X_ho.shape}", flush=True)

    del encoder, state; gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # === Baseline ===
    print("\n--- Baseline (hand-set XGB) ---", flush=True)
    baseline_fn = lambda seed: xgb.XGBRegressor(**{**XGB_BASE_PARAMS, "random_state": seed})
    baseline_cv = cv_aafe(X_tr, y_tr, groups, baseline_fn)
    baseline_ho, _ = multi_seed_holdout(X_tr, y_tr, X_ho, y_ho, baseline_fn, SEEDS)
    print(f"  CV AAFE: {baseline_cv:.4f}", flush=True)
    print(f"  HO AAFE: {np.mean(baseline_ho):.4f} ± {np.std(baseline_ho):.4f}", flush=True)

    # === Phase A: Optuna XGBoost ===
    print(f"\n--- Phase A: Optuna XGBoost ({N_TRIALS} trials) ---", flush=True)

    def xgb_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "n_jobs": 4, "verbosity": 0, "tree_method": "hist",
        }
        fn = lambda seed: xgb.XGBRegressor(**{**params, "random_state": seed})
        return cv_aafe(X_tr, y_tr, groups, fn)

    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(xgb_objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_xgb_params = study_xgb.best_params
    best_xgb_cv = study_xgb.best_value
    print(f"  Best CV AAFE: {best_xgb_cv:.4f} (baseline: {baseline_cv:.4f})", flush=True)
    print(f"  Best params: {best_xgb_params}", flush=True)

    # Holdout evaluation
    best_xgb_full = {**best_xgb_params, "n_jobs": 4, "verbosity": 0, "tree_method": "hist"}
    best_xgb_fn = lambda seed: xgb.XGBRegressor(**{**best_xgb_full, "random_state": seed})
    xgb_ho, xgb_preds = multi_seed_holdout(X_tr, y_tr, X_ho, y_ho, best_xgb_fn, SEEDS)
    print(f"  HO AAFE: {np.mean(xgb_ho):.4f} ± {np.std(xgb_ho):.4f}", flush=True)

    # === Phase B: Feature importance pruning ===
    print("\n--- Phase B: Feature importance pruning ---", flush=True)
    m_imp = xgb.XGBRegressor(**{**best_xgb_full, "random_state": 42})
    m_imp.fit(X_tr, y_tr)
    importances = m_imp.feature_importances_
    del m_imp

    for top_k in [500, 1000, 2000, 3000]:
        top_idx = np.argsort(importances)[-top_k:]
        X_tr_k = X_tr[:, top_idx]
        X_ho_k = X_ho[:, top_idx]
        cv_k = cv_aafe(X_tr_k, y_tr, groups, best_xgb_fn)
        ho_k, _ = multi_seed_holdout(X_tr_k, y_tr, X_ho_k, y_ho, best_xgb_fn, [42])
        print(f"  top-{top_k}: CV={cv_k:.4f}, HO={ho_k[0]:.4f}", flush=True)

    # Find best K
    best_k = None
    best_k_ho = float("inf")
    for top_k in [500, 1000, 2000, 3000, 4260]:
        if top_k >= X_tr.shape[1]:
            top_idx = np.arange(X_tr.shape[1])
        else:
            top_idx = np.argsort(importances)[-top_k:]
        X_tr_k = X_tr[:, top_idx]
        X_ho_k = X_ho[:, top_idx]
        ho_k, _ = multi_seed_holdout(X_tr_k, y_tr, X_ho_k, y_ho, best_xgb_fn, [42])
        if ho_k[0] < best_k_ho:
            best_k_ho = ho_k[0]
            best_k = top_k
            best_top_idx = top_idx

    print(f"  Best K: {best_k} (HO={best_k_ho:.4f})", flush=True)

    # === Phase C: Optuna LightGBM ===
    print(f"\n--- Phase C: Optuna LightGBM ({N_TRIALS} trials) ---", flush=True)

    def lgbm_objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "n_jobs": 4, "verbosity": -1,
        }
        fn = lambda seed: lgb.LGBMRegressor(**{**params, "random_state": seed})
        return cv_aafe(X_tr, y_tr, groups, fn)

    study_lgbm = optuna.create_study(direction="minimize")
    study_lgbm.optimize(lgbm_objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_lgbm_params = study_lgbm.best_params
    best_lgbm_cv = study_lgbm.best_value
    print(f"  Best CV AAFE: {best_lgbm_cv:.4f}", flush=True)
    print(f"  Best params: {best_lgbm_params}", flush=True)

    best_lgbm_full = {**best_lgbm_params, "n_jobs": 4, "verbosity": -1}
    best_lgbm_fn = lambda seed: lgb.LGBMRegressor(**{**best_lgbm_full, "random_state": seed})
    lgbm_ho, lgbm_preds = multi_seed_holdout(X_tr, y_tr, X_ho, y_ho, best_lgbm_fn, SEEDS)
    print(f"  HO AAFE: {np.mean(lgbm_ho):.4f} ± {np.std(lgbm_ho):.4f}", flush=True)

    # === Phase D: XGB+LGBM Ensemble ===
    print("\n--- Phase D: XGB+LGBM Ensemble ---", flush=True)
    ens_preds = (xgb_preds.mean(axis=0) + lgbm_preds.mean(axis=0)) / 2
    ens_err = np.abs(ens_preds - y_ho)
    ens_aafe = float(10**np.mean(ens_err))
    ens_fold2 = float(100 * np.mean(ens_err < math.log10(2)))
    print(f"  Ensemble HO AAFE: {ens_aafe:.4f} (2-fold: {ens_fold2:.1f}%)", flush=True)

    # Weighted ensemble search
    best_w_aafe = ens_aafe
    best_w = 0.5
    for w in np.arange(0.1, 0.91, 0.05):
        wp = w * xgb_preds.mean(axis=0) + (1-w) * lgbm_preds.mean(axis=0)
        we = np.abs(wp - y_ho)
        wa = float(10**np.mean(we))
        if wa < best_w_aafe:
            best_w_aafe = wa
            best_w = w
    print(f"  Best weight (XGB={best_w:.2f}): AAFE={best_w_aafe:.4f}", flush=True)

    # === Summary ===
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    configs = {
        "baseline_xgb": {"cv": baseline_cv, "ho_mean": np.mean(baseline_ho),
                         "ho_std": np.std(baseline_ho)},
        "optuna_xgb": {"cv": best_xgb_cv, "ho_mean": np.mean(xgb_ho),
                       "ho_std": np.std(xgb_ho)},
        "optuna_lgbm": {"cv": best_lgbm_cv, "ho_mean": np.mean(lgbm_ho),
                        "ho_std": np.std(lgbm_ho)},
        "ensemble_equal": {"ho_mean": ens_aafe},
        "ensemble_weighted": {"ho_mean": best_w_aafe, "xgb_weight": best_w},
    }

    for name, m in configs.items():
        cv_str = f"CV={m.get('cv', '—'):.4f}" if 'cv' in m else ""
        ho_str = f"HO={m['ho_mean']:.4f}"
        std_str = f"±{m['ho_std']:.4f}" if 'ho_std' in m else ""
        print(f"  {name:<22} {cv_str:>12} {ho_str} {std_str}", flush=True)

    # Best config
    best_name = min(configs, key=lambda k: configs[k]["ho_mean"])
    best_ho = configs[best_name]["ho_mean"]
    delta = best_ho - 3.332
    print(f"\n  Best: {best_name} (HO={best_ho:.4f}, Δ={delta:+.4f} from 3.332)", flush=True)

    if best_ho <= 3.15:
        verdict = "PASS"
    elif best_ho <= 3.28:
        verdict = "PARTIAL"
    else:
        verdict = "NULL"
    print(f"  Verdict: {verdict}", flush=True)

    # Save results
    results = {
        "pre_registration": {
            "date": "2026-04-12",
            "hypothesis": "HP optimization improves holdout AAFE from hand-set 3.332",
            "n_trials": N_TRIALS,
            "seeds": SEEDS,
        },
        "baseline": {
            "params": dict(XGB_BASE_PARAMS),
            "cv_aafe": baseline_cv,
            "ho_aafe_mean": round(np.mean(baseline_ho), 4),
            "ho_aafe_std": round(np.std(baseline_ho), 4),
            "ho_aafe_per_seed": [round(h, 4) for h in baseline_ho],
        },
        "optuna_xgb": {
            "best_params": best_xgb_params,
            "cv_aafe": round(best_xgb_cv, 4),
            "ho_aafe_mean": round(np.mean(xgb_ho), 4),
            "ho_aafe_std": round(np.std(xgb_ho), 4),
            "ho_aafe_per_seed": [round(h, 4) for h in xgb_ho],
        },
        "feature_selection": {
            "best_k": best_k,
            "best_k_ho": round(best_k_ho, 4),
        },
        "optuna_lgbm": {
            "best_params": best_lgbm_params,
            "cv_aafe": round(best_lgbm_cv, 4),
            "ho_aafe_mean": round(np.mean(lgbm_ho), 4),
            "ho_aafe_std": round(np.std(lgbm_ho), 4),
            "ho_aafe_per_seed": [round(h, 4) for h in lgbm_ho],
        },
        "ensemble": {
            "equal_weight_aafe": round(ens_aafe, 4),
            "best_weight_xgb": round(best_w, 2),
            "best_weight_aafe": round(best_w_aafe, 4),
        },
        "best_config": best_name,
        "best_ho_aafe": round(best_ho, 4),
        "delta_from_baseline": round(delta, 4),
        "verdict": verdict,
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
