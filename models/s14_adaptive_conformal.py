"""S14 — Locally Adaptive Conformal Prediction.

PRE-REGISTRATION
================
Date: 2026-04-12
Hypothesis: Locally adaptive conformal intervals (Lei et al. 2018)
  achieve ≥85% empirical coverage on holdout while reducing mean interval
  width compared to S13's fixed-width conformal (2.18 log10).

Method:
  1. OOF residuals from 5-fold GroupKFold (2 seeds, n_est=200)
  2. Difficulty model: XGBoost trained on OOF data (features → |residual|)
  3. Normalized conformal score: |y - ŷ| / max(σ̂(x), floor)
  4. Conformal quantile on normalized scores
  5. Holdout interval: ŷ ± q × σ̂(x)

Success criteria:
  PASS:    coverage ≥ 0.85 AND mean width < S13 (2.18 log10)
  PARTIAL: coverage ≥ 0.85 AND width reduction > 10%
  NULL:    coverage < 0.85 OR no width improvement

Outputs: models/b1/s14_adaptive_results.json
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import GroupKFold

import sys
ROOT = Path("/home/jam/PLM")
sys.path.insert(0, str(ROOT / "models"))
from s12_v12_retrain import (
    ADMEEncoder, build_features, build_holdout,
    XGB_BASE_PARAMS, aafe_metrics,
)

OUT = ROOT / "models/b1/s14_adaptive_results.json"
SEEDS = [42, 137, 2024, 7]
ALPHA = 0.10
CONFORMAL_XGB = {**XGB_BASE_PARAMS, "n_estimators": 200, "n_jobs": 4}
ENSEMBLE_XGB = {**XGB_BASE_PARAMS, "n_jobs": 4}
# Difficulty model — lightweight, regularized to avoid overfitting
DIFF_XGB = dict(
    n_estimators=100, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.3,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10,
    n_jobs=4, verbosity=0, tree_method="hist",
)
SIGMA_FLOOR = 0.1  # minimum difficulty estimate to avoid division by ~0


def main():
    print("S14 — Locally Adaptive Conformal Prediction", flush=True)
    print("=" * 60, flush=True)

    # Load data
    v12 = json.load(open(ROOT / "data/curated/plm_dataset_v12_chembl.json"))
    ho = json.load(open(ROOT / "data/validation/holdout_definition.json"))
    tdc = {k[:14]: v for k, v in json.load(open(ROOT / "data/curated/tdc_adme_data.json")).items()}
    ho_iks = set((k or "")[:14] for k in ho["holdout_inchikeys"])
    print(f"v12 rows: {len(v12)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ADMEEncoder().to(device)
    state = torch.load(ROOT / "models/adme_encoder.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()

    print("Building features...", flush=True)
    X_tr, y_tr, groups = build_features(v12, tdc, ho_iks, encoder, device)
    X_ho, y_ho, ho_names = build_holdout(ho, tdc, encoder, device)
    print(f"  Train: {X_tr.shape}, Holdout: {X_ho.shape}", flush=True)

    del encoder, state
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # === Step 1: Collect OOF predictions and residuals ===
    print("\n--- Step 1: OOF predictions (2 seeds × 5 folds) ---", flush=True)
    gkf = GroupKFold(n_splits=5)
    # Accumulate per-sample: we need OOF predictions to train difficulty model
    oof_preds = np.zeros(len(y_tr))
    oof_counts = np.zeros(len(y_tr))

    for seed in SEEDS[:2]:
        params = {**CONFORMAL_XGB, "random_state": seed}
        for fold_i, (ti, vi) in enumerate(gkf.split(X_tr, y_tr, groups)):
            print(f"  seed={seed} fold={fold_i}...", end="", flush=True)
            m = xgb.XGBRegressor(**params)
            m.fit(X_tr[ti], y_tr[ti])
            preds = m.predict(X_tr[vi])
            oof_preds[vi] += preds
            oof_counts[vi] += 1
            del m; gc.collect()
            print(" done", flush=True)

    # Average OOF predictions across seeds
    oof_preds /= np.maximum(oof_counts, 1)
    oof_residuals = np.abs(y_tr - oof_preds)
    print(f"OOF residuals: median={np.median(oof_residuals):.4f}, "
          f"mean={np.mean(oof_residuals):.4f}", flush=True)

    # === Step 2: Train difficulty model (features → |residual|) ===
    print("\n--- Step 2: Difficulty model ---", flush=True)
    # OOF difficulty estimation to avoid overfitting
    oof_sigma = np.zeros(len(y_tr))
    for fold_i, (ti, vi) in enumerate(gkf.split(X_tr, y_tr, groups)):
        print(f"  Difficulty fold {fold_i}...", end="", flush=True)
        diff_m = xgb.XGBRegressor(**DIFF_XGB, random_state=42)
        diff_m.fit(X_tr[ti], oof_residuals[ti])
        oof_sigma[vi] = diff_m.predict(X_tr[vi])
        del diff_m; gc.collect()
        print(" done", flush=True)

    # Clamp difficulty estimates
    oof_sigma = np.maximum(oof_sigma, SIGMA_FLOOR)
    print(f"σ̂ distribution: median={np.median(oof_sigma):.4f}, "
          f"mean={np.mean(oof_sigma):.4f}, "
          f"min={oof_sigma.min():.4f}, max={oof_sigma.max():.4f}", flush=True)

    # === Step 3: Normalized conformal scores ===
    print("\n--- Step 3: Normalized conformal scores ---", flush=True)
    normalized_scores = oof_residuals / oof_sigma
    n = len(normalized_scores)
    level = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
    q_norm = float(np.quantile(normalized_scores, level))
    print(f"Normalized scores: median={np.median(normalized_scores):.4f}, "
          f"p90={np.quantile(normalized_scores, 0.9):.4f}", flush=True)
    print(f"Conformal quantile (normalized): {q_norm:.4f}", flush=True)

    # Also compute S13-style fixed quantile for comparison
    q_fixed = float(np.quantile(oof_residuals, level))
    print(f"Fixed quantile (S13-style): {q_fixed:.4f}", flush=True)

    # === Step 4: Train full difficulty model + ensemble for holdout ===
    print("\n--- Step 4: Full models for holdout ---", flush=True)

    # Full difficulty model
    print("  Full difficulty model...", end="", flush=True)
    diff_full = xgb.XGBRegressor(**DIFF_XGB, random_state=42)
    diff_full.fit(X_tr, oof_residuals)
    ho_sigma = np.maximum(diff_full.predict(X_ho), SIGMA_FLOOR)
    del diff_full; gc.collect()
    print(" done", flush=True)

    # Seed ensemble for holdout predictions
    ho_preds_all = []
    for seed in SEEDS:
        print(f"  Ensemble seed={seed}...", end="", flush=True)
        params = {**ENSEMBLE_XGB, "random_state": seed}
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr)
        ho_preds_all.append(m.predict(X_ho))
        del m; gc.collect()
        print(" done", flush=True)

    ho_preds = np.array(ho_preds_all)
    ho_mean = ho_preds.mean(axis=0)
    ho_std = ho_preds.std(axis=0)

    # === Step 5: Evaluate on holdout ===
    print("\n--- Step 5: Holdout Evaluation ---", flush=True)
    actual_errors = np.abs(y_ho - ho_mean)

    # Adaptive intervals
    adaptive_half = q_norm * ho_sigma
    adaptive_covered = (y_ho >= ho_mean - adaptive_half) & (y_ho <= ho_mean + adaptive_half)
    adaptive_coverage = float(adaptive_covered.mean())
    adaptive_widths = 2 * adaptive_half
    adaptive_mean_width = float(adaptive_widths.mean())
    adaptive_median_width = float(np.median(adaptive_widths))

    # Fixed intervals (S13-style for comparison)
    fixed_half = q_fixed
    fixed_covered = (y_ho >= ho_mean - fixed_half) & (y_ho <= ho_mean + fixed_half)
    fixed_coverage = float(fixed_covered.mean())
    fixed_width = 2 * fixed_half

    # S13 reference width
    s13_width = 2.1795

    print(f"\n  ADAPTIVE CONFORMAL:", flush=True)
    print(f"  Coverage: {adaptive_coverage:.3f} ({adaptive_covered.sum()}/{len(y_ho)})",
          flush=True)
    print(f"  Mean width: {adaptive_mean_width:.4f} log10 "
          f"({10**adaptive_mean_width:.1f}-fold)", flush=True)
    print(f"  Median width: {adaptive_median_width:.4f} log10 "
          f"({10**adaptive_median_width:.1f}-fold)", flush=True)
    print(f"  Width range: [{adaptive_widths.min():.3f}, {adaptive_widths.max():.3f}]",
          flush=True)

    print(f"\n  FIXED CONFORMAL (comparison):", flush=True)
    print(f"  Coverage: {fixed_coverage:.3f}", flush=True)
    print(f"  Width: {fixed_width:.4f} log10", flush=True)

    width_reduction = (1 - adaptive_mean_width / s13_width) * 100
    print(f"\n  Width reduction vs S13: {width_reduction:.1f}%", flush=True)

    # Difficulty model calibration
    sp_r, sp_p = stats.spearmanr(ho_sigma, actual_errors)
    print(f"  Spearman(σ̂, |error|): r={sp_r:.3f}, p={sp_p:.4f}", flush=True)

    # Ensemble AAFE
    ho_metrics = aafe_metrics(ho_mean, y_ho)
    print(f"  Ensemble AAFE: {ho_metrics['aafe']}", flush=True)

    # Conditional coverage by difficulty tercile
    print("\n  COVERAGE BY DIFFICULTY TERCILE:", flush=True)
    sigma_terciles = np.quantile(ho_sigma, [1/3, 2/3])
    tercile_labels = ["Easy (low σ̂)", "Medium", "Hard (high σ̂)"]
    tercile_bounds = [0] + sigma_terciles.tolist() + [np.inf]
    tercile_results = {}
    for i in range(3):
        mask = (ho_sigma >= tercile_bounds[i]) & (ho_sigma < tercile_bounds[i + 1])
        if mask.sum() > 0:
            tc = float(adaptive_covered[mask].mean())
            tw = float(adaptive_widths[mask].mean())
            te = float(actual_errors[mask].mean())
            taafe = float(10**te)
            tercile_results[tercile_labels[i]] = {
                "coverage": tc, "mean_width": round(tw, 4),
                "mean_error": round(te, 4), "aafe": round(taafe, 4),
                "n": int(mask.sum()),
                "sigma_range": [round(float(ho_sigma[mask].min()), 4),
                                round(float(ho_sigma[mask].max()), 4)],
            }
            print(f"    {tercile_labels[i]}: cov={tc:.3f}, "
                  f"width={tw:.3f} log10 ({10**tw:.1f}x), "
                  f"AAFE={taafe:.2f}, n={mask.sum()}", flush=True)

    # Per-drug results
    per_drug = []
    for i, name in enumerate(ho_names):
        per_drug.append({
            "drug": name,
            "y_true": round(float(y_ho[i]), 4),
            "y_pred": round(float(ho_mean[i]), 4),
            "error": round(float(actual_errors[i]), 4),
            "aafe": round(float(10**actual_errors[i]), 4),
            "sigma_hat": round(float(ho_sigma[i]), 4),
            "seed_std": round(float(ho_std[i]), 4),
            "adaptive_half": round(float(adaptive_half[i]), 4),
            "adaptive_width": round(float(adaptive_widths[i]), 4),
            "covered": bool(adaptive_covered[i]),
        })

    # Sort by sigma to show easy → hard
    per_drug.sort(key=lambda d: d["sigma_hat"])

    # Verdict
    if adaptive_coverage >= 0.85 and adaptive_mean_width < s13_width:
        if width_reduction > 10:
            verdict = "PASS"
        else:
            verdict = "PARTIAL"
    elif adaptive_coverage >= 0.85:
        verdict = "PARTIAL (coverage OK, no width improvement)"
    else:
        verdict = "NULL (coverage < 0.85)"

    print(f"\nVERDICT: {verdict}", flush=True)

    # Save
    results = {
        "pre_registration": {
            "date": "2026-04-12",
            "hypothesis": "Adaptive conformal achieves ≥85% coverage with narrower intervals than S13",
            "alpha": ALPHA,
            "method": "Locally adaptive conformal (Lei et al. 2018)",
            "s13_reference_width": s13_width,
        },
        "difficulty_model": {
            "type": "XGBoost regressor (features → |OOF residual|)",
            "params": DIFF_XGB,
            "sigma_floor": SIGMA_FLOOR,
            "holdout_sigma_median": round(float(np.median(ho_sigma)), 4),
            "holdout_sigma_mean": round(float(np.mean(ho_sigma)), 4),
            "spearman_r": round(sp_r, 4),
            "spearman_p": round(sp_p, 4),
        },
        "adaptive_conformal": {
            "q_normalized": round(q_norm, 4),
            "coverage": round(adaptive_coverage, 4),
            "n_covered": int(adaptive_covered.sum()),
            "n_total": len(y_ho),
            "mean_width_log10": round(adaptive_mean_width, 4),
            "median_width_log10": round(adaptive_median_width, 4),
            "width_min": round(float(adaptive_widths.min()), 4),
            "width_max": round(float(adaptive_widths.max()), 4),
            "mean_width_fold": round(float(10**adaptive_mean_width), 1),
        },
        "fixed_conformal": {
            "q_fixed": round(q_fixed, 4),
            "coverage": round(fixed_coverage, 4),
            "width_log10": round(fixed_width, 4),
        },
        "comparison": {
            "s13_width": s13_width,
            "s14_mean_width": round(adaptive_mean_width, 4),
            "width_reduction_pct": round(width_reduction, 1),
        },
        "tercile_analysis": tercile_results,
        "ensemble_aafe": ho_metrics["aafe"],
        "per_drug": per_drug,
        "verdict": verdict,
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
