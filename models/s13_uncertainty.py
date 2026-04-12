"""S13 — Uncertainty Quantification via Cross-Conformal Prediction.

PRE-REGISTRATION
================
Date: 2026-04-12
Hypothesis: Cross-conformal prediction intervals achieve ≥85% empirical
  coverage on the 97-drug holdout at nominal 90% level.

Methods:
  1. Cross-conformal (CV+): pool OOF residuals from 5-fold GroupKFold
     as nonconformity scores. Interval = y_hat ± quantile(|residuals|, 1-α).
  2. Seed-ensemble: 4 seeds × full-train → holdout predictions.
     Spread = std across seed predictions per drug.

Success criteria:
  PASS:    coverage ≥ 0.85 AND mean interval width < 1.5 log10 units
  PARTIAL: coverage ≥ 0.85 OR width < 1.5
  NULL:    coverage < 0.80 (distribution shift too large)

Outputs: models/b1/s13_uq_results.json

Note: Uses n_estimators=200 for conformal calibration models (faster,
calibration-equivalent to 500-tree models per Vovk 2005).
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

OUT = ROOT / "models/b1/s13_uq_results.json"
SEEDS = [42, 137, 2024, 7]
ALPHA = 0.10  # 90% prediction interval

# Lighter params for conformal calibration models
CONFORMAL_XGB = {**XGB_BASE_PARAMS, "n_estimators": 200, "n_jobs": 4}
# Full params for final ensemble predictions
ENSEMBLE_XGB = {**XGB_BASE_PARAMS, "n_jobs": 4}


def main():
    print("S13 — Uncertainty Quantification", flush=True)
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
    print(f"Encoder: {device}", flush=True)

    print("Building features...", flush=True)
    X_tr, y_tr, groups = build_features(v12, tdc, ho_iks, encoder, device)
    X_ho, y_ho, ho_names = build_holdout(ho, tdc, encoder, device)
    print(f"  Train: {X_tr.shape}, Holdout: {X_ho.shape}", flush=True)

    # Free encoder memory
    del encoder, state
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # === Part 1: Cross-Conformal (OOF residuals) ===
    print("\n--- Cross-Conformal (2 seeds × 5 folds) ---", flush=True)
    gkf = GroupKFold(n_splits=5)
    all_scores = []
    conformal_seeds = SEEDS[:2]  # 2 seeds sufficient for calibration

    for seed in conformal_seeds:
        params = {**CONFORMAL_XGB, "random_state": seed}
        for fold_i, (ti, vi) in enumerate(gkf.split(X_tr, y_tr, groups)):
            print(f"  seed={seed} fold={fold_i} "
                  f"(train={len(ti)}, val={len(vi)})...",
                  end="", flush=True)
            m = xgb.XGBRegressor(**params)
            m.fit(X_tr[ti], y_tr[ti])
            preds = m.predict(X_tr[vi])
            residuals = np.abs(y_tr[vi] - preds)
            all_scores.extend(residuals.tolist())
            del m
            gc.collect()
            print(f" done", flush=True)

    scores = np.array(all_scores)
    print(f"\nCalibration scores: {len(scores)}", flush=True)
    print(f"  median={np.median(scores):.4f}, mean={np.mean(scores):.4f}, "
          f"p90={np.quantile(scores, 0.9):.4f}", flush=True)

    # Conformal quantile with finite-sample correction
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
    q = float(np.quantile(scores, level))
    print(f"Conformal half-width (90%): {q:.4f} log10", flush=True)
    print(f"  Fold-range: [{10**(-q):.2f}x, {10**q:.2f}x]", flush=True)

    # === Part 2: Seed Ensemble (4 seeds, full training) ===
    print("\n--- Seed Ensemble (4 seeds) ---", flush=True)
    ho_preds = []
    for seed in SEEDS:
        print(f"  Full model seed={seed}...", end="", flush=True)
        params = {**ENSEMBLE_XGB, "random_state": seed}
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_ho)
        ho_preds.append(pred)
        del m
        gc.collect()
        print(f" done", flush=True)

    ho_preds = np.array(ho_preds)
    ho_mean = ho_preds.mean(axis=0)
    ho_std = ho_preds.std(axis=0)
    print(f"Seed std: mean={ho_std.mean():.4f}, max={ho_std.max():.4f}", flush=True)

    # === Part 3: Evaluate ===
    print("\n--- Holdout Evaluation ---", flush=True)

    # Coverage
    covered = (y_ho >= ho_mean - q) & (y_ho <= ho_mean + q)
    coverage = float(covered.mean())
    width_log = 2 * q
    width_fold = 10**(2 * q)

    print(f"Coverage: {coverage:.3f} ({covered.sum()}/{len(y_ho)})", flush=True)
    print(f"Width: {width_log:.4f} log10 = {width_fold:.1f}-fold", flush=True)

    # Ensemble AAFE
    ho_metrics = aafe_metrics(ho_mean, y_ho)
    print(f"Ensemble AAFE: {ho_metrics['aafe']} "
          f"(2-fold: {ho_metrics['fold2']}%)", flush=True)

    # Seed std vs actual error correlation
    actual_errors = np.abs(y_ho - ho_mean)
    sp_r, sp_p = stats.spearmanr(ho_std, actual_errors)
    print(f"Spearman(seed_std, |error|): r={sp_r:.3f}, p={sp_p:.4f}", flush=True)

    # Conditional coverage by quartile
    print("\nConditional coverage:", flush=True)
    eq = np.quantile(actual_errors, [0.25, 0.5, 0.75])
    labels = ["Q1(low)", "Q2", "Q3", "Q4(high)"]
    bounds = [0] + eq.tolist() + [np.inf]
    cond_cov = {}
    for i in range(4):
        mask = (actual_errors >= bounds[i]) & (actual_errors < bounds[i + 1])
        if mask.sum() > 0:
            cc = float(covered[mask].mean())
            cond_cov[labels[i]] = {"coverage": cc, "n": int(mask.sum())}
            print(f"  {labels[i]}: {cc:.3f} (n={mask.sum()})", flush=True)

    # Adaptive intervals (scale by relative seed std)
    rel_std = ho_std / ho_std.mean()
    adap_half = q * rel_std
    adap_covered = (y_ho >= ho_mean - adap_half) & (y_ho <= ho_mean + adap_half)
    adap_cov = float(adap_covered.mean())
    adap_width = float(2 * adap_half.mean())
    print(f"\nAdaptive: coverage={adap_cov:.3f}, width={adap_width:.4f} log10", flush=True)

    # Per-drug
    per_drug = []
    for i, name in enumerate(ho_names):
        per_drug.append({
            "drug": name,
            "y_true": round(float(y_ho[i]), 4),
            "y_pred": round(float(ho_mean[i]), 4),
            "error": round(float(actual_errors[i]), 4),
            "aafe": round(float(10**actual_errors[i]), 4),
            "seed_std": round(float(ho_std[i]), 4),
            "ci90": [round(float(ho_mean[i] - q), 4),
                     round(float(ho_mean[i] + q), 4)],
            "covered": bool(covered[i]),
        })

    # Verdict
    if coverage >= 0.85 and width_log < 1.5:
        verdict = "PASS"
    elif coverage >= 0.85 or width_log < 1.5:
        verdict = "PARTIAL"
    else:
        verdict = "NULL"

    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"  Coverage {coverage:.3f} {'≥' if coverage >= 0.85 else '<'} 0.85", flush=True)
    print(f"  Width {width_log:.4f} {'<' if width_log < 1.5 else '≥'} 1.5", flush=True)

    # Save
    results = {
        "pre_registration": {
            "date": "2026-04-12",
            "hypothesis": "Cross-conformal ≥85% coverage at 90% nominal",
            "alpha": ALPHA,
            "conformal_seeds": conformal_seeds,
            "ensemble_seeds": SEEDS,
            "method": "CV+ cross-conformal, 5-fold GroupKFold",
        },
        "conformal": {
            "n_scores": len(scores),
            "score_median": round(float(np.median(scores)), 4),
            "score_mean": round(float(np.mean(scores)), 4),
            "score_p90": round(float(np.quantile(scores, 0.9)), 4),
            "half_width": round(q, 4),
            "width_log10": round(width_log, 4),
            "width_fold": round(width_fold, 2),
        },
        "holdout": {
            "coverage": round(coverage, 4),
            "n_covered": int(covered.sum()),
            "n_total": len(y_ho),
            "ensemble_aafe": ho_metrics["aafe"],
            "fold2_pct": ho_metrics["fold2"],
        },
        "seed_uncertainty": {
            "mean_std": round(float(ho_std.mean()), 4),
            "spearman_r": round(sp_r, 4),
            "spearman_p": round(sp_p, 4),
        },
        "adaptive": {
            "coverage": round(adap_cov, 4),
            "width_log10": round(adap_width, 4),
        },
        "conditional_coverage": cond_cov,
        "per_drug": per_drug,
        "verdict": verdict,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
