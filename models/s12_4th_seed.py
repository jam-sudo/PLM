"""S12 4th-seed replication — adds seed=7 to existing 3-seed S12 results.

Purpose: S12's ΔHO=−0.045 was 2.4σ with 3 seeds. Adding a 4th seed
increases paired t-test power to determine if PARTIAL → PASS or regresses.

Design: identical pipeline to s12_v12_retrain.py, single seed=7.
Results appended to existing s12_v12_results.json and re-aggregated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jam/PLM")

import sys
sys.path.insert(0, str(ROOT / "models"))
from s12_v12_retrain import (
    ADMEEncoder, build_features, build_holdout, run_config,
    XGB_BASE_PARAMS, ROOT,
)

NEW_SEED = 7


def main():
    print("S12 4th-seed replication (seed=7)")
    print("=" * 60)

    v11 = json.load(open(ROOT / "data/curated/plm_dataset_v11_llm.json"))
    v12 = json.load(open(ROOT / "data/curated/plm_dataset_v12_chembl.json"))
    ho = json.load(open(ROOT / "data/validation/holdout_definition.json"))
    tdc = {k[:14]: v for k, v in json.load(open(ROOT / "data/curated/tdc_adme_data.json")).items()}
    ho_iks = set((k or "")[:14] for k in ho["holdout_inchikeys"])

    print(f"v11 rows: {len(v11)}")
    print(f"v12 rows: {len(v12)} (+{len(v12)-len(v11)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ADMEEncoder().to(device)
    state = torch.load(ROOT / "models/adme_encoder.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()

    print("\nBuilding features A (v11 baseline)...")
    X_A, y_A, g_A = build_features(v11, tdc, ho_iks, encoder, device)
    print(f"  A: {X_A.shape}")

    print("Building features B (v12 expanded)...")
    X_B, y_B, g_B = build_features(v12, tdc, ho_iks, encoder, device)
    print(f"  B: {X_B.shape}")

    print("Building holdout features...")
    X_ho, y_ho, names = build_holdout(ho, tdc, encoder, device)
    print(f"  Holdout: {X_ho.shape}")

    print(f"\n--- Run A (v11 baseline, seed={NEW_SEED}) ---")
    res_A = run_config("v11", X_A, y_A, g_A, X_ho, y_ho, [NEW_SEED])

    print(f"\n--- Run B (v12 + ChEMBL, seed={NEW_SEED}) ---")
    res_B = run_config("v12", X_B, y_B, g_B, X_ho, y_ho, [NEW_SEED])

    # Load existing results and append
    existing_path = ROOT / "models/b1/s12_v12_results.json"
    existing = json.loads(existing_path.read_text())

    existing["v11"].append(res_A[0])
    existing["v12"].append(res_B[0])

    # Re-aggregate with 4 seeds
    ho_A = np.array([r["holdout"]["aafe"] for r in existing["v11"]])
    ho_B = np.array([r["holdout"]["aafe"] for r in existing["v12"]])
    cv_A = np.array([r["cv"]["aafe"] for r in existing["v11"]])
    cv_B = np.array([r["cv"]["aafe"] for r in existing["v12"]])

    delta = ho_B - ho_A
    paired_mean = float(delta.mean())
    paired_std = float(delta.std(ddof=1))
    t_stat = paired_mean / (paired_std / math.sqrt(len(delta)))
    # One-sided p-value (H0: delta >= 0, H1: delta < 0)
    from scipy import stats
    p_value = float(stats.t.cdf(t_stat, df=len(delta)-1))

    print("\n" + "=" * 60)
    print("4-SEED AGGREGATE")
    print("=" * 60)
    print(f"Seeds: {[r['seed'] for r in existing['v11']]}")
    print(f"v11 HO: {ho_A.mean():.4f} ± {ho_A.std():.4f}")
    print(f"v12 HO: {ho_B.mean():.4f} ± {ho_B.std():.4f}")
    print(f"ΔHO (v12-v11): {paired_mean:+.4f} ± {paired_std:.4f}")
    print(f"t-stat: {t_stat:.3f}, one-sided p: {p_value:.4f}")
    print(f"Per-seed deltas: {[f'{d:+.4f}' for d in delta]}")
    print(f"CV-HO gaps v11: {[r['cv_ho_gap'] for r in existing['v11']]}")
    print(f"CV-HO gaps v12: {[r['cv_ho_gap'] for r in existing['v12']]}")

    if paired_mean <= -0.05:
        verdict = "PASS"
    elif paired_mean <= -0.02:
        verdict = "PARTIAL"
    elif paired_mean < 0.05:
        verdict = "NULL"
    else:
        verdict = "HARM"
    print(f"Verdict: {verdict} (p={p_value:.4f})")

    # Update summary
    existing["pre_registration"]["seeds"] = [r["seed"] for r in existing["v11"]]
    existing["summary"] = {
        "v11_rows": len(v11), "v12_rows": len(v12),
        "cv_v11_mean": float(cv_A.mean()), "cv_v12_mean": float(cv_B.mean()),
        "ho_v11_mean": float(ho_A.mean()), "ho_v12_mean": float(ho_B.mean()),
        "delta_ho_mean": paired_mean,
        "delta_ho_std": paired_std,
        "t_stat": t_stat,
        "p_value_one_sided": p_value,
        "n_seeds": len(delta),
    }
    existing["verdict"] = f"{verdict} (4 seeds, p={p_value:.4f})"
    existing_path.write_text(json.dumps(existing, indent=2, default=float))
    print(f"\nUpdated {existing_path}")


if __name__ == "__main__":
    main()
