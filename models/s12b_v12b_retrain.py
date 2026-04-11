"""S12b — Retrain on v12b (v11 + ChEMBL v3 refined after audit).

Addresses audit finding: v12 (ChEMBL v2 strict) had 20% contamination
(metabolite Cmax + multi-dose SS). v12b uses v3 refined (107 clean rows).

Compares v12b (107 clean) vs v12 (164 contaminated) vs v11 baseline.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import GroupKFold

# Reuse everything from s12_v12_retrain
import sys
sys.path.insert(0, str(Path(__file__).parent))
from s12_v12_retrain import (
    ADMEEncoder, build_features, build_holdout, run_config,
    XGB_BASE_PARAMS, SEEDS, ROOT
)


def main():
    print("S12b — v12b refined retrain")
    print("=" * 60)

    v11 = json.load(open(ROOT / "data/curated/plm_dataset_v11_llm.json"))
    v12b = json.load(open(ROOT / "data/curated/plm_dataset_v12b_chembl_refined.json"))
    ho = json.load(open(ROOT / "data/validation/holdout_definition.json"))
    tdc = {k[:14]: v for k, v in json.load(open(ROOT / "data/curated/tdc_adme_data.json")).items()}
    ho_iks = set((k or "")[:14] for k in ho["holdout_inchikeys"])

    print(f"v11 rows: {len(v11)}")
    print(f"v12b rows: {len(v12b)} (+{len(v12b)-len(v11)})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ADMEEncoder().to(device)
    state = torch.load(ROOT / "models/adme_encoder.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()

    print("\nBuilding features A (v11 baseline)...")
    X_A, y_A, g_A = build_features(v11, tdc, ho_iks, encoder, device)
    print(f"  A: {X_A.shape}")

    print("Building features B (v12b refined)...")
    X_B, y_B, g_B = build_features(v12b, tdc, ho_iks, encoder, device)
    print(f"  B: {X_B.shape}")

    print("Building holdout features...")
    X_ho, y_ho, names = build_holdout(ho, tdc, encoder, device)
    print(f"  Holdout: {X_ho.shape}")

    print("\n--- Run A (v11 baseline) ---")
    res_A = run_config("v11", X_A, y_A, g_A, X_ho, y_ho, SEEDS)

    print("\n--- Run B (v12b refined) ---")
    res_B = run_config("v12b", X_B, y_B, g_B, X_ho, y_ho, SEEDS)

    ho_A = np.array([r["holdout"]["aafe"] for r in res_A])
    ho_B = np.array([r["holdout"]["aafe"] for r in res_B])
    cv_A = np.array([r["cv"]["aafe"] for r in res_A])
    cv_B = np.array([r["cv"]["aafe"] for r in res_B])

    delta = ho_B - ho_A
    paired_mean = float(delta.mean())
    paired_std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0

    print("\n" + "=" * 60)
    print("AGGREGATE (S12b v12b refined)")
    print("=" * 60)
    print(f"v11  CV: {cv_A.mean():.3f}±{cv_A.std():.3f}  HO: {ho_A.mean():.3f}±{ho_A.std():.3f}")
    print(f"v12b CV: {cv_B.mean():.3f}±{cv_B.std():.3f}  HO: {ho_B.mean():.3f}±{ho_B.std():.3f}")
    print(f"ΔHO (v12b-v11): {paired_mean:+.4f} ± {paired_std:.4f}")

    # Compare to S12 original
    s12_path = ROOT / "models/b1/s12_v12_results.json"
    if s12_path.exists():
        s12 = json.loads(s12_path.read_text())
        s12_delta = s12["summary"]["delta_ho_mean"]
        print(f"\nS12 original (v12 164 rows):  ΔHO = {s12_delta:+.4f}")
        print(f"S12b refined (v12b 107 rows): ΔHO = {paired_mean:+.4f}")
        print(f"Refinement impact: {paired_mean - s12_delta:+.4f}")

    if paired_mean <= -0.05:
        verdict = "PASS"
    elif paired_mean <= -0.02:
        verdict = "PARTIAL"
    elif paired_mean < 0.05:
        verdict = "NULL"
    else:
        verdict = "HARM"
    print(f"Verdict: {verdict}")

    out_path = ROOT / "models/b1/s12b_v12b_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out = {
        "hypothesis": "v12b (v11 + 107 audit-refined ChEMBL rows) improves HO vs v11 baseline",
        "v11": res_A, "v12b": res_B,
        "summary": {
            "v11_rows": len(v11), "v12b_rows": len(v12b),
            "cv_v11_mean": float(cv_A.mean()), "cv_v12b_mean": float(cv_B.mean()),
            "ho_v11_mean": float(ho_A.mean()), "ho_v12b_mean": float(ho_B.mean()),
            "delta_ho_mean": paired_mean, "delta_ho_std": paired_std,
        },
        "verdict": verdict,
    }
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
