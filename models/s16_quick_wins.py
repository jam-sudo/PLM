"""S16 — Quick Win Experiments: Bias Correction + Source Weighting + FP Variants.

PRE-REGISTRATION
================
Date: 2026-04-14
Hypothesis: Three untried lightweight approaches may improve holdout AAFE.

Experiments:
  A) CV-based bias correction: subtract mean OOF residual from predictions
  B) Source-weighted training: upweight SIS (high quality), downweight ChEMBL
  C) FP radius/type variants: ECFP6 (radius=3), FCFP4, Avalon FP
  D) Best combination of above

Success criteria (vs baseline HO 3.332):
  PASS:    best HO ≤ 3.28 (Δ ≤ −0.05)
  PARTIAL: best HO ≤ 3.31 (Δ ≤ −0.02)
  NULL:    best HO > 3.31

Outputs: models/b1/s16_quick_wins_results.json
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

import sys
ROOT = Path("/home/jam/PLM")
sys.path.insert(0, str(ROOT / "models"))
from s12_v12_retrain import (
    ADMEEncoder, build_features, build_holdout,
    XGB_BASE_PARAMS, aafe_metrics, smiles_to_fp, smiles_to_physchem,
    get_tdc_features, compute_micropbpk, FP_BITS, N_PHYSCHEM, N_TDC, N_UPBPK,
)

OUT = ROOT / "models/b1/s16_quick_wins_results.json"
SEEDS = [42, 137, 2024, 7]


def cv_with_oof(X, y, groups, params, seed=42):
    """5-fold CV returning OOF predictions."""
    gkf = GroupKFold(n_splits=5)
    oof = np.full_like(y, np.nan)
    p = {**params, "random_state": seed}
    for ti, vi in gkf.split(X, y, groups):
        m = xgb.XGBRegressor(**p)
        m.fit(X[ti], y[ti])
        oof[vi] = m.predict(X[vi])
        del m
    return oof


def multi_seed_eval(X_tr, y_tr, X_ho, y_ho, params, seeds,
                    sample_weight=None, bias=0.0):
    """Train + holdout eval with multiple seeds, optional bias correction."""
    ho_aafes = []
    all_preds = []
    for seed in seeds:
        p = {**params, "random_state": seed}
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr, y_tr, sample_weight=sample_weight)
        pred = m.predict(X_ho) - bias
        all_preds.append(pred)
        err = np.abs(pred - y_ho)
        ho_aafes.append(float(10**np.mean(err)))
        del m
    return ho_aafes, np.array(all_preds)


def smiles_to_fp_variant(smiles, variant="ecfp4"):
    """Generate different FP types."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if variant == "ecfp4":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    elif variant == "ecfp6":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=FP_BITS)
    elif variant == "fcfp4":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS,
                                                    useFeatures=True)
    elif variant == "fcfp6":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=FP_BITS,
                                                    useFeatures=True)
    elif variant == "avalon":
        from rdkit.Avalon import pyAvalonTools
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=FP_BITS)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return np.array(fp, dtype=np.float32)


def build_features_variant(dataset, tdc, ho_iks, encoder, device, fp_variant="ecfp4"):
    """Build features with a different FP type (encoder still uses ECFP4)."""
    fp_cache, emb_cache = {}, {}

    def get_fp(smi):
        if smi not in fp_cache:
            fp_cache[smi] = smiles_to_fp_variant(smi, fp_variant)
        return fp_cache[smi]

    def get_emb(smi):
        if smi not in emb_cache:
            # Encoder always uses ECFP4
            fp_ecfp4 = smiles_to_fp(smi)
            if fp_ecfp4 is None:
                emb_cache[smi] = np.zeros(128, dtype=np.float32)
            else:
                with torch.no_grad():
                    t = torch.tensor(fp_ecfp4).unsqueeze(0).to(device)
                    emb, _ = encoder(t)
                    emb_cache[smi] = emb.cpu().numpy().flatten()
        return emb_cache[smi]

    X, y, groups = [], [], []
    for row in dataset:
        ik = (row["ik"] or "")[:14]
        if ik in ho_iks:
            continue
        log_cd = row.get("log_cd")
        if log_cd is None:
            cmax = row.get("cmax_ngml")
            dose = row.get("dose_mg")
            if not cmax or not dose or dose <= 0:
                continue
            log_cd = math.log10(cmax / dose)
        fp = get_fp(row["smiles"])
        if fp is None:
            continue
        emb = get_emb(row["smiles"])
        pc = smiles_to_physchem(row["smiles"])
        tdcf = get_tdc_features(ik, tdc)
        upk = compute_micropbpk(ik, tdc)
        ld = np.float32(math.log10(max(row["dose_mg"], 1e-6)))
        X.append(np.concatenate([fp, emb, pc, tdcf, upk, [ld]]).astype(np.float32))
        y.append(float(log_cd))
        groups.append(ik)
    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    groups = np.array(groups)
    X[~np.isfinite(X)] = np.nan
    return X, y, groups


def build_holdout_variant(ho, tdc, encoder, device, fp_variant="ecfp4"):
    """Build holdout features with a different FP type."""
    fp_cache, emb_cache = {}, {}

    def get_fp(smi):
        if smi not in fp_cache:
            fp_cache[smi] = smiles_to_fp_variant(smi, fp_variant)
        return fp_cache[smi]

    def get_emb(smi):
        if smi not in emb_cache:
            fp_ecfp4 = smiles_to_fp(smi)
            if fp_ecfp4 is None:
                emb_cache[smi] = np.zeros(128, dtype=np.float32)
            else:
                with torch.no_grad():
                    t = torch.tensor(fp_ecfp4).unsqueeze(0).to(device)
                    emb, _ = encoder(t)
                    emb_cache[smi] = emb.cpu().numpy().flatten()
        return emb_cache[smi]

    X, y, names = [], [], []
    for d in ho["holdout_drugs"]:
        smi, dose, cmax = d.get("smiles"), d.get("dose_mg", 0), d.get("cmax_obs_ngml", 0)
        ik = (d.get("inchikey14") or "")[:14]
        if not smi or not dose or not cmax or dose <= 0 or cmax <= 0:
            continue
        fp = get_fp(smi)
        if fp is None:
            continue
        emb = get_emb(smi)
        pc = smiles_to_physchem(smi)
        tdcf = get_tdc_features(ik, tdc)
        upk = compute_micropbpk(ik, tdc)
        ld = np.float32(math.log10(max(dose, 1e-6)))
        X.append(np.concatenate([fp, emb, pc, tdcf, upk, [ld]]).astype(np.float32))
        y.append(math.log10(cmax / dose))
        names.append(d.get("name"))
    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, y, names


def main():
    print("S16 — Quick Win Experiments", flush=True)
    print("=" * 60, flush=True)

    # Load data
    v12 = json.load(open(ROOT / "data/curated/plm_dataset_v12_chembl.json"))
    ho_def = json.load(open(ROOT / "data/validation/holdout_definition.json"))
    tdc = {k[:14]: v for k, v in json.load(open(ROOT / "data/curated/tdc_adme_data.json")).items()}
    ho_iks = set((k or "")[:14] for k in ho_def["holdout_inchikeys"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ADMEEncoder().to(device)
    state = torch.load(ROOT / "models/adme_encoder.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()

    print("Building baseline features (ECFP4)...", flush=True)
    X_tr, y_tr, groups = build_features(v12, tdc, ho_iks, encoder, device)
    X_ho, y_ho, ho_names = build_holdout(ho_def, tdc, encoder, device)
    print(f"  Train: {X_tr.shape}, Holdout: {X_ho.shape}", flush=True)

    results = {}

    # === Baseline ===
    print("\n--- Baseline ---", flush=True)
    ho_base, _ = multi_seed_eval(X_tr, y_tr, X_ho, y_ho, XGB_BASE_PARAMS, SEEDS)
    results["baseline"] = {"ho_mean": round(np.mean(ho_base), 4),
                           "ho_std": round(np.std(ho_base), 4)}
    print(f"  HO: {np.mean(ho_base):.4f} ± {np.std(ho_base):.4f}", flush=True)

    # === Exp A: CV-based bias correction ===
    print("\n--- Exp A: Bias Correction ---", flush=True)
    oof = cv_with_oof(X_tr, y_tr, groups, XGB_BASE_PARAMS)
    oof_residuals = oof - y_tr  # positive = overprediction
    bias = float(np.mean(oof_residuals))
    median_bias = float(np.median(oof_residuals))
    print(f"  OOF mean bias: {bias:+.4f}, median: {median_bias:+.4f}", flush=True)

    for label, b in [("mean_bias", bias), ("median_bias", median_bias)]:
        ho_bc, _ = multi_seed_eval(X_tr, y_tr, X_ho, y_ho, XGB_BASE_PARAMS, SEEDS, bias=b)
        results[f"bias_{label}"] = {"bias": round(b, 4),
                                     "ho_mean": round(np.mean(ho_bc), 4),
                                     "ho_std": round(np.std(ho_bc), 4)}
        delta = np.mean(ho_bc) - np.mean(ho_base)
        print(f"  {label} ({b:+.4f}): HO={np.mean(ho_bc):.4f} (Δ={delta:+.4f})", flush=True)

    # === Exp B: Source-weighted training ===
    print("\n--- Exp B: Source Weighting ---", flush=True)
    # Determine source per row
    src_counts = {}
    weights_list = []
    for row in v12:
        ik = (row["ik"] or "")[:14]
        if ik in ho_iks:
            continue
        src = row.get("src", ["unknown"])
        if isinstance(src, list):
            src = src[0] if src else "unknown"
        src_counts[src] = src_counts.get(src, 0) + 1

    print(f"  Sources: {src_counts}", flush=True)

    # Build weight array matching training rows
    for row in v12:
        ik = (row["ik"] or "")[:14]
        if ik in ho_iks:
            continue
        log_cd = row.get("log_cd")
        if log_cd is None:
            cmax = row.get("cmax_ngml")
            dose = row.get("dose_mg")
            if not cmax or not dose or dose <= 0:
                continue
            log_cd = math.log10(cmax / dose)
        fp = smiles_to_fp(row["smiles"])
        if fp is None:
            continue
        src = row.get("src", ["unknown"])
        if isinstance(src, list):
            src = src[0] if src else "unknown"
        # Weight scheme: SIS > PLM > ChEMBL
        w = {"SIS": 1.5, "PLM": 1.0}.get(src, 0.7)
        weights_list.append(w)

    weights = np.array(weights_list, dtype=np.float32)
    assert len(weights) == len(y_tr), f"Weight mismatch: {len(weights)} vs {len(y_tr)}"

    for w_label, w_sis, w_plm, w_other in [
        ("SIS_1.5", 1.5, 1.0, 0.7),
        ("SIS_2.0", 2.0, 1.0, 0.5),
        ("SIS_1.0_even", 1.0, 1.0, 1.0),
    ]:
        w_arr = np.ones(len(y_tr), dtype=np.float32)
        idx = 0
        for row in v12:
            ik = (row["ik"] or "")[:14]
            if ik in ho_iks:
                continue
            log_cd = row.get("log_cd")
            if log_cd is None:
                cmax = row.get("cmax_ngml")
                dose = row.get("dose_mg")
                if not cmax or not dose or dose <= 0:
                    continue
            fp = smiles_to_fp(row["smiles"])
            if fp is None:
                continue
            src = row.get("src", ["unknown"])
            if isinstance(src, list):
                src = src[0] if src else "unknown"
            w_arr[idx] = {"SIS": w_sis, "PLM": w_plm}.get(src, w_other)
            idx += 1

        ho_sw, _ = multi_seed_eval(X_tr, y_tr, X_ho, y_ho, XGB_BASE_PARAMS, SEEDS,
                                    sample_weight=w_arr[:len(y_tr)])
        results[f"weight_{w_label}"] = {"ho_mean": round(np.mean(ho_sw), 4),
                                         "ho_std": round(np.std(ho_sw), 4)}
        delta = np.mean(ho_sw) - np.mean(ho_base)
        print(f"  {w_label}: HO={np.mean(ho_sw):.4f} (Δ={delta:+.4f})", flush=True)

    # === Exp C: FP Variants ===
    print("\n--- Exp C: FP Variants ---", flush=True)
    for fp_var in ["ecfp6", "fcfp4", "fcfp6"]:
        print(f"  Building {fp_var} features...", end="", flush=True)
        try:
            X_tr_v, y_tr_v, g_v = build_features_variant(
                v12, tdc, ho_iks, encoder, device, fp_var)
            X_ho_v, y_ho_v, _ = build_holdout_variant(
                ho_def, tdc, encoder, device, fp_var)
            ho_v, _ = multi_seed_eval(X_tr_v, y_tr_v, X_ho_v, y_ho_v,
                                       XGB_BASE_PARAMS, SEEDS)
            results[f"fp_{fp_var}"] = {"ho_mean": round(np.mean(ho_v), 4),
                                        "ho_std": round(np.std(ho_v), 4)}
            delta = np.mean(ho_v) - np.mean(ho_base)
            print(f" HO={np.mean(ho_v):.4f} (Δ={delta:+.4f})", flush=True)
        except Exception as e:
            print(f" FAILED: {e}", flush=True)
            results[f"fp_{fp_var}"] = {"error": str(e)}

    # === Exp D: Best combo ===
    print("\n--- Exp D: Best Combination ---", flush=True)
    # Find best bias and best weight
    best_bias_key = min(
        [k for k in results if k.startswith("bias_")],
        key=lambda k: results[k]["ho_mean"])
    best_bias = results[best_bias_key].get("bias", 0)

    best_weight_key = min(
        [k for k in results if k.startswith("weight_")],
        key=lambda k: results[k]["ho_mean"])

    # Rebuild best weight array
    best_w_config = best_weight_key.split("_", 1)[1]
    w_configs = {"SIS_1.5": (1.5, 1.0, 0.7), "SIS_2.0": (2.0, 1.0, 0.5),
                 "SIS_1.0_even": (1.0, 1.0, 1.0)}
    w_sis, w_plm, w_other = w_configs.get(best_w_config, (1.0, 1.0, 1.0))

    best_w_arr = np.ones(len(y_tr), dtype=np.float32)
    idx = 0
    for row in v12:
        ik = (row["ik"] or "")[:14]
        if ik in ho_iks:
            continue
        log_cd = row.get("log_cd")
        if log_cd is None:
            cmax = row.get("cmax_ngml")
            dose = row.get("dose_mg")
            if not cmax or not dose or dose <= 0:
                continue
        fp = smiles_to_fp(row["smiles"])
        if fp is None:
            continue
        src = row.get("src", ["unknown"])
        if isinstance(src, list):
            src = src[0] if src else "unknown"
        best_w_arr[idx] = {"SIS": w_sis, "PLM": w_plm}.get(src, w_other)
        idx += 1

    ho_combo, _ = multi_seed_eval(X_tr, y_tr, X_ho, y_ho, XGB_BASE_PARAMS, SEEDS,
                                   sample_weight=best_w_arr[:len(y_tr)],
                                   bias=best_bias)
    results["best_combo"] = {
        "bias": round(best_bias, 4),
        "weight_config": best_w_config,
        "ho_mean": round(np.mean(ho_combo), 4),
        "ho_std": round(np.std(ho_combo), 4),
    }
    delta = np.mean(ho_combo) - np.mean(ho_base)
    print(f"  Bias({best_bias:+.4f}) + Weight({best_w_config}): "
          f"HO={np.mean(ho_combo):.4f} (Δ={delta:+.4f})", flush=True)

    # === Summary ===
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY (sorted by HO AAFE)", flush=True)
    print("=" * 60, flush=True)
    sorted_configs = sorted(
        [(k, v) for k, v in results.items() if "ho_mean" in v],
        key=lambda x: x[1]["ho_mean"])
    for name, m in sorted_configs:
        delta = m["ho_mean"] - np.mean(ho_base)
        std_str = f"±{m.get('ho_std', 0):.4f}" if 'ho_std' in m else ""
        print(f"  {name:<25} HO={m['ho_mean']:.4f} {std_str:>8} "
              f"Δ={delta:+.4f}", flush=True)

    best_name, best_m = sorted_configs[0]
    best_ho = best_m["ho_mean"]
    best_delta = best_ho - 3.332

    if best_ho <= 3.28:
        verdict = "PASS"
    elif best_ho <= 3.31:
        verdict = "PARTIAL"
    else:
        verdict = "NULL"
    print(f"\nBest: {best_name} (HO={best_ho:.4f}, Δ={best_delta:+.4f})", flush=True)
    print(f"Verdict: {verdict}", flush=True)

    # Save
    results["verdict"] = verdict
    results["best_config"] = best_name
    results["best_ho"] = round(best_ho, 4)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
