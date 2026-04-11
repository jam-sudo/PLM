"""S12 — Pre-registered retrain on v12 (v11 + ChEMBL v2 strict).

PRE-REGISTRATION
================
Date: 2026-04-11
Hypothesis: Adding ChEMBL v2 strict human-oral-filtered rows to v11 reduces HO AAFE.
Baseline (S11 fp_enc_base): 3.372 ± 0.010 (3 seeds)

Success criteria (applied to ΔHO = new − baseline, lower is better):
  PASS   (improves):    ΔHO ≤ −0.05 (new_HO ≤ 3.322)
  PARTIAL:              −0.05 < ΔHO ≤ −0.02
  NULL:                 −0.02 < ΔHO < +0.05
  HARM:                 ΔHO ≥ +0.05 (new_HO ≥ 3.422)

Design: same fp_enc_base pipeline as S11 but swap v11 → v12.
  - 5-fold GroupKFold by IK14
  - 3 seeds: 42, 137, 2024
  - Feature stack: FP4096 + ADME encoder 128 + physchem 20 + TDC 9 + μPBPK 6 + log_dose
  - Compare A=v12, B=v11 (same seeds, paired)

Outputs: models/b1/s12_v12_results.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/jam/PLM")
OUT = ROOT / "models/b1/s12_v12_results.json"

FP_BITS = 4096
EMB_DIM = 128
N_PHYSCHEM = 20
N_TDC = 9
N_UPBPK = 6

ENCODER_HIDDEN = (768, 384, EMB_DIM)
N_ADME_TASKS = 11

XGB_BASE_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.3,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5,
    n_jobs=8, verbosity=0, tree_method="hist",
)

SEEDS = [42, 137, 2024]


class ADMEEncoder(nn.Module):
    def __init__(self, input_dim=FP_BITS, hidden=ENCODER_HIDDEN, n_tasks=N_ADME_TASKS, dropout=0.2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden[-1], 1) for _ in range(n_tasks)])

    def forward(self, x):
        emb = self.encoder(x)
        return emb, [h(emb).squeeze(-1) for h in self.heads]


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    return np.array(fp, dtype=np.float32)


def smiles_to_physchem(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(N_PHYSCHEM, np.nan, dtype=np.float32)

    def safe(fn):
        try:
            v = fn(mol)
            return v if (v is not None and np.isfinite(v)) else np.nan
        except Exception:
            return np.nan

    return np.array([
        safe(Descriptors.ExactMolWt), safe(Descriptors.MolLogP), safe(Descriptors.TPSA),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol), safe(Descriptors.FractionCSP3),
        Descriptors.HeavyAtomCount(mol), Descriptors.NumHeteroatoms(mol),
        safe(Descriptors.LabuteASA), safe(Descriptors.BertzCT),
        safe(Descriptors.Chi0v), safe(Descriptors.Chi1v),
        safe(Descriptors.HallKierAlpha), safe(Descriptors.Kappa1),
        safe(Descriptors.Kappa2), safe(Descriptors.MaxPartialCharge),
        safe(Descriptors.MinPartialCharge),
    ], dtype=np.float32)


def get_tdc_features(ik14, tdc):
    e = tdc.get(ik14, {})
    keys = ["logS","caco2_logPapp","ppb_pct","vd_L_kg","half_life_h",
            "clearance_ul_min_mg","clearance_ul_min_million_cells","logD","bioavailability_binary"]
    return np.array([e.get(k, np.nan) for k in keys], dtype=np.float32)


def compute_micropbpk(ik14, tdc):
    e = tdc.get(ik14, {})
    f = np.full(N_UPBPK, np.nan, dtype=np.float32)
    caco2, ppb = e.get("caco2_logPapp"), e.get("ppb_pct")
    cl_hep, cl_mic = e.get("clearance_ul_min_million_cells"), e.get("clearance_ul_min_mg")
    vd, th = e.get("vd_L_kg"), e.get("half_life_h")
    if caco2 is not None:
        papp = 10**caco2
        f[0] = min(max(papp/(papp+1e-6), 0.01), 1.0)
    if ppb is not None:
        f[1] = max((100-ppb)/100, 0.001)
    Q = 1500
    if cl_hep is not None:
        ci = cl_hep*120*20/1000
        f[2] = min(max(ci/(Q+ci), 1e-3), 0.999)
    elif cl_mic is not None:
        ci = cl_mic*45*20/1000
        f[2] = min(max(ci/(Q+ci), 1e-3), 0.999)
    if not np.isnan(f[0]) and not np.isnan(f[2]):
        f[3] = f[0]*(1-f[2])
    if vd is not None:
        f[4] = vd
    if th is not None and th > 0:
        f[5] = 0.693/th
    return f


def aafe_metrics(pred, true):
    err = np.abs(pred - true)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return dict(aafe=float("nan"), fold2=float("nan"), fold3=float("nan"), n=0)
    return dict(
        aafe=round(float(10**np.mean(err)), 4),
        fold2=round(float(100*np.mean(err < math.log10(2))), 2),
        fold3=round(float(100*np.mean(err < math.log10(3))), 2),
        n=int(len(err)),
    )


def build_features(dataset, tdc, ho_iks, encoder, device):
    fp_cache, emb_cache = {}, {}

    def get_fp(smi):
        if smi not in fp_cache:
            fp_cache[smi] = smiles_to_fp(smi)
        return fp_cache[smi]

    def get_emb(smi, fp):
        if smi not in emb_cache:
            with torch.no_grad():
                t = torch.tensor(fp).unsqueeze(0).to(device)
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
        emb = get_emb(row["smiles"], fp)
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


def build_holdout(ho, tdc, encoder, device):
    fp_cache, emb_cache = {}, {}

    def get_fp(smi):
        if smi not in fp_cache:
            fp_cache[smi] = smiles_to_fp(smi)
        return fp_cache[smi]

    def get_emb(smi, fp):
        if smi not in emb_cache:
            with torch.no_grad():
                t = torch.tensor(fp).unsqueeze(0).to(device)
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
        emb = get_emb(smi, fp)
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


def run_config(tag, X_tr, y_tr, groups, X_ho, y_ho, seeds):
    gkf = GroupKFold(n_splits=5)
    results = []
    for seed in seeds:
        params = {**XGB_BASE_PARAMS, "random_state": seed}
        cv_preds = np.full_like(y_tr, np.nan)
        for ti, vi in gkf.split(X_tr, y_tr, groups):
            m = xgb.XGBRegressor(**params)
            m.fit(X_tr[ti], y_tr[ti])
            cv_preds[vi] = m.predict(X_tr[vi])
        cv_m = aafe_metrics(cv_preds, y_tr)
        m_full = xgb.XGBRegressor(**params)
        m_full.fit(X_tr, y_tr)
        ho_pred = m_full.predict(X_ho)
        ho_m = aafe_metrics(ho_pred, y_ho)
        gap = ho_m["aafe"] - cv_m["aafe"]
        results.append({"seed": seed, "cv": cv_m, "holdout": ho_m, "cv_ho_gap": round(gap, 4)})
        print(f"  [{tag}] seed={seed}: CV={cv_m['aafe']}  HO={ho_m['aafe']}  gap={gap:.3f}")
    return results


def main():
    print("S12 — v12 retrain (v11 + ChEMBL v2 strict)")
    print("=" * 60)

    v11 = json.load(open(ROOT / "data/curated/plm_dataset_v11_llm.json"))
    v12 = json.load(open(ROOT / "data/curated/plm_dataset_v12_chembl.json"))
    ho = json.load(open(ROOT / "data/validation/holdout_definition.json"))
    tdc = {k[:14]: v for k, v in json.load(open(ROOT / "data/curated/tdc_adme_data.json")).items()}
    ho_iks = set((k or "")[:14] for k in ho["holdout_inchikeys"])

    print(f"v11 rows: {len(v11)}")
    print(f"v12 rows: {len(v12)} (+{len(v12)-len(v11)})")
    print(f"Holdout: {len(ho_iks)} IK14s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ADMEEncoder().to(device)
    state = torch.load(ROOT / "models/adme_encoder.pt", map_location=device, weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()
    print(f"Encoder loaded ({device})")

    print("\nBuilding features A (v11 baseline)...")
    X_A, y_A, g_A = build_features(v11, tdc, ho_iks, encoder, device)
    print(f"  A: {X_A.shape}")

    print("Building features B (v12 expanded)...")
    X_B, y_B, g_B = build_features(v12, tdc, ho_iks, encoder, device)
    print(f"  B: {X_B.shape}")

    print("Building holdout features...")
    X_ho, y_ho, names = build_holdout(ho, tdc, encoder, device)
    print(f"  Holdout: {X_ho.shape}, y={len(y_ho)}")

    print("\n--- Run A (v11 baseline) ---")
    res_A = run_config("v11", X_A, y_A, g_A, X_ho, y_ho, SEEDS)

    print("\n--- Run B (v12 + ChEMBL v2) ---")
    res_B = run_config("v12", X_B, y_B, g_B, X_ho, y_ho, SEEDS)

    ho_A = np.array([r["holdout"]["aafe"] for r in res_A])
    ho_B = np.array([r["holdout"]["aafe"] for r in res_B])
    cv_A = np.array([r["cv"]["aafe"] for r in res_A])
    cv_B = np.array([r["cv"]["aafe"] for r in res_B])

    delta = ho_B - ho_A  # negative = v12 better
    paired_mean = delta.mean()
    paired_std = delta.std(ddof=1) if len(delta) > 1 else 0.0

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    print(f"v11 CV AAFE: {cv_A.mean():.3f}±{cv_A.std():.3f}  HO: {ho_A.mean():.3f}±{ho_A.std():.3f}")
    print(f"v12 CV AAFE: {cv_B.mean():.3f}±{cv_B.std():.3f}  HO: {ho_B.mean():.3f}±{ho_B.std():.3f}")
    print(f"ΔHO (v12-v11): {paired_mean:+.4f} ± {paired_std:.4f}")

    if paired_mean <= -0.05:
        verdict = "PASS (v12 improves HO)"
    elif paired_mean <= -0.02:
        verdict = "PARTIAL (v12 improves HO modestly)"
    elif paired_mean < 0.05:
        verdict = "NULL (no HO change)"
    else:
        verdict = "HARM (v12 worsens HO)"
    print(f"Verdict: {verdict}")

    out = {
        "pre_registration": {
            "hypothesis": "ChEMBL v2 strict row addition to v11 reduces HO AAFE",
            "baseline_s11_ho": 3.372,
            "bands": {"PASS": "delta_ho <= -0.05", "PARTIAL": "delta_ho <= -0.02",
                       "NULL": "-0.02 to +0.05", "HARM": ">= +0.05"},
            "seeds": SEEDS,
        },
        "v11": res_A, "v12": res_B,
        "summary": {
            "v11_rows": len(v11), "v12_rows": len(v12),
            "cv_v11_mean": float(cv_A.mean()), "cv_v12_mean": float(cv_B.mean()),
            "ho_v11_mean": float(ho_A.mean()), "ho_v12_mean": float(ho_B.mean()),
            "delta_ho_mean": float(paired_mean),
            "delta_ho_std": float(paired_std),
        },
        "verdict": verdict,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
