"""
Train XGBoost with DrugBank synthetic data added to v10 training set.
Compares: baseline (v10 only) vs expanded (v10 + DrugBank synthetic).

Usage:
    python -m pipeline.train_with_drugbank
"""

import json, math, warnings, sys
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import xgboost as xgb
warnings.filterwarnings('ignore')

sys.path.insert(0, 'pipeline')
from llm_enriched_experiment import (
    smi_to_ik, build_sample, CANONICAL_COND, normalize_condition, XGB_PARAMS,
    smiles_to_fp,
)
from ho_diagnostic import build_training, morgan_fp_2048


def load_drugbank_as_v10(
    path: str = "data/curated/synthetic_drugbank_ct.json",
    ho_iks: set = None,
    existing_iks: set = None,
) -> list:
    """Convert DrugBank synthetic profiles to v10 format."""
    with open(path) as f:
        synthetic = json.load(f)

    entries = []
    for p in synthetic:
        smi = p.get("smiles")
        dose = p.get("dose_mg", 0)
        cmax = p.get("cmax_ngml", 0)
        if not smi or dose <= 0 or cmax <= 0:
            continue

        ik = p.get("inchikey_14", "")

        # Exclude holdout
        if ho_iks and ik in ho_iks:
            continue

        # Exclude already in training (optional, for dedup)
        if existing_iks and ik in existing_iks:
            continue

        # Validate SMILES
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Compute InChIKey if missing
        if not ik:
            try:
                inchi = Chem.MolToInchi(mol)
                ik = Chem.InchiToInchiKey(inchi)[:14] if inchi else ""
            except:
                continue

        log_cd = math.log10(cmax / dose)

        entries.append({
            "smiles": smi,
            "dose_mg": dose,
            "cmax_ngml": cmax,
            "ik": ik,
            "src": "DrugBank_syn",
            "log_cd": log_cd,
        })

    return entries


def train_and_evaluate(X_tr, Y_tr, g_tr, W_tr, smi_tr, holdout_drugs, tdc, label=""):
    """Train XGBoost and evaluate on holdout. Returns AAFE and details."""
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)

    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, Y_tr, sample_weight=W_tr)

    # Predict holdout
    results = []
    for d in holdout_drugs:
        smi, dose, cmax = d.get("smiles"), d.get("dose_mg"), d.get("cmax_obs_ngml")
        if not smi or not dose or dose <= 0 or not cmax or cmax <= 0:
            continue
        ik = d.get("inchikey14", "")
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None:
            continue
        X_h = np.array([s], dtype=np.float32)
        X_h = np.where(np.isinf(X_h), np.nan, X_h)
        pred = float(m.predict(X_h)[0])
        actual = math.log10(cmax / dose)
        err = abs(pred - actual)
        results.append({
            "name": d["name"],
            "actual_log_cd": actual,
            "pred_log_cd": pred,
            "abs_err": err,
            "fold_err": 10 ** err,
        })

    if not results:
        return None

    errs = [r["abs_err"] for r in results]
    aafe = 10 ** np.mean(errs)
    f2 = np.mean([1 for e in errs if e < np.log10(2)]) / len(errs) * 100
    f3 = np.mean([1 for e in errs if e < np.log10(3)]) / len(errs) * 100

    return {
        "label": label,
        "n_train": len(Y_tr),
        "n_train_drugs": len(set(g_tr)),
        "n_holdout": len(results),
        "aafe": round(float(aafe), 3),
        "f2_pct": round(float(f2), 1),
        "f3_pct": round(float(f3), 1),
        "per_drug": results,
    }


def main():
    print("=" * 75)
    print("TRAIN WITH DRUGBANK — A/B Comparison")
    print("=" * 75)

    # Load data
    with open("data/curated/tdc_adme_data.json") as f:
        tdc = json.load(f)
    with open("data/validation/holdout_definition.json") as f:
        ho_data = json.load(f)
    holdout_drugs = ho_data["holdout_drugs"]
    ho_iks = set(d["inchikey14"] for d in holdout_drugs)
    with open("data/curated/plm_dataset_v10_labels.json") as f:
        v10 = json.load(f)
    with open("data/llm_extracted/pk_llm_merged.json") as f:
        llm = json.load(f)

    # ── Baseline: v10 + LLM (same as ho_diagnostic) ──
    print("\n>>> Baseline: v10 + LLM")
    X_base, Y_base, g_base, W_base, smi_base = build_training(v10, llm, ho_iks, tdc)
    print(f"  Training: {len(Y_base)} samples, {len(set(g_base))} drugs")
    r_base = train_and_evaluate(X_base, Y_base, g_base, W_base, smi_base, holdout_drugs, tdc, "baseline")

    # ── Expanded: v10 + LLM + DrugBank synthetic ──
    print("\n>>> Expanded: v10 + LLM + DrugBank synthetic")
    existing_iks = set(g_base)
    db_entries = load_drugbank_as_v10(ho_iks=ho_iks, existing_iks=existing_iks)
    print(f"  DrugBank new entries: {len(db_entries)}")

    # Add DrugBank entries to v10 so build_training picks them up
    v10_expanded = v10 + db_entries
    X_exp, Y_exp, g_exp, W_exp, smi_exp = build_training(v10_expanded, llm, ho_iks, tdc)
    print(f"  Training: {len(Y_exp)} samples, {len(set(g_exp))} drugs")
    r_exp = train_and_evaluate(X_exp, Y_exp, g_exp, W_exp, smi_exp, holdout_drugs, tdc, "expanded")

    # ── DrugBank only (diagnostic) ──
    print("\n>>> DrugBank only (diagnostic — no v10/LLM)")
    db_all = load_drugbank_as_v10(ho_iks=ho_iks, existing_iks=set())
    print(f"  DrugBank entries: {len(db_all)}")
    X_db, Y_db, g_db, W_db, smi_db = [], [], [], [], []
    for p in db_all:
        smi, dose, ik = p["smiles"], p["dose_mg"], p["ik"]
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None:
            continue
        X_db.append(s)
        Y_db.append(p["log_cd"])
        g_db.append(ik)
        W_db.append(0.5)  # lower weight for synthetic
        smi_db.append(smi)
    if X_db:
        X_db = np.array(X_db, dtype=np.float32)
        Y_db = np.array(Y_db, dtype=np.float32)
        g_db = np.array(g_db)
        W_db = np.array(W_db, dtype=np.float32)
        print(f"  Training: {len(Y_db)} samples, {len(set(g_db))} drugs")
        r_db = train_and_evaluate(X_db, Y_db, g_db, W_db, smi_db, holdout_drugs, tdc, "drugbank_only")
    else:
        r_db = None

    # ── Expanded with lower weight for synthetic ──
    print("\n>>> Expanded (DrugBank weight=0.3)")
    X_ew, Y_ew, g_ew, W_ew, smi_ew = build_training(v10_expanded, llm, ho_iks, tdc)
    # Downweight DrugBank entries
    for i, g in enumerate(g_ew):
        # Find if this is a DrugBank entry by checking source
        if i >= len(X_base):  # entries added after baseline are DrugBank
            W_ew[i] *= 0.3
    print(f"  Training: {len(Y_ew)} samples, {len(set(g_ew))} drugs")
    r_ew = train_and_evaluate(X_ew, Y_ew, g_ew, W_ew, smi_ew, holdout_drugs, tdc, "expanded_w0.3")

    # ── Results ──
    print("\n" + "=" * 75)
    print("RESULTS COMPARISON")
    print("=" * 75)
    header = f"{'Scenario':<30s} {'N_train':>8s} {'N_drugs':>8s} {'HO AAFE':>9s} {'2-fold%':>9s} {'3-fold%':>9s}"
    print(header)
    print("-" * len(header))

    for r in [r_base, r_exp, r_ew, r_db]:
        if r is None:
            continue
        print(
            f"{r['label']:<30s} {r['n_train']:>8d} {r['n_train_drugs']:>8d} "
            f"{r['aafe']:>9.3f} {r['f2_pct']:>8.1f}% {r['f3_pct']:>8.1f}%"
        )

    print(f"\nSisyphus Meta:                                      2.283")
    print(f"Sisyphus ML:                                        2.336")

    # Save
    output = {
        "baseline": {k: v for k, v in r_base.items() if k != "per_drug"} if r_base else None,
        "expanded": {k: v for k, v in r_exp.items() if k != "per_drug"} if r_exp else None,
        "expanded_w0.3": {k: v for k, v in r_ew.items() if k != "per_drug"} if r_ew else None,
        "drugbank_only": {k: v for k, v in r_db.items() if k != "per_drug"} if r_db else None,
    }
    with open("data/validation/drugbank_expansion_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to data/validation/drugbank_expansion_results.json")


if __name__ == "__main__":
    main()
