"""Build v12b = v11 + ChEMBL v3 refined (post-audit contamination removal)."""

import json
from pathlib import Path

ROOT = Path("/home/jam/PLM")

def main():
    v11 = json.loads((ROOT / "data/curated/plm_dataset_v11_llm.json").read_text())
    chembl = json.loads((ROOT / "data/curated/chembl_v3_refined.json").read_text())
    ho = json.loads((ROOT / "data/validation/holdout_definition.json").read_text())
    ho_iks = set((d.get("inchikey14") or "")[:14] for d in ho["holdout_drugs"])

    print(f"v11: {len(v11)} rows")
    print(f"ChEMBL v3 refined: {len(chembl['entries'])} (drug,dose) pairs")

    new_rows = []
    for e in chembl["entries"]:
        ik14 = (e["ik"] or "")[:14]
        if ik14 in ho_iks:
            continue
        new_rows.append({
            "smiles": e["smiles"], "dose_mg": e["dose_mg"],
            "cmax_ngml": e["cmax_ngml"], "ik": ik14,
            "src": "CHEMBL_v3_refined", "log_cd": e["log_cd"],
        })

    seen = set()
    dedup = []
    for r in new_rows:
        key = (r["ik"], round(r["dose_mg"], 2))
        if key not in seen:
            seen.add(key)
            dedup.append(r)

    v11_key = set((r["ik"][:14], round(r["dose_mg"], 2)) for r in v11 if r.get("ik") and r.get("dose_mg"))
    unique_new = [r for r in dedup if (r["ik"], round(r["dose_mg"], 2)) not in v11_key]

    print(f"After holdout filter: {len(new_rows)}")
    print(f"After dedup: {len(dedup)}")
    print(f"After v11 overlap drop: {len(unique_new)}")

    v12b = v11 + unique_new
    iks = set(r["ik"][:14] for r in v12b if r.get("ik"))
    print(f"\nv12b total: {len(v12b)} rows (+{len(unique_new)}), unique drugs: {len(iks)}")

    out = ROOT / "data/curated/plm_dataset_v12b_chembl_refined.json"
    out.write_text(json.dumps(v12b, indent=None))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
