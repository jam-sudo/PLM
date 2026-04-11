"""
Build v12 training dataset = v11_llm + ChEMBL v2 strict extraction.

Merges filtered ChEMBL rows (src=CHEMBL_v2_strict) into v11.
Holdout protection: excludes any IK14 in the 97-drug holdout.
"""

import json
from pathlib import Path

ROOT = Path("/home/jam/PLM")

def main():
    v11 = json.loads((ROOT / "data/curated/plm_dataset_v11_llm.json").read_text())
    chembl = json.loads((ROOT / "data/curated/chembl_v2_strict.json").read_text())
    ho = json.loads((ROOT / "data/validation/holdout_definition.json").read_text())
    ho_iks = set((d.get("inchikey14") or "")[:14] for d in ho["holdout_drugs"])

    print(f"v11: {len(v11)} rows")
    print(f"ChEMBL v2 entries: {len(chembl['entries'])} (drug,dose) pairs")
    print(f"Holdout: {len(ho_iks)} drugs")

    # Build new rows matching v11 schema (smiles, dose_mg, cmax_ngml, ik, src, log_cd)
    new_rows = []
    for e in chembl["entries"]:
        ik14 = (e["ik"] or "")[:14]
        if ik14 in ho_iks:
            continue
        new_rows.append({
            "smiles": e["smiles"],
            "dose_mg": e["dose_mg"],
            "cmax_ngml": e["cmax_ngml"],
            "ik": ik14,
            "src": "CHEMBL_v2_strict",
            "log_cd": e["log_cd"],
        })
    print(f"New rows after holdout filter: {len(new_rows)}")

    # Drop duplicates within new_rows by (ik14, dose_mg)
    seen = set()
    dedup = []
    for r in new_rows:
        key = (r["ik"], round(r["dose_mg"], 2))
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    print(f"After dedup: {len(dedup)}")

    # Drop any that overlap with v11 by (ik14, dose_mg) — avoid same drug/dose double-counting
    v11_key = set((r["ik"][:14], round(r["dose_mg"], 2)) for r in v11 if r.get("ik") and r.get("dose_mg"))
    unique_new = [r for r in dedup if (r["ik"], round(r["dose_mg"], 2)) not in v11_key]
    print(f"After v11 overlap drop: {len(unique_new)}")

    v12 = v11 + unique_new
    print(f"\nv12 total: {len(v12)} rows (+{len(unique_new)} new)")

    # Unique drugs
    v12_iks = set(r["ik"][:14] for r in v12 if r.get("ik"))
    print(f"v12 unique drugs: {len(v12_iks)}")

    out = ROOT / "data/curated/plm_dataset_v12_chembl.json"
    out.write_text(json.dumps(v12, indent=None))
    print(f"\nWrote {out}")

    # Also save a small summary
    summary = {
        "date": "2026-04-11",
        "v11_rows": len(v11),
        "chembl_entries_raw": len(chembl["entries"]),
        "after_holdout_filter": len(new_rows),
        "after_dedup": len(dedup),
        "after_v11_overlap_drop": len(unique_new),
        "v12_total_rows": len(v12),
        "v12_unique_drugs": len(v12_iks),
        "v11_unique_drugs": len(set(r["ik"][:14] for r in v11 if r.get("ik"))),
    }
    (ROOT / "data/curated/v12_merge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
