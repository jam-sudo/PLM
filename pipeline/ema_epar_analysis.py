"""Analyze EMA medicines catalog for EPAR mining feasibility.

Questions:
1. How many human oral small molecules are authorized?
2. How many overlap with PLM's existing v11/v12 training set?
3. How many are truly novel (not in v11)?
4. Can we access EPAR PDFs programmatically?
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path("/home/jam/PLM")


def is_likely_oral_small_molecule(entry: dict) -> tuple[bool, str]:
    """Heuristic: check if entry is likely an oral small molecule."""
    # Must be human
    if entry.get("category") != "Human":
        return False, "not_human"

    # Must be authorized
    if entry.get("medicine_status") != "Authorised":
        return False, "not_authorized"

    # Skip biologics indicators
    active = (entry.get("active_substance") or "").lower()
    name = (entry.get("name_of_medicine") or "").lower()
    indication = (entry.get("therapeutic_indication") or "").lower()
    group = (entry.get("pharmacotherapeutic_group_human") or "").lower()

    biologic_keywords = [
        "antibod", "mab", "vaccine", "insulin", "interferon", "factor viii",
        "factor ix", "erythropoietin", "filgrastim", "pegfilgrastim",
        "adalimumab", "trastuzumab", "rituximab", "bevacizumab", "infliximab",
        "immunoglobulin", "serum", "plasma", "blood", "recombinant",
        "monoclonal", "peptide", "protein", "enzyme", "botulinum",
        "coagulation factor", "antithrombin",
    ]

    for kw in biologic_keywords:
        if kw in active or kw in name:
            return False, f"biologic_{kw}"

    # Skip biosimilars and generics (won't have unique PK data)
    if entry.get("biosimilar") == "Yes":
        return False, "biosimilar"
    if entry.get("generic") == "Yes":
        return False, "generic"

    # Check for oral route hints in indication/formulation
    oral_hints = ["oral", "tablet", "capsule", "mouth", "film-coated"]
    non_oral_hints = [
        "injection", "infusion", "intravenous", "subcutaneous",
        "intramuscular", "topical", "cream", "ointment", "eye drop",
        "inhaler", "inhalation", "nasal", "suppository", "implant",
        "transdermal", "patch", "solution for injection", "powder for solution",
        "concentrate for solution for infusion", "lyophilisate",
    ]

    indication_lower = indication.lower()

    is_oral = any(h in indication_lower for h in oral_hints)
    is_non_oral = any(h in indication_lower for h in non_oral_hints)

    # For drugs where indication doesn't mention route, check name patterns
    # (most oral drugs have "tablet" or "capsule" in their product info)
    if not is_oral and not is_non_oral:
        return False, "route_unknown"

    if is_non_oral and not is_oral:
        return False, "non_oral"

    return True, "oral_sm"


def main():
    catalog = json.loads((ROOT / "data/raw/ema_medicines.json").read_text())
    entries = catalog["data"]
    print(f"Total EMA catalog entries: {len(entries)}")

    # Filter to likely oral small molecules
    verdicts = Counter()
    oral_sms = []
    for e in entries:
        ok, reason = is_likely_oral_small_molecule(e)
        verdicts[reason] += 1
        if ok:
            oral_sms.append(e)

    print(f"\nClassification breakdown:")
    for v, c in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {c}")

    print(f"\nLikely oral small molecules: {len(oral_sms)}")

    # Load v11 drugs to check overlap
    v11 = json.loads((ROOT / "data/curated/plm_dataset_v11_llm.json").read_text())
    v11_drugs = set()
    for r in v11:
        dn = (r.get("drug_name") or "").lower().strip()
        if dn:
            v11_drugs.add(dn)
        # Also add INN/active substance variants
        ik = (r.get("ik") or "")[:14]
        if ik:
            v11_drugs.add(ik)

    v12 = json.loads((ROOT / "data/curated/plm_dataset_v12_chembl.json").read_text())
    v12_drugs = set()
    for r in v12:
        dn = (r.get("drug_name") or "").lower().strip()
        if dn:
            v12_drugs.add(dn)

    # Check overlap by active substance name
    novel = []
    overlap = []
    for e in oral_sms:
        active = (e.get("active_substance") or "").lower().strip()
        inn = (e.get("international_non_proprietary_name_common_name") or "").lower().strip()

        in_v11 = active in v11_drugs or inn in v11_drugs
        if in_v11:
            overlap.append(e)
        else:
            novel.append(e)

    print(f"\nOverlap with v11 by name: {len(overlap)}")
    print(f"Potentially novel: {len(novel)}")

    # Show sample of novel drugs
    print(f"\n=== Sample novel oral SMs (first 30) ===")
    for e in novel[:30]:
        active = e.get("active_substance", "")
        group = e.get("pharmacotherapeutic_group_human", "")
        url = e.get("medicine_url", "")
        print(f"  {active} — {group} — {url}")

    # Check EPAR URL pattern
    print(f"\n=== EPAR access check ===")
    sample_url = novel[0]["medicine_url"] if novel else ""
    print(f"Sample medicine URL: {sample_url}")
    print(f"EPAR PDF pattern: <medicine_url>/all-documents")
    print(f"Each medicine page links to 'European public assessment report (EPAR)'")
    print(f"EPAR typically includes 'Scientific discussion' PDF with PK tables")

    # Therapeutic area distribution
    areas = Counter()
    for e in novel:
        area = e.get("therapeutic_area_mesh", "Unknown")
        areas[area[:50]] += 1
    print(f"\nNovel drugs by therapeutic area (top 15):")
    for a, c in areas.most_common(15):
        print(f"  {a}: {c}")

    # Save analysis
    out = {
        "total_catalog": len(entries),
        "oral_small_molecules": len(oral_sms),
        "overlap_v11": len(overlap),
        "novel": len(novel),
        "novel_drugs": [
            {
                "active_substance": e.get("active_substance"),
                "inn": e.get("international_non_proprietary_name_common_name"),
                "group": e.get("pharmacotherapeutic_group_human"),
                "area": e.get("therapeutic_area_mesh"),
                "url": e.get("medicine_url"),
                "product_number": e.get("ema_product_number"),
            }
            for e in novel
        ],
        "overlap_drugs": [
            e.get("active_substance") for e in overlap
        ],
    }
    out_path = ROOT / "data/curated/ema_epar_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
