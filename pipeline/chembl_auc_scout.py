"""Scout ChEMBL AUC records — assess volume and usability.

Checks:
1. How many AUC activities exist (standard_type variants)?
2. What units are used?
3. What does the description distribution look like?
4. Can we extract human-oral single-dose AUC rows?

AUC itself can't replace Cmax, but AUC-bearing drugs may also have
Cmax in external literature, AND AUC can serve as an auxiliary feature.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/jam/PLM")


def main():
    from chembl_webresource_client.new_client import new_client

    activity = new_client.activity

    # 1. Check AUC variant counts
    auc_types = ["AUC", "AUC0-inf", "AUC0-t", "AUC0-24", "AUC0-12",
                 "AUC 0-inf", "AUC 0-t", "AUCtau", "AUClast", "AUC(0-inf)",
                 "AUCinf", "AUCtotal", "AUC_0-inf", "AUC_0-t"]

    print("=== ChEMBL AUC variant scan ===")
    found_types = {}
    for st in auc_types:
        try:
            acts = activity.filter(standard_type=st)
            count = len(acts)
            if count > 0:
                found_types[st] = count
                print(f"  {st}: {count:,} records")
            else:
                print(f"  {st}: 0")
        except Exception as e:
            print(f"  {st}: ERROR {e}")

    if not found_types:
        print("\nNo AUC records found in ChEMBL.")
        return

    # 2. Deep-dive on the largest type
    best_type = max(found_types, key=found_types.get)
    print(f"\n=== Deep dive: '{best_type}' ({found_types[best_type]:,} records) ===")

    # Sample first 2000 for analysis
    acts = activity.filter(standard_type=best_type)
    sample = []
    for i, a in enumerate(acts):
        if i >= 2000:
            break
        sample.append(a)

    print(f"Sampled {len(sample)} records")

    # Unit distribution
    units = Counter(a.get("standard_units") for a in sample)
    print(f"\nUnit distribution (top 10):")
    for u, c in units.most_common(10):
        print(f"  {u}: {c} ({100*c/len(sample):.1f}%)")

    # Description analysis (human/animal keywords)
    animal_kw = {"rat", "mouse", "mice", "dog", "monkey", "rabbit", "rodent",
                 "canine", "murine", "beagle", "sprague", "wistar"}
    human_kw = {"human", "healthy", "patient", "clinical", "volunteer", "subject"}
    oral_kw = {"oral", "po", "tablet", "capsule"}

    has_desc = 0
    animal_count = 0
    human_count = 0
    oral_count = 0
    ambiguous = 0

    for a in sample:
        desc = (a.get("assay_description") or "").lower()
        if not desc:
            continue
        has_desc += 1

        is_animal = any(kw in desc for kw in animal_kw)
        is_human = any(kw in desc for kw in human_kw)
        is_oral = any(kw in desc for kw in oral_kw)

        if is_animal:
            animal_count += 1
        if is_human:
            human_count += 1
        if is_oral:
            oral_count += 1
        if not is_animal and not is_human:
            ambiguous += 1

    print(f"\nDescription analysis ({has_desc} with descriptions):")
    print(f"  Animal keywords: {animal_count} ({100*animal_count/max(has_desc,1):.1f}%)")
    print(f"  Human keywords: {human_count} ({100*human_count/max(has_desc,1):.1f}%)")
    print(f"  Oral keywords: {oral_count} ({100*oral_count/max(has_desc,1):.1f}%)")
    print(f"  Ambiguous (neither): {ambiguous} ({100*ambiguous/max(has_desc,1):.1f}%)")

    # Organism distribution
    orgs = Counter(a.get("assay_organism") for a in sample)
    print(f"\nOrganism distribution (top 5):")
    for o, c in orgs.most_common(5):
        print(f"  {o}: {c}")

    # Show a few human oral examples
    print(f"\n=== Sample human oral AUC descriptions ===")
    count = 0
    for a in sample:
        desc = (a.get("assay_description") or "").lower()
        if any(kw in desc for kw in human_kw) and any(kw in desc for kw in oral_kw):
            if not any(kw in desc for kw in animal_kw):
                print(f"  [{a.get('standard_units')}] {a.get('standard_value')} — {desc[:120]}")
                count += 1
                if count >= 10:
                    break

    # Summary
    total_all = sum(found_types.values())
    print(f"\n=== SUMMARY ===")
    print(f"Total AUC-type records across all variants: {total_all:,}")
    print(f"Largest variant: '{best_type}' with {found_types[best_type]:,}")
    est_human_oral = int(human_count * oral_count / max(has_desc, 1))
    print(f"Estimated human+oral fraction: ~{100*est_human_oral/max(has_desc,1):.1f}%")
    print(f"Projected human oral AUC rows from {best_type}: ~{found_types[best_type] * est_human_oral // max(has_desc,1)}")

    # Save scout results
    out = {
        "auc_type_counts": found_types,
        "total_auc_records": total_all,
        "sample_size": len(sample),
        "unit_distribution": dict(units.most_common(10)),
        "description_stats": {
            "has_desc": has_desc,
            "animal": animal_count,
            "human": human_count,
            "oral": oral_count,
            "ambiguous": ambiguous,
        },
    }
    out_path = ROOT / "data/curated/chembl_auc_scout.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
