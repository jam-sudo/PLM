"""
Build plm_dataset_v0.6_cleaned.json by removing 18 high-confidence
contaminated entries from v0.5.

Contamination evidence: batches 2-5 of visual_extraction (2026-04-10), where
source figures were visually confirmed to NOT be single-dose oral parent-drug
PK profiles. See data/curated/visual_extraction_full_findings.json and
docs/prereg_x2_cleanup.md.

Removal rules match by (drug_name_lower, source_nda, dose_mg, approximate cmax).
All removals logged with exact justification.
"""

import json
from pathlib import Path

ROOT = Path('/home/jam/PLM')
V05_PATH = ROOT / 'data/curated/plm_dataset_v0.5_cleaned.json'
V06_PATH = ROOT / 'data/curated/plm_dataset_v0.6_cleaned.json'
LOG_PATH = ROOT / 'data/curated/v06_cleanup_log.json'

# Rules: each rule matches drug_name, source_nda; optionally dose/cmax filter.
# Removal triggers if rule conditions met.
REMOVAL_RULES = [
    {
        "drug": "motixafortide", "source_nda": "NDA217159",
        "reason": "Figure was CD34+ cell count (PD response), not PK. Motixafortide is SC peptide, not oral.",
        "all_matching": True,  # remove ALL motixafortide entries from this NDA
    },
    {
        "drug": "nitroglycerin", "source_nda": "NDA208424",
        "reason": "Figure was 1,2-GDN metabolite (not parent GTN). Stored as 'oral 6.5 mg' but real NTG is 0.4-0.8 mg sublingual. Triple contamination: wrong analyte, wrong dose, wrong route.",
        "all_matching": True,
    },
    {
        "drug": "oxymetazoline", "source_nda": "NDA208032",
        "reason": "Real oxymetazoline is 0.05-0.2 mg intranasal (pediatric Kovanaze). Stored as 'oral 18 mg' is 100x dose error + wrong route. Auto-digitizer parsed text erroneously.",
        "all_matching": True,
    },
    {
        "drug": "naloxone", "source_nda": "NDA205777",
        "reason": "Oral naloxone bioavailability is ~2%; Cmax at 20 mg should be <1 ng/mL. Stored values 2-102 ng/mL impossible. Figure was simulated multi-dose oxycodone/naloxone SS.",
        "all_matching": True,
    },
    {
        "drug": "nirmatrelvir", "source_nda": "NDA217188",
        "reason": "Real nirmatrelvir 100 mg gives Cmax ~1000 ng/mL; stored 2.0 ng/mL is 500x off. Figure was DDI scatter plot (AUCR), not C-t profile.",
        "all_matching": True,
    },
    {
        "drug": "rucaparib", "source_nda": "NDA209115",
        "reason": "Real rucaparib 600 mg gives Cmax ~1900 ng/mL; stored 4.3 ng/mL is 400x off. Figure was BID steady-state (flat profile from t=0, pre-dose baseline), not single dose.",
        "all_matching": True,
    },
    {
        "drug": "rolapitant", "source_nda": "NDA206500",
        "reason": "Figure was dexamethasone (DEX) DDI victim, not rolapitant. Stored as rolapitant but analyte was DEX.",
        "all_matching": True,
    },
    {
        "drug": "netupitant", "source_nda": "NDA205718",
        "reason": "Specific entry with Cmax ~7.9 ng/mL at 300 mg oral — figure showed digoxin DDI. Digoxin Cmax values are typically <10 μg/L and this matches the mis-parsed digoxin data.",
        "cmax_lt": 20,  # only remove netupitant entries with implausibly low Cmax
        "dose_eq": 300.0,
    },
    {
        "drug": "sarecycline", "source_nda": "NDA209521",
        "reason": "Figure was urinary excretion (Ae, CLR) TABLE, not plasma C-t. Specific entry at 100 mg Cmax 6912 is 6x real literature value.",
        "dose_eq": 100.0,
        "cmax_gt": 3000,
    },
]


def matches_rule(profile: dict, rule: dict) -> bool:
    """Check if profile matches removal rule."""
    dn = profile.get('drug_name', '').lower()
    src = profile.get('source_nda', '')
    if rule['drug'] not in dn:
        return False
    if rule['source_nda'] not in src:
        return False
    if rule.get('all_matching'):
        return True
    # Conditional filters
    dose = profile.get('dose_mg', None)
    cmax = profile.get('cmax_reported') or profile.get('cmax_ngml')
    if 'dose_eq' in rule:
        if dose is None or abs(float(dose) - rule['dose_eq']) > 0.1:
            return False
    if 'cmax_lt' in rule and (cmax is None or float(cmax) >= rule['cmax_lt']):
        return False
    if 'cmax_gt' in rule and (cmax is None or float(cmax) <= rule['cmax_gt']):
        return False
    return True


def main():
    v05 = json.loads(V05_PATH.read_text())
    profiles = v05['profiles']
    n_before = len(profiles)
    print(f"v0.5 input: {n_before} profiles, {len(set(p.get('drug_name','').lower() for p in profiles))} drugs")

    keep = []
    removed = []
    for p in profiles:
        matched_rule = None
        for rule in REMOVAL_RULES:
            if matches_rule(p, rule):
                matched_rule = rule
                break
        if matched_rule:
            removed.append({
                "profile": {
                    "drug_name": p.get('drug_name'),
                    "source_nda": p.get('source_nda'),
                    "dose_mg": p.get('dose_mg'),
                    "cmax_reported": p.get('cmax_reported') or p.get('cmax_ngml'),
                    "route": p.get('route'),
                },
                "rule_drug": matched_rule['drug'],
                "reason": matched_rule['reason'],
            })
        else:
            keep.append(p)

    n_after = len(keep)
    print(f"\nRemoved: {n_before - n_after} profiles")
    print(f"v0.6 output: {n_after} profiles, {len(set(p.get('drug_name','').lower() for p in keep))} drugs")

    # Group removal by drug for summary
    from collections import Counter
    by_drug = Counter(r['profile']['drug_name'] for r in removed)
    print("\n=== Removal breakdown by drug ===")
    for d, n in sorted(by_drug.items(), key=lambda x: -x[1]):
        print(f"  {d}: {n} entries")

    # Write v0.6
    v06 = dict(v05)
    v06['version'] = '0.6'
    v06['creation_date'] = '2026-04-10'
    v06['n_profiles'] = n_after
    v06['n_unique_drugs'] = len(set(p.get('drug_name', '').lower() for p in keep))
    v06['n_unique_smiles'] = len(set(p.get('smiles', '') for p in keep))
    v06['filtering_v05'] = v05.get('filtering', None)
    v06['v06_cleanup'] = {
        "date": "2026-04-10",
        "removed_count": n_before - n_after,
        "reason": "Visual-inspection-confirmed contamination (wrong analyte, wrong route/dose, multi-dose SS)",
        "source_evidence": "data/curated/visual_extraction_full_findings.json",
        "pre_registration": "docs/prereg_x2_cleanup.md",
    }
    v06['profiles'] = keep
    V06_PATH.write_text(json.dumps(v06, indent=2))
    print(f"\nWrote {V06_PATH}")

    # Write removal log
    log = {
        "date": "2026-04-10",
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_before - n_after,
        "by_drug": dict(by_drug),
        "rules_applied": REMOVAL_RULES,
        "removed_profiles": removed,
    }
    LOG_PATH.write_text(json.dumps(log, indent=2, default=str))
    print(f"Wrote {LOG_PATH}")


if __name__ == '__main__':
    main()
