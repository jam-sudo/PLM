"""
Post-hoc refinement of chembl_v2_strict.json addressing audit findings.

Audit of 10 random rows revealed 20% contamination:
1. "assessed as [metabolite name]" — parent SMILES with metabolite Cmax
2. "twice daily / BID / TID / QID / for N days" — multi-dose steady-state

This script loads chembl_v2_strict.json, applies additional filters,
and writes chembl_v3_refined.json.
"""

import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path("/home/jam/PLM")

# Metabolite measurement patterns
METABOLITE_PATTERNS = [
    re.compile(r"assessed as (?!dose|cmax|\d)", re.IGNORECASE),  # "assessed as [name]" but not "assessed as dose/Cmax"
    re.compile(r"\bmetabolite\b", re.IGNORECASE),
    re.compile(r"\bn-oxide\b", re.IGNORECASE),
    re.compile(r"[A-Z]+-\d+\s+\d+[- ]?(keto|hydroxy|oxo|methyl|amine|acid|glucuronide|sulfate)", re.IGNORECASE),
    re.compile(r"\b(M\d+|M-\d+)\b"),  # M1, M2, M-3 metabolite labels
]

# Multi-dose steady-state patterns
MULTIDOSE_PATTERNS = [
    re.compile(r"\btwice[\s-]?daily\b", re.IGNORECASE),
    re.compile(r"\b(?:bid|t\.?i\.?d\.?|q\.?i\.?d\.?|q\s?d|qh)\b", re.IGNORECASE),
    re.compile(r"\bthrice[\s-]?daily\b", re.IGNORECASE),
    re.compile(r"\bonce[\s-]?daily\s+for\s+\d", re.IGNORECASE),
    re.compile(r"\bfor\s+\d+\s+days?\b", re.IGNORECASE),
    re.compile(r"\bsteady[\s-]?state\b", re.IGNORECASE),
    re.compile(r"\bmultiple\s+dose", re.IGNORECASE),
    re.compile(r"\bdosing\s+continued\b", re.IGNORECASE),
    re.compile(r"\bon day \d+\b", re.IGNORECASE),
    re.compile(r"\bwith\s+food", re.IGNORECASE),  # borderline but often SS
]

# Explicit single-dose positive mention is a whitelist override
SINGLE_DOSE_PATTERNS = [
    re.compile(r"\bsingle\s+(?:oral\s+)?dose\b", re.IGNORECASE),
    re.compile(r"\bsingle\s+ascending\s+dose", re.IGNORECASE),
]


def is_metabolite(desc: str) -> bool:
    return any(p.search(desc or "") for p in METABOLITE_PATTERNS)


def is_multidose(desc: str) -> bool:
    return any(p.search(desc or "") for p in MULTIDOSE_PATTERNS)


def is_single_dose_explicit(desc: str) -> bool:
    return any(p.search(desc or "") for p in SINGLE_DOSE_PATTERNS)


def main(input_path: Path = None):
    if input_path is None:
        input_path = ROOT / "data/curated/chembl_v2_strict.json"
    out_path = ROOT / "data/curated/chembl_v3_refined.json"

    d = json.loads(input_path.read_text())
    entries = d["entries"]
    print(f"Input: {len(entries)} (drug, dose) pairs from {input_path.name}")

    kept = []
    rejected = Counter()
    for e in entries:
        desc = e.get("sample_desc", "") or ""

        # Rule 1: reject metabolite patterns
        if is_metabolite(desc):
            rejected["metabolite"] += 1
            continue

        # Rule 2: reject multi-dose UNLESS explicit single-dose mention overrides
        if is_multidose(desc) and not is_single_dose_explicit(desc):
            rejected["multidose"] += 1
            continue

        kept.append(e)

    print(f"\nKept: {len(kept)}")
    print(f"Rejected: {sum(rejected.values())}")
    for k, v in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    yield_pct = 100 * len(kept) / len(entries) if entries else 0
    print(f"\nPost-refinement yield: {yield_pct:.1f}%")

    out = {
        "source": str(input_path.name),
        "refinement_date": "2026-04-11",
        "audit_finding": "20% contamination in v2 (10 random samples): metabolite Cmax + multi-dose SS",
        "input_count": len(entries),
        "kept": len(kept),
        "rejected_counts": dict(rejected),
        "entries": kept,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        main()
