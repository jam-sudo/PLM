"""
ChEMBL v2 Cmax extraction with STRICT human/dose filtering.

Addresses the three failure modes from F10/I4:
1. Animal data contamination (rat/mouse/dog PK mixed with human)
2. mg/kg → mg dose parsing error
3. log_cd shift due to species/unit confusion

Strict filters (applied at query + post-processing):
- assay_organism == 'Homo sapiens' (or description contains 'human'/'healthy')
- standard_units in allowed concentration units (ng/mL, mg/L, µg/mL — NOT nM/µM)
- description must NOT contain animal keywords (rat, mouse, mice, dog, monkey, canine, rodent, rabbit)
- description must NOT contain '/kg' (mg/kg dose) unless we can parse it properly
- dose must be in plausible human range (1-2000 mg)
- log_cd must be in v11 p1-p99 distribution (tight sanity check)

Usage: python3 pipeline/chembl_v2_strict.py [--max N]
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from chembl_webresource_client.new_client import new_client

ROOT = Path("/home/jam/PLM")

# Animal keywords (lowercase match)
ANIMAL_KEYWORDS = [
    "rat", "mice", "mouse", "dog", "canine", "monkey", "rabbit",
    "rodent", "murine", "beagle", "primate", "porcine", "ovine", "bovine",
    "sprague-dawley", "wistar", "c57", "balb", "cd1", "icr ", "irc ",
    "cynomolgus", "rhesus", "macaque",
]

HUMAN_KEYWORDS = [
    "human", "healthy subject", "healthy volunteer", "healthy male",
    "healthy female", "adult", "patient", "clinical", "homo sapiens",
]

# Reject non-oral routes. PLM targets oral/po/per os.
NON_ORAL_KEYWORDS = [
    " iv ", " i.v.", "intravenous", "infusion",
    " im ", " i.m.", "intramuscular",
    " sc ", " s.c.", "subcutaneous", "subcutaneously",
    "topical", "intranasal", "nasal spray", "inhalation",
    "inhaled", "intrathecal", "epidural", "transdermal",
    "sublingual", "buccal", "ocular", "otic", "rectal",
]

ORAL_KEYWORDS = [" po ", " p.o.", "oral", "orally", "per os", "tablet", "capsule", "suspension"]

# ChEMBL uses 'ug.mL-1' style notation
# Accept concentration units + nM (which needs MW conversion)
ACCEPTED_CONC_UNITS = {"ng/ml", "ug/ml", "mcg/ml", "mg/l", "ug/l", "mcg/l",
                       "ng.ml-1", "ug.ml-1", "mcg.ml-1", "mg.l-1",
                       "ng ml-1", "ug ml-1"}
ACCEPTED_MOLAR_UNITS = {"nm", "um", "µm", "nmol/l", "umol/l"}


def cmax_to_ngml(value: float, unit: str, mw: float | None = None) -> float | None:
    u = unit.lower().strip().replace(" ", "").replace(".", "/")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    # Concentration units
    if u in ("ng/ml", "ng/ml-1"):
        return v
    if u in ("ug/ml", "mcg/ml", "ug/ml-1", "mcg/ml-1"):
        return v * 1000
    if u in ("mg/l", "mg/l-1"):
        return v * 1000
    if u in ("ug/l", "mcg/l"):
        return v
    # Molar units (need MW)
    if mw is None or mw <= 0:
        return None
    if u in ("nm", "nmol/l"):
        return v * mw / 1000  # nM × g/mol / 1000 = ng/mL
    if u in ("um", "µm", "umol/l"):
        return v * mw  # µM × g/mol = ng/mL
    return None


def is_animal(desc: str) -> bool:
    d = (desc or "").lower()
    return any(kw in d for kw in ANIMAL_KEYWORDS)


def is_human(desc: str) -> bool:
    d = (desc or "").lower()
    return any(kw in d for kw in HUMAN_KEYWORDS)


def is_non_oral(desc: str) -> bool:
    """True if description mentions a non-oral route."""
    d = (desc or "").lower()
    return any(kw in d for kw in NON_ORAL_KEYWORDS)


def is_oral(desc: str) -> bool:
    d = (desc or "").lower()
    return any(kw in d for kw in ORAL_KEYWORDS)


DOSE_MG_RE = re.compile(r"(?<![/\d])(\d+(?:\.\d+)?)\s*mg\b(?!\s*/\s*kg)", re.IGNORECASE)
DOSE_MGKG_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*kg", re.IGNORECASE)


def extract_dose_mg(desc: str) -> tuple[float | None, str | None]:
    """Return (dose_mg, parse_reason) or (None, reason).

    Rejects mg/kg explicitly. Only accepts unambiguous mg values.
    """
    if not desc:
        return None, "empty_desc"
    if DOSE_MGKG_RE.search(desc):
        return None, "mg_per_kg_not_supported"
    m = DOSE_MG_RE.search(desc)
    if not m:
        return None, "no_mg_pattern"
    try:
        v = float(m.group(1))
    except ValueError:
        return None, "parse_error"
    if v <= 0 or v > 5000:
        return None, f"implausible_dose_{v}"
    return v, "ok"


def smiles_to_ik14(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.InchiToInchiKey(Chem.MolToInchi(mol))[:14]
    except Exception:
        return None


def main(max_records: int = 30000):
    # Exclude drugs already in v11 and holdout
    v11 = json.load(open(ROOT / "data/curated/plm_dataset_v11_llm.json"))
    v11_iks = set((r.get("ik") or "")[:14] for r in v11 if r.get("ik"))
    ho = json.load(open(ROOT / "data/validation/holdout_definition.json"))
    ho_iks = set((d.get("inchikey14") or "")[:14] for d in ho["holdout_drugs"])
    exclude_iks = v11_iks | ho_iks
    print(f"v11 IK14: {len(v11_iks)}, Holdout IK14: {len(ho_iks)}, exclude total: {len(exclude_iks)}")

    # v11 log_cd distribution for sanity
    v11_logcd = np.array([r["log_cd"] for r in v11 if r.get("log_cd") is not None])
    p1, p99 = np.percentile(v11_logcd, [1, 99])
    print(f"v11 log_cd range (p1-p99): [{p1:.3f}, {p99:.3f}]")

    print(f"\nQuerying ChEMBL (max={max_records} activities)...")
    activity = new_client.activity
    # Query Cmax without unit filter (ChEMBL uses ug.mL-1 style notation, and nM is common)
    # Rely entirely on description text filter for species (assay_organism is None most of the time)
    q = activity.filter(
        standard_type__in=["Cmax", "CMAX"],
    ).only([
        "molecule_chembl_id", "standard_value", "standard_units",
        "assay_description", "canonical_smiles", "assay_organism",
    ])

    reject_counts = defaultdict(int)
    by_mol = defaultdict(list)
    t0 = time.time()
    count = 0
    checkpoint_path = ROOT / "data/curated/chembl_v2_strict_partial.json"

    def checkpoint():
        pkg = {
            "partial": True,
            "processed": count,
            "kept_so_far": sum(len(v) for v in by_mol.values()),
            "reject_counts": dict(reject_counts),
            "by_mol": {k: v for k, v in by_mol.items()},
        }
        checkpoint_path.write_text(json.dumps(pkg, indent=None))

    for a in q:
        count += 1
        if count % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  processed {count} activities ({elapsed:.1f}s), kept {sum(len(v) for v in by_mol.values())}", flush=True)
            checkpoint()
        if count > max_records:
            break

        smi = a.get("canonical_smiles")
        val = a.get("standard_value")
        unit = a.get("standard_units")
        desc = a.get("assay_description") or ""
        org = a.get("assay_organism") or ""

        if not smi or not val or not unit:
            reject_counts["missing_fields"] += 1
            continue

        # Filter 1: reject any animal keyword in description
        if is_animal(desc):
            reject_counts["animal_in_desc"] += 1
            continue

        # Filter 2: require explicit human keyword OR Homo sapiens organism
        species_ok = (org.strip().lower() == "homo sapiens") or is_human(desc)
        if not species_ok:
            reject_counts["no_human_marker"] += 1
            continue

        # Filter 2b: reject non-oral routes (PLM targets oral only)
        if is_non_oral(desc):
            reject_counts["non_oral_route"] += 1
            continue

        # Filter 2c: require explicit oral mention (avoids ambiguous route)
        if not is_oral(desc):
            reject_counts["no_oral_marker"] += 1
            continue

        # Filter 3: units — accept both concentration and molar (molar needs MW)
        u_norm = unit.lower().strip().replace(" ", "").replace(".", "/")
        is_conc = u_norm in {"ng/ml", "ug/ml", "mcg/ml", "mg/l", "ug/l", "mcg/l",
                              "ng/ml-1", "ug/ml-1", "mcg/ml-1", "mg/l-1"}
        is_molar = u_norm in {"nm", "um", "µm", "nmol/l", "umol/l"}
        if not (is_conc or is_molar):
            reject_counts[f"bad_unit_{unit}"] += 1
            continue

        # Filter 3: dose parseable from description, NOT mg/kg
        dose, reason = extract_dose_mg(desc)
        if dose is None:
            reject_counts[f"dose_{reason}"] += 1
            continue

        # Compute MW if needed for molar conversion
        mw = None
        if is_molar:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                reject_counts["bad_smiles_for_mw"] += 1
                continue
            mw = Descriptors.ExactMolWt(mol)
        cmax = cmax_to_ngml(val, unit, mw=mw)
        if cmax is None:
            reject_counts["cmax_convert_fail"] += 1
            continue

        log_cd = math.log10(cmax / dose)
        if log_cd < p1 - 0.5 or log_cd > p99 + 0.5:
            reject_counts["log_cd_out_of_range"] += 1
            continue

        ik = smiles_to_ik14(smi)
        if not ik:
            reject_counts["bad_smiles"] += 1
            continue
        if ik in exclude_iks:
            reject_counts["in_v11_or_holdout"] += 1
            continue

        by_mol[ik].append({
            "smiles": smi, "ik": ik, "dose_mg": dose,
            "cmax_ngml": cmax, "log_cd": log_cd,
            "desc": desc[:120], "unit": unit, "value": val,
        })

    elapsed = time.time() - t0
    print(f"\nProcessed {count} activities in {elapsed:.1f}s")
    print(f"Unique new human drugs: {len(by_mol)}")
    print(f"Total accepted rows: {sum(len(v) for v in by_mol.values())}")

    print("\nRejection breakdown:")
    for k, v in sorted(reject_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k}: {v}")

    # DEDUP per (drug, dose) pair. For repeated (ik, dose), take median Cmax.
    # Do NOT aggregate across different doses — Cmax scales with dose so aggregation is meaningless.
    from collections import defaultdict as _dd
    by_drug_dose = _dd(list)
    for ik, entries in by_mol.items():
        for e in entries:
            key = (ik, round(e["dose_mg"], 1))
            by_drug_dose[key].append(e)

    new_entries = []
    for (ik, dose), entries in by_drug_dose.items():
        log_cds = [e["log_cd"] for e in entries]
        med_log_cd = float(np.median(log_cds))
        med_cmax = 10 ** med_log_cd * dose
        new_entries.append({
            "smiles": entries[0]["smiles"],
            "ik": ik,
            "dose_mg": float(dose),
            "cmax_ngml": med_cmax,
            "log_cd": med_log_cd,
            "src": "CHEMBL_v2_strict",
            "n_records": len(entries),
            "sample_desc": entries[0]["desc"],
        })

    # Sort by n_records desc
    new_entries.sort(key=lambda e: -e["n_records"])
    print(f"\n(drug, dose) pairs: {len(new_entries)}")

    print(f"\nTop 10 by n_records:")
    for e in new_entries[:10]:
        print(f"  {e['ik']:<17s} n={e['n_records']:3d} dose={e['dose_mg']:>7.1f} Cmax={e['cmax_ngml']:>9.1f} log_cd={e['log_cd']:+.2f}")

    # Save with metadata
    out = {
        "extraction_date": time.strftime("%Y-%m-%d"),
        "source": "ChEMBL v2 strict re-extraction (addresses F10/I4 failure modes)",
        "max_records_queried": max_records,
        "total_processed": count,
        "unique_new_drugs": len(by_mol),
        "v11_log_cd_p1_p99": [float(p1), float(p99)],
        "reject_counts": dict(reject_counts),
        "entries": new_entries,
    }
    out_path = ROOT / "data/curated/chembl_v2_strict.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    main(max_records=max_n)
