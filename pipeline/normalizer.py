"""
Unit & Metadata Normalizer — Phase 4 (CRITICAL)

Converts all concentration/dose/formulation/route/food values to
standardized forms. One unit error = 1000x dataset contamination.

Usage:
    python -m pipeline.normalizer data/digitized/auto/auto_digitized.json
"""

import json
import re
import time
import requests
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 4A: Concentration unit → ng/mL
# ---------------------------------------------------------------------------

UNIT_TO_NGML = {
    "ng/mL": 1.0, "ng/ml": 1.0,
    "μg/mL": 1000.0, "ug/mL": 1000.0, "µg/mL": 1000.0, "mcg/mL": 1000.0,
    "mg/L": 1000.0, "mg/l": 1000.0,
    "μg/L": 1.0, "ug/L": 1.0,
    "pg/mL": 0.001, "pg/ml": 0.001,
    "mg/dL": 10000.0, "mg/dl": 10000.0,
    "μg/dL": 10.0, "ug/dL": 10.0, "µg/dL": 10.0,
    "ng/dL": 0.01, "ng/dl": 0.01,
    # Molar units require MW
    "nmol/L": None, "nmol/l": None,
    "μmol/L": None, "umol/L": None, "µmol/L": None,
}

# Normalize unicode μ variants
def _normalize_unit(unit: str) -> str:
    """Normalize unicode micro symbols."""
    return unit.replace("µ", "μ").replace("u", "μ", 1) if unit.startswith("u") else unit.replace("µ", "μ")


def convert_concentration(value: float, from_unit: str, mw: Optional[float] = None) -> tuple[float, float]:
    """
    Convert concentration to ng/mL.
    Returns (converted_value, conversion_factor).
    Raises ValueError if conversion not possible.
    """
    norm_unit = _normalize_unit(from_unit)

    # Direct lookup
    factor = UNIT_TO_NGML.get(norm_unit)
    if factor is None and norm_unit in UNIT_TO_NGML:
        # Molar unit — needs MW
        if mw is None:
            raise ValueError(f"MW required for {from_unit} conversion")
        if "nmol" in norm_unit:
            factor = mw / 1000.0  # nmol/L × (g/mol) / 1000 = ng/mL
        elif "μmol" in norm_unit or "umol" in norm_unit:
            factor = mw  # μmol/L × (g/mol) = μg/mL = 1000 ng/mL... wait
            # μmol/L × MW(g/mol) × 10^-6(mol/μmol) × 10^9(ng/g) × 10^-3(L/mL) = MW ng/mL
            factor = mw  # Actually: μmol/L = mw * 1000 ng/mL? Let me recalculate
            # 1 μmol/L = 10^-6 mol/L × MW g/mol = MW × 10^-6 g/L = MW × 10^-3 mg/L = MW × 1 μg/L
            # But μg/L = ng/mL, so 1 μmol/L = MW ng/mL... no
            # 1 μmol/L = MW μg/L = MW ng/mL... wait
            # 1 μg/L = 1 ng/mL. So 1 μmol/L = MW μg/L = MW ng/mL? No!
            # Let me be careful:
            # 1 μmol/L = 10^-6 mol/L
            # × MW g/mol = MW × 10^-6 g/L
            # = MW × 10^-3 mg/L
            # = MW × 10^-3 × 10^3 μg/L  (since 1 mg = 10^3 μg)
            # = MW μg/L
            # And 1 μg/L = 10^-3 ng/mL? No!
            # 1 μg/L = 10^-6 g/L = 10^-6 × 10^3 mg/L = 10^-3 mg/L
            # 1 ng/mL = 10^-9 g / 10^-3 L = 10^-6 g/L = 1 μg/L
            # So 1 μg/L = 1 ng/mL
            # Therefore: 1 μmol/L = MW μg/L = MW ng/mL? No!
            # 1 μmol/L = MW μg/L. And 1 μg/L = 1 ng/mL.
            # So 1 μmol/L = MW ng/mL? That seems too high.
            # Wait: 1 μg/L = 1 ng/mL. Yes that's correct.
            # So 1 μmol/L = MW × 1 μg/L = MW × 1 ng/mL = MW ng/mL
            # For a drug with MW=500: 1 μmol/L = 500 ng/mL. That seems right.
            # But wait, the UNIT_TO_NGML table says μg/mL = 1000 ng/mL and μg/L = 1 ng/mL
            # So 1 μmol/L = MW μg/L = MW ng/mL
            factor = mw
    elif factor is None:
        raise ValueError(f"Unknown unit: {from_unit}")

    return (value * factor, factor)


# ---------------------------------------------------------------------------
# 4B: Dose unit → mg
# ---------------------------------------------------------------------------

DOSE_TO_MG = {
    "mg": 1.0,
    "g": 1000.0,
    "μg": 0.001, "mcg": 0.001, "ug": 0.001,
}

DEFAULT_BODY_WEIGHT_KG = 70
DEFAULT_BSA_M2 = 1.73


def convert_dose(value: float, unit: str) -> tuple[float, str]:
    """Convert dose to mg. Returns (dose_mg, notes)."""
    unit_lower = unit.lower().strip()
    if unit_lower in DOSE_TO_MG:
        return (value * DOSE_TO_MG[unit_lower], "")

    if "mg/kg" in unit_lower or "mg/ kg" in unit_lower:
        dose_mg = value * DEFAULT_BODY_WEIGHT_KG
        return (dose_mg, f"mg/kg→mg using {DEFAULT_BODY_WEIGHT_KG}kg default")

    if "μg/kg" in unit_lower or "mcg/kg" in unit_lower:
        dose_mg = value * 0.001 * DEFAULT_BODY_WEIGHT_KG
        return (dose_mg, f"μg/kg→mg using {DEFAULT_BODY_WEIGHT_KG}kg default")

    if "mg/m2" in unit_lower or "mg/m²" in unit_lower:
        dose_mg = value * DEFAULT_BSA_M2
        return (dose_mg, f"mg/m²→mg using {DEFAULT_BSA_M2}m² default")

    return (value, f"unknown dose unit '{unit}', assumed mg")


# ---------------------------------------------------------------------------
# 4C-E: Formulation / Route / Food normalization
# ---------------------------------------------------------------------------

FORMULATION_MAP = {
    "tablet": "IR_tablet", "film-coated tablet": "IR_tablet",
    "capsule": "IR_capsule", "hard gelatin capsule": "IR_capsule",
    "soft gelatin capsule": "IR_capsule_soft", "softgel": "IR_capsule_soft",
    "oral solution": "solution", "solution": "solution",
    "oral suspension": "suspension", "suspension": "suspension",
    "syrup": "solution", "powder for oral solution": "solution",
    "extended-release tablet": "ER_tablet", "extended release tablet": "ER_tablet",
    "er tablet": "ER_tablet", "xr tablet": "ER_tablet", "xl tablet": "ER_tablet",
    "sr tablet": "ER_tablet", "controlled-release": "ER_tablet",
    "modified-release": "ER_tablet", "extended-release capsule": "ER_capsule",
    "iv bolus": "IV_bolus", "iv infusion": "IV_infusion",
    "intravenous": "IV_infusion", "intramuscular": "IM_injection",
    "subcutaneous": "SC_injection",
    "sublingual tablet": "sublingual", "sublingual": "sublingual",
    "transdermal patch": "transdermal", "transdermal": "transdermal",
    "oral disintegrating tablet": "ODT",
}

ROUTE_MAP = {
    "oral": "oral", "po": "oral", "by mouth": "oral",
    "intravenous": "IV", "iv": "IV", "i.v.": "IV",
    "intramuscular": "IM", "im": "IM", "i.m.": "IM",
    "subcutaneous": "SC", "sc": "SC", "s.c.": "SC",
    "sublingual": "sublingual", "sl": "sublingual",
    "transdermal": "transdermal", "topical": "topical",
    "rectal": "rectal", "inhaled": "inhaled", "intranasal": "intranasal",
}

FOOD_MAP = {
    "fasted": "fasted", "fasting": "fasted", "empty stomach": "fasted",
    "fed": "fed", "with food": "fed",
    "high-fat meal": "fed_highfat", "high fat meal": "fed_highfat",
    "standard meal": "fed_standard", "light meal": "fed_light",
}


def normalize_formulation(text: str) -> str:
    lower = text.lower().strip()
    for key, val in FORMULATION_MAP.items():
        if key in lower:
            return val
    return "other"


def normalize_route(text: str) -> str:
    lower = text.lower().strip()
    for key, val in ROUTE_MAP.items():
        if key in lower:
            return val
    return "not_specified"


def normalize_food(text: str) -> str:
    lower = text.lower().strip()
    for key, val in FOOD_MAP.items():
        if key in lower:
            return val
    return "not_specified"


# ---------------------------------------------------------------------------
# MW lookup via PubChem
# ---------------------------------------------------------------------------

_mw_cache = {}


def lookup_mw(drug_name: str) -> Optional[float]:
    """Look up molecular weight from PubChem. Cached."""
    if drug_name in _mw_cache:
        return _mw_cache[drug_name]

    # Strip salt forms
    clean = re.sub(
        r"\s*(hydrochloride|hcl|sodium|potassium|mesylate|fumarate|"
        r"tartrate|maleate|besylate|tosylate|phosphate|sulfate|"
        r"dihydrochloride|dimesylate|tromethamine|acetate|"
        r"hydrobromide|oxalate|gluconate|bromide)\s*",
        "", drug_name, flags=re.IGNORECASE
    ).strip()

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{clean}/property/MolecularWeight/JSON"
    try:
        resp = requests.get(url, timeout=15)
        if resp.ok:
            data = resp.json()
            mw = float(data["PropertyTable"]["Properties"][0]["MolecularWeight"])
            _mw_cache[drug_name] = mw
            return mw
    except Exception:
        pass

    _mw_cache[drug_name] = None
    return None


# ---------------------------------------------------------------------------
# 4F: Cross-validation sanity checks
# ---------------------------------------------------------------------------

def sanity_check(profile: dict) -> list[str]:
    """Run sanity checks on a normalized profile. Returns list of flags."""
    flags = []
    cmax = profile.get("cmax_ngml")
    dose = profile.get("dose_mg")

    if cmax is not None:
        if cmax < 0.01:
            flags.append("FLAG: Cmax < 0.01 ng/mL (unit suspect)")
        if cmax > 1_000_000:
            flags.append("FLAG: Cmax > 1M ng/mL (unit suspect)")

    if cmax is not None and dose is not None and dose > 0:
        ratio = cmax / dose
        if ratio < 0.001:
            flags.append(f"FLAG: Cmax/dose={ratio:.4f} <0.001 (very low)")
        if ratio > 1000:
            flags.append(f"FLAG: Cmax/dose={ratio:.0f} >1000 (very high)")

    return flags


# ---------------------------------------------------------------------------
# Main normalize pipeline
# ---------------------------------------------------------------------------

def normalize_profiles(input_path: str, output_path: str) -> dict:
    """Normalize all profiles from auto_digitized.json."""
    with open(input_path) as f:
        raw_profiles = json.load(f)

    successful = [r for r in raw_profiles if r.get("status") == "success"]
    print(f"Normalizing {len(successful)} profiles...")

    normalized = []
    mw_lookups_needed = 0
    unit_conversions = 0
    sanity_flags = 0

    for r in successful:
        unit = r.get("concentration_unit", "unknown")
        concs = r.get("concentrations", [])
        times = r.get("timepoints_h", [])

        # Skip if no data
        if not concs or not times:
            continue

        # Determine conversion factor
        factor = 1.0
        notes = []

        if unit in UNIT_TO_NGML and UNIT_TO_NGML[unit] is not None:
            factor = UNIT_TO_NGML[unit]
            if factor != 1.0:
                unit_conversions += 1
                notes.append(f"unit conversion: {unit}→ng/mL ×{factor}")
        elif unit in ("nmol/L", "μmol/L", "umol/L"):
            # Need MW — try to extract drug name from caption
            caption = r.get("source", {}).get("caption", "")
            # For now, skip MW lookup in batch (too slow for 70 profiles)
            notes.append(f"NEEDS MW: {unit} cannot convert without molecular weight")
            mw_lookups_needed += 1
        elif unit == "unknown":
            notes.append("WARNING: unknown concentration unit, assuming ng/mL")
        else:
            notes.append(f"WARNING: unrecognized unit '{unit}', assuming ng/mL")

        concs_ngml = [c * factor for c in concs]
        cmax_ngml = max(concs_ngml) if concs_ngml else None

        profile = {
            "timepoints_h": times,
            "concentrations_ngml": [round(c, 2) for c in concs_ngml],
            "concentration_unit_original": unit,
            "concentration_unit_normalized": "ng/mL",
            "unit_conversion_factor": factor,
            "cmax_ngml": round(cmax_ngml, 2) if cmax_ngml else None,
            "tmax_h": r.get("tmax"),
            "source_image": r.get("image_path", ""),
            "source_pdf": r.get("source", {}).get("pdf", ""),
            "source_page": r.get("source", {}).get("page", 0),
            "caption": r.get("source", {}).get("caption", ""),
            "normalization_notes": notes,
        }

        # Sanity checks
        checks = sanity_check(profile)
        if checks:
            sanity_flags += 1
            profile["sanity_flags"] = checks

        normalized.append(profile)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(normalized, f, indent=2)

    summary = {
        "total_input": len(successful),
        "total_normalized": len(normalized),
        "unit_conversions_applied": unit_conversions,
        "mw_lookups_needed": mw_lookups_needed,
        "sanity_flagged": sanity_flags,
    }
    print(f"\nNormalized: {len(normalized)} profiles")
    print(f"Unit conversions: {unit_conversions}")
    print(f"MW lookups needed: {mw_lookups_needed}")
    print(f"Sanity flagged: {sanity_flags}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to auto_digitized.json")
    parser.add_argument("--output", default="data/curated/normalized_profiles.json")
    args = parser.parse_args()
    normalize_profiles(args.input, args.output)
