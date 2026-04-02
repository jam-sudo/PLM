"""
C-t Profile Digitizer

Extracts (time, concentration) data points from C-t profile images.
Currently uses LLM visual analysis; future: automated axis OCR + curve tracing.

Usage:
    python -m pipeline.digitizer data/figures/NDA208658/p011_img00.png --drug empagliflozin
"""

import json
from pathlib import Path
from typing import Optional


def create_profile_template(
    drug_name: str,
    smiles: Optional[str] = None,
    dose_mg: Optional[float] = None,
    route: str = "oral",
    formulation: str = "IR_tablet",
    food_effect: str = "not_specified",
    population: str = "healthy_adult",
    n_subjects: Optional[int] = None,
    timepoints_h: Optional[list[float]] = None,
    concentrations: Optional[list[float]] = None,
    concentration_unit: str = "ng/mL",
    source_nda: str = "",
    source_page: int = 0,
    source_figure: str = "",
    curve_label: str = "",
    digitization_method: str = "visual_llm",
    notes: str = "",
) -> dict:
    """Create a standardized C-t profile data structure."""
    conc_key = f"concentrations_{concentration_unit.replace('/', '_').lower()}"

    profile = {
        "drug_name": drug_name,
        "smiles": smiles,
        "dose_mg": dose_mg,
        "route": route,
        "formulation": formulation,
        "food_effect": food_effect,
        "population": population,
        "n_subjects": n_subjects,
        "timepoints_h": timepoints_h or [],
        conc_key: concentrations or [],
        "concentration_unit": concentration_unit,
        "cmax_digitized": max(concentrations) if concentrations else None,
        "tmax_digitized": (
            timepoints_h[concentrations.index(max(concentrations))]
            if timepoints_h and concentrations
            else None
        ),
        "source_nda": source_nda,
        "source_page": source_page,
        "source_figure": source_figure,
        "curve_label": curve_label,
        "digitization_method": digitization_method,
        "digitization_notes": notes,
        "qc_status": "pending",
    }
    return profile


def save_profile(profile: dict, outdir: Path) -> Path:
    """Save a digitized profile to JSON."""
    outdir.mkdir(parents=True, exist_ok=True)
    nda = profile.get("source_nda", "unknown")
    drug = profile.get("drug_name", "unknown")
    fname = f"{nda}_{drug}.json"
    fpath = outdir / fname

    with open(fpath, "w") as f:
        json.dump(profile, f, indent=2)
    return fpath


def validate_profile(profile: dict) -> list[str]:
    """Basic validation of a digitized profile."""
    issues = []
    tp = profile.get("timepoints_h", [])

    # Find concentration key
    conc = None
    for key in profile:
        if key.startswith("concentrations_"):
            conc = profile[key]
            break

    if not tp:
        issues.append("No timepoints")
    if not conc:
        issues.append("No concentrations")
    if tp and conc and len(tp) != len(conc):
        issues.append(f"Timepoint/concentration length mismatch: {len(tp)} vs {len(conc)}")
    if tp and tp != sorted(tp):
        issues.append("Timepoints not monotonically increasing")
    if not profile.get("smiles") and profile.get("route") != "SC_injection":
        issues.append("Missing SMILES")
    if not profile.get("dose_mg"):
        issues.append("Missing dose")
    if len(tp) < 5:
        issues.append(f"Too few timepoints ({len(tp)}); minimum 5 recommended")

    return issues


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("figure", type=str, help="Path to C-t profile image")
    parser.add_argument("--drug", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="data/digitized/feasibility_samples/")
    args = parser.parse_args()

    print(f"Digitizer: {args.figure}")
    print(f"Drug: {args.drug}")
    print()
    print("NOTE: Current digitization requires visual LLM analysis.")
    print("Open the image in Claude Code and manually extract data points.")
    print(f"Save output to {args.outdir}")
