"""Generate synthetic C-t profiles from DrugBank PK parameters.

Uses 1-compartment oral model:
    ke = 0.693 / t1/2
    ka = estimated (~1.5/h for oral)
    C(t) = (Dose * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

Target is log10(C_ngml / dose_mg), which is dose-invariant for linear PK.

Usage:
    python -m pipeline.synthetic_ct
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# PLM standard 13-point grid
GRID_H = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24], dtype=float)

# Physiological bounds for quality filtering
BOUNDS = {
    "t_half_h": (0.1, 500),       # 6 min to 21 days
    "vd_L": (1.0, 5000),          # 1L to 5000L
    "cl_Lh": (0.01, 500),         # 0.01 to 500 L/h
    "ke": (0.001, 7.0),           # t1/2 ~6min to ~700h
    "ka": (0.3, 5.0),             # oral absorption range
}

DEFAULT_WEIGHT_KG = 70
REFERENCE_DOSE_MG = 100  # arbitrary, cancels in log10(C/dose)
DEFAULT_KA = 1.5  # /h, typical oral absorption


def load_drugbank(
    pk_path: str = "/home/jam/Sisyphus/data/drugbank/pk_data.csv",
    drugs_path: str = "/home/jam/Sisyphus/data/drugbank/drugs.csv",
) -> pd.DataFrame:
    """Load and pivot DrugBank PK data, merge with drug SMILES."""
    pk = pd.read_csv(pk_path)
    drugs = pd.read_csv(drugs_path)

    # Pivot PK fields per drug
    records = []
    for dbid, grp in pk.groupby("drugbank_id"):
        rec = {"drugbank_id": dbid, "drug_name": grp["drug_name"].iloc[0]}
        for _, row in grp.iterrows():
            field = row["field"]
            val = row["parsed_value"]
            unit = row.get("parsed_unit", "")
            if pd.isna(val):
                continue
            if field == "half_life":
                rec["t_half_h"] = val
            elif field == "volume_of_distribution":
                if "L/kg" in str(unit):
                    rec["vd_L"] = val * DEFAULT_WEIGHT_KG
                    rec["vd_source"] = "L/kg"
                else:
                    rec["vd_L"] = val
                    rec["vd_source"] = "L"
            elif field == "clearance":
                rec["cl_Lh"] = val  # already normalized to L/h
            elif field == "protein_binding":
                rec["fup"] = 1.0 - val / 100.0 if val <= 100 else None
        records.append(rec)

    df = pd.DataFrame(records)

    # Merge SMILES
    df = df.merge(
        drugs[["drugbank_id", "smiles", "inchikey_14", "mw", "logp_calc", "state", "groups"]],
        on="drugbank_id",
        how="left",
    )
    return df


def generate_synthetic_profile(
    t_half_h: float,
    vd_L: float,
    cl_Lh: float | None = None,
    ka: float = DEFAULT_KA,
    dose_mg: float = REFERENCE_DOSE_MG,
    tlag_h: float = 0.25,
) -> dict | None:
    """Generate a synthetic 1-compartment oral C-t profile.

    Returns dict with timepoints_h, concentrations_ngml, and PK params,
    or None if parameters are non-physical.
    """
    # Derive ke
    if t_half_h <= 0:
        return None
    ke = 0.693 / t_half_h

    # Cross-validate with CL if available
    if cl_Lh is not None and cl_Lh > 0 and vd_L > 0:
        ke_from_cl = cl_Lh / vd_L
        # If >5x discrepancy, flag but use t1/2-derived ke (more reliable)
        ratio = ke / ke_from_cl if ke_from_cl > 0 else 999
        if ratio > 5 or ratio < 0.2:
            return None  # PK params internally inconsistent

    # Bounds check
    if not (BOUNDS["ke"][0] <= ke <= BOUNDS["ke"][1]):
        return None
    if not (BOUNDS["vd_L"][0] <= vd_L <= BOUNDS["vd_L"][1]):
        return None

    # Ensure ka != ke (degenerate case)
    if abs(ka - ke) < 1e-6:
        ka = ke * 1.1

    # Generate C(t) on standard grid
    t_eff = np.maximum(0.0, GRID_H - tlag_h)
    coeff = (dose_mg * ka) / (vd_L * (ka - ke))
    c_mgL = coeff * (np.exp(-ke * t_eff) - np.exp(-ka * t_eff))
    c_mgL[GRID_H < tlag_h] = 0.0
    c_mgL = np.maximum(0.0, c_mgL)  # no negative concentrations

    # Convert mg/L -> ng/mL
    c_ngml = c_mgL * 1000.0

    # Sanity: Cmax should be > 0
    if np.max(c_ngml) <= 0:
        return None

    # Compute log10(C/dose) target
    target = np.full(len(GRID_H), np.nan)
    for i, c in enumerate(c_ngml):
        if c > 0:
            target[i] = math.log10(c / dose_mg)

    return {
        "timepoints_h": GRID_H.tolist(),
        "concentrations_ngml": [round(c, 4) for c in c_ngml],
        "target_log_cd": [round(t, 6) if not np.isnan(t) else None for t in target],
        "cmax_ngml": round(float(np.max(c_ngml)), 4),
        "tmax_h": float(GRID_H[np.argmax(c_ngml)]),
        "auc_ngml_h": round(float(np.trapz(c_ngml, GRID_H)), 4),
        "pk_params": {
            "ka": round(ka, 4),
            "ke": round(ke, 6),
            "vd_L": round(vd_L, 2),
            "t_half_h": round(t_half_h, 4),
            "cl_Lh": round(cl_Lh, 4) if cl_Lh else None,
            "tlag_h": tlag_h,
        },
    }


def build_synthetic_dataset(
    output_path: str = "data/curated/synthetic_drugbank_ct.json",
    require_cl: bool = False,
) -> dict:
    """Build synthetic C-t profiles from DrugBank PK parameters."""
    print("Loading DrugBank data...")
    df = load_drugbank()

    # Filter: need t1/2 + Vd + SMILES at minimum
    mask = (
        df["t_half_h"].notna()
        & df["vd_L"].notna()
        & df["smiles"].notna()
        & (df["t_half_h"] > 0)
        & (df["vd_L"] > 0)
    )
    if require_cl:
        mask = mask & df["cl_Lh"].notna() & (df["cl_Lh"] > 0)

    candidates = df[mask].copy()
    print(f"Candidates with t1/2 + Vd + SMILES: {len(candidates)}")

    # Apply physiological bounds
    candidates = candidates[
        candidates["t_half_h"].between(*BOUNDS["t_half_h"])
        & candidates["vd_L"].between(*BOUNDS["vd_L"])
    ]
    print(f"After physiological bounds: {len(candidates)}")

    # Filter out biologics (MW > 1500 or non-solid)
    if "mw" in candidates.columns:
        candidates = candidates[
            candidates["mw"].isna() | (candidates["mw"] <= 1500)
        ]
        print(f"After MW <= 1500 (small molecules): {len(candidates)}")

    # Generate profiles
    profiles = []
    skipped = {"inconsistent_pk": 0, "zero_cmax": 0, "other": 0}

    for _, row in candidates.iterrows():
        cl = row.get("cl_Lh") if pd.notna(row.get("cl_Lh")) else None

        result = generate_synthetic_profile(
            t_half_h=row["t_half_h"],
            vd_L=row["vd_L"],
            cl_Lh=cl,
            ka=DEFAULT_KA,
            dose_mg=REFERENCE_DOSE_MG,
        )

        if result is None:
            skipped["inconsistent_pk"] += 1
            continue

        profile = {
            "drug_name": row["drug_name"],
            "drugbank_id": row["drugbank_id"],
            "smiles": row["smiles"],
            "inchikey_14": row.get("inchikey_14"),
            "dose_mg": REFERENCE_DOSE_MG,
            "route": "oral",
            "formulation": "IR_tablet",
            "food_effect": "not_specified",
            "source": "drugbank_synthetic",
            "mw": round(row["mw"], 2) if pd.notna(row.get("mw")) else None,
            **result,
        }
        profiles.append(profile)

    print(f"\nGenerated: {len(profiles)} synthetic profiles")
    print(f"Skipped: {skipped}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved to {output_path}")

    # Summary stats
    cmaxes = [p["cmax_ngml"] for p in profiles]
    thalfs = [p["pk_params"]["t_half_h"] for p in profiles]
    vds = [p["pk_params"]["vd_L"] for p in profiles]

    summary = {
        "n_profiles": len(profiles),
        "n_unique_drugs": len(set(p["drugbank_id"] for p in profiles)),
        "n_with_smiles": sum(1 for p in profiles if p["smiles"]),
        "skipped": skipped,
        "cmax_ngml": {
            "median": round(float(np.median(cmaxes)), 2),
            "p5": round(float(np.percentile(cmaxes, 5)), 2),
            "p95": round(float(np.percentile(cmaxes, 95)), 2),
        },
        "t_half_h": {
            "median": round(float(np.median(thalfs)), 2),
            "p5": round(float(np.percentile(thalfs, 5)), 2),
            "p95": round(float(np.percentile(thalfs, 95)), 2),
        },
        "vd_L": {
            "median": round(float(np.median(vds)), 2),
            "p5": round(float(np.percentile(vds, 5)), 2),
            "p95": round(float(np.percentile(vds, 95)), 2),
        },
    }

    print(f"\n=== Summary ===")
    print(f"Profiles: {summary['n_profiles']}")
    print(f"Unique drugs: {summary['n_unique_drugs']}")
    print(f"Cmax median: {summary['cmax_ngml']['median']:.1f} ng/mL (5-95%: {summary['cmax_ngml']['p5']:.1f}-{summary['cmax_ngml']['p95']:.1f})")
    print(f"t1/2 median: {summary['t_half_h']['median']:.1f}h (5-95%: {summary['t_half_h']['p5']:.1f}-{summary['t_half_h']['p95']:.1f})")
    print(f"Vd median: {summary['vd_L']['median']:.1f}L (5-95%: {summary['vd_L']['p5']:.1f}-{summary['vd_L']['p95']:.1f})")

    return summary


if __name__ == "__main__":
    build_synthetic_dataset()
