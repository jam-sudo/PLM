"""
C-t Profile Interpolator — Phase 5

Interpolates profiles to standard 13-point timepoint grid.
Computes log10(C/dose) for model training.

No extrapolation beyond observed time range.
C(0) = 0 → log_c_over_dose = NaN (not -inf).
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


ORAL_GRID = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]
IV_GRID = [0, 0.033, 0.083, 0.167, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24]


def interpolate_ct(timepoints: list[float], concentrations: list[float],
                   grid: list[float] = None, route: str = "oral") -> list[Optional[float]]:
    """
    Interpolate C-t profile to standard grid.
    Returns concentrations at grid points. NaN for out-of-range.
    """
    if grid is None:
        grid = IV_GRID if route == "IV" else ORAL_GRID

    t = np.array(timepoints, dtype=float)
    c = np.array(concentrations, dtype=float)

    # Remove NaN/inf
    valid = np.isfinite(t) & np.isfinite(c)
    t = t[valid]
    c = c[valid]

    if len(t) < 2:
        return [None] * len(grid)

    # Remove C=0 for log-linear interpolation (log10(0) undefined)
    nonzero = c > 0
    if np.sum(nonzero) < 2:
        return [None] * len(grid)

    t_nz = t[nonzero]
    c_nz = c[nonzero]
    log_c = np.log10(c_nz)

    # Build interpolator (no extrapolation)
    f = interp1d(t_nz, log_c, kind="linear", bounds_error=False, fill_value=np.nan)

    result = []
    for g in grid:
        if g < t_nz.min() or g > t_nz.max():
            result.append(None)  # No extrapolation
        else:
            val = f(g)
            if np.isfinite(val):
                result.append(round(float(10 ** val), 2))
            else:
                result.append(None)

    return result


def compute_log_c_over_dose(concentrations: list[Optional[float]],
                            dose_mg: Optional[float]) -> list[Optional[float]]:
    """Compute log10(C(t)/dose) at each grid point."""
    if dose_mg is None or dose_mg <= 0:
        return [None] * len(concentrations)

    result = []
    for c in concentrations:
        if c is None or c <= 0:
            result.append(None)
        else:
            result.append(round(math.log10(c / dose_mg), 4))
    return result


def compute_auc(timepoints: list[float], concentrations: list[float]) -> float:
    """Trapezoidal AUC from raw (not interpolated) data."""
    t = np.array(timepoints, dtype=float)
    c = np.array(concentrations, dtype=float)
    valid = np.isfinite(t) & np.isfinite(c) & (c >= 0)
    t = t[valid]
    c = c[valid]
    if len(t) < 2:
        return 0.0
    return float(np.trapz(c, t))


def interpolate_profiles(input_path: str, output_path: str) -> dict:
    """Interpolate all normalized profiles to standard grid."""
    with open(input_path) as f:
        profiles = json.load(f)

    interpolated = []
    for p in profiles:
        times = p.get("timepoints_h", [])
        concs = p.get("concentrations_ngml", [])
        route = p.get("route", "oral")
        dose = p.get("dose_mg")

        grid = IV_GRID if route == "IV" else ORAL_GRID
        grid_concs = interpolate_ct(times, concs, grid, route)
        log_c_dose = compute_log_c_over_dose(grid_concs, dose)
        auc = compute_auc(times, concs)

        p["grid_timepoints_h"] = grid
        p["grid_concentrations_ngml"] = grid_concs
        p["log_c_over_dose"] = log_c_dose
        p["auc_ngml_h"] = round(auc, 1)
        interpolated.append(p)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(interpolated, f, indent=2)

    n_with_data = sum(1 for p in interpolated
                      if any(v is not None for v in p["grid_concentrations_ngml"]))
    print(f"Interpolated {len(interpolated)} profiles ({n_with_data} with grid data)")
    return {"total": len(interpolated), "with_grid_data": n_with_data}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Normalized profiles JSON")
    parser.add_argument("--output", default="data/curated/interpolated_profiles.json")
    args = parser.parse_args()
    interpolate_profiles(args.input, args.output)
