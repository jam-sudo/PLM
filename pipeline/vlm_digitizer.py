"""
VLM-based C-t curve digitizer for failed auto-digitizer figures.

Uses Claude/Anthropic vision API to extract concentration-time data points
from pharmacokinetic figures that the OCR-based auto-digitizer couldn't process.

Usage:
    python -m pipeline.vlm_digitizer [--max N] [--dry-run]

Requires: ANTHROPIC_API_KEY environment variable
"""

from __future__ import annotations

import json
import base64
import os
import time
import math
from pathlib import Path
from typing import Optional

import numpy as np

GRID_H = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]

VLM_PROMPT = """You are a pharmacokinetics data extraction expert. Extract concentration-time data from this figure.

INSTRUCTIONS:
1. Identify if this is a plasma concentration vs time plot. If NOT a C-t plot, respond with: {"is_ct_plot": false, "reason": "..."}
2. If it IS a C-t plot, extract data points for the PRIMARY curve only (typically the drug alone, not DDI conditions).
3. Read the axis labels carefully for units and scale (linear vs log).
4. Extract (time, concentration) pairs by reading the data points/line from the figure.

OUTPUT FORMAT (JSON only, no markdown):
{
  "is_ct_plot": true,
  "drug_name": "<drug name from figure/legend>",
  "x_axis_label": "<exact x-axis label>",
  "y_axis_label": "<exact y-axis label>",
  "x_unit": "h",
  "y_unit": "ng/mL",
  "y_scale": "linear" or "log",
  "curve_label": "<which curve you extracted>",
  "n_subjects": <int or null>,
  "timepoints": [<list of time values in hours>],
  "concentrations": [<list of concentration values in stated units>],
  "confidence": "high" or "medium" or "low",
  "notes": "<any caveats>"
}

RULES:
- Extract at LEAST 6 data points (more is better)
- Include t=0 if visible
- Include Cmax timepoint
- Include terminal phase points
- If multiple curves (e.g., DDI study), extract the CONTROL/ALONE arm
- If y-axis is log scale, read the actual values (not log-transformed)
- Report concentration in the units shown on the y-axis
"""


def encode_image(image_path: str) -> tuple[str, str]:
    """Read and base64-encode an image file."""
    ext = os.path.splitext(image_path)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def extract_ct_from_figure(
    image_path: str,
    client,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Use Claude vision to extract C-t data from a figure."""
    img_data, media_type = encode_image(image_path)

    message = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data,
                    },
                },
                {"type": "text", "text": VLM_PROMPT},
            ],
        }],
    )

    # Parse response
    text = message.content[0].text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"is_ct_plot": False, "reason": f"JSON parse error: {text[:200]}"}

    result["source_image"] = image_path
    return result


def normalize_to_ngml(value: float, unit: str) -> Optional[float]:
    """Convert concentration to ng/mL."""
    unit = unit.lower().strip()
    conversions = {
        "ng/ml": 1.0,
        "ug/ml": 1000.0, "µg/ml": 1000.0, "mcg/ml": 1000.0,
        "mg/l": 1000.0,
        "ug/l": 1.0, "µg/l": 1.0,
        "pg/ml": 0.001,
        "ng/dl": 0.01,
    }
    factor = conversions.get(unit)
    if factor is None:
        return None
    return value * factor


def process_extraction(result: dict) -> Optional[dict]:
    """Convert VLM extraction to PLM-compatible format."""
    if not result.get("is_ct_plot"):
        return None

    times = result.get("timepoints", [])
    concs = result.get("concentrations", [])
    if len(times) < 4 or len(concs) < 4 or len(times) != len(concs):
        return None

    # Convert to ng/mL
    y_unit = result.get("y_unit", "ng/mL")
    concs_ngml = []
    for c in concs:
        converted = normalize_to_ngml(c, y_unit)
        if converted is None:
            return None
        concs_ngml.append(converted)

    # Interpolate to standard grid
    from scipy.interpolate import interp1d
    times_arr = np.array(times, dtype=float)
    concs_arr = np.array(concs_ngml, dtype=float)

    # Sort by time
    sort_idx = np.argsort(times_arr)
    times_arr = times_arr[sort_idx]
    concs_arr = concs_arr[sort_idx]

    # Interpolate (linear, no extrapolation)
    max_t = times_arr[-1]
    grid = [t for t in GRID_H if t <= max_t]
    if len(grid) < 4:
        return None

    f_interp = interp1d(times_arr, concs_arr, kind="linear", fill_value=0, bounds_error=False)
    grid_concs = f_interp(grid).tolist()
    grid_concs = [max(0, c) for c in grid_concs]

    cmax = max(concs_ngml)
    tmax = times[concs_ngml.index(cmax)]

    return {
        "drug_name": result.get("drug_name", "unknown"),
        "timepoints_h": grid,
        "concentrations_ngml": [round(c, 2) for c in grid_concs],
        "cmax_ngml": round(cmax, 2),
        "tmax_h": round(tmax, 2),
        "concentration_unit": "ng/mL",
        "source_image": result.get("source_image"),
        "source_method": "VLM_digitization",
        "vlm_confidence": result.get("confidence", "medium"),
        "curve_label": result.get("curve_label"),
        "n_raw_points": len(times),
        "n_grid_points": len(grid),
        "y_unit_original": y_unit,
        "notes": result.get("notes", ""),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=5, help="Max figures to process")
    parser.add_argument("--dry-run", action="store_true", help="Just list candidates, don't call API")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    # Load C-t candidates
    with open("/tmp/ct_candidates.json") as f:
        ct_paths = json.load(f)

    print(f"C-t curve candidates: {len(ct_paths)}")
    print(f"Processing: {'DRY RUN' if args.dry_run else f'up to {args.max}'}")

    if args.dry_run:
        for p in ct_paths[:20]:
            print(f"  {p}")
        return

    # Initialize Anthropic client
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"ERROR: Cannot initialize Anthropic client: {e}")
        print("Set ANTHROPIC_API_KEY environment variable")
        return

    results = []
    profiles = []
    n_processed = 0
    n_ct = 0
    n_not_ct = 0
    n_error = 0

    for path in ct_paths[:args.max]:
        if not os.path.exists(path):
            continue

        print(f"\n--- Processing: {path} ---")
        try:
            result = extract_ct_from_figure(path, client, model=args.model)
            results.append(result)

            if result.get("is_ct_plot"):
                profile = process_extraction(result)
                if profile:
                    profiles.append(profile)
                    n_ct += 1
                    print(f"  OK: {profile['drug_name']}, Cmax={profile['cmax_ngml']} ng/mL, {profile['n_raw_points']} points")
                else:
                    n_error += 1
                    print(f"  EXTRACTED but processing failed")
            else:
                n_not_ct += 1
                print(f"  NOT C-t: {result.get('reason', '?')}")

        except Exception as e:
            n_error += 1
            print(f"  ERROR: {e}")

        n_processed += 1
        time.sleep(1)  # Rate limit

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Processed: {n_processed}")
    print(f"C-t extracted: {n_ct}")
    print(f"Not C-t: {n_not_ct}")
    print(f"Errors: {n_error}")

    # Save
    output_dir = "data/digitized/vlm"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{output_dir}/vlm_extractions.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(f"{output_dir}/vlm_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
