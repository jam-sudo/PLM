"""
Automated C-t Profile Digitizer

Extracts (time, concentration) data from C-t profile images using:
- pytesseract for axis label reading
- OpenCV for curve tracing

Usage:
    python -m pipeline.auto_digitizer data/figures/NDA208658/p011_img00.png
    python -m pipeline.auto_digitizer --batch data/figures/ct_candidates_scaleup.json
"""

import json
import re
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# Lazy-load easyocr reader
_ocr_reader = None


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


# ---------------------------------------------------------------------------
# 1. OCR
# ---------------------------------------------------------------------------

def ocr_image(image_path: str) -> list[dict]:
    """Run easyocr, return text regions with pixel positions."""
    reader = _get_reader()
    results = reader.readtext(image_path)
    regions = []
    for bbox, text, conf in results:
        cx = (bbox[0][0] + bbox[2][0]) / 2
        cy = (bbox[0][1] + bbox[2][1]) / 2
        regions.append({
            "text": text.strip(),
            "cx": cx,
            "cy": cy,
            "x1": min(p[0] for p in bbox),
            "y1": min(p[1] for p in bbox),
            "x2": max(p[0] for p in bbox),
            "y2": max(p[1] for p in bbox),
            "conf": conf,
        })
    return regions


# ---------------------------------------------------------------------------
# 2. Number parsing
# ---------------------------------------------------------------------------

def parse_number(text: str) -> Optional[float]:
    """Parse a numeric value from OCR text. Strict: reject text with letters."""
    text = text.strip()
    if not text or len(text) > 10:
        return None
    # STRICT: reject if text contains letters (catches "mg/1000", "(N=30)", etc.)
    # Allow only digits, dots, minus, commas, spaces, and common OCR noise (|-)
    alpha_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    if alpha_count > 0 or digit_count == 0:
        return None
    text = text.replace(",", "").replace(" ", "")
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 3. Axis detection
# ---------------------------------------------------------------------------

def detect_axes(ocr_regions: list[dict], img_w: int, img_h: int) -> dict:
    """
    Detect x-axis (time) and y-axis (concentration) tick labels.

    Strategy:
    - Y-axis labels: left 20% of image, vertically spread, x-aligned
    - X-axis labels: bottom 35% of image, horizontally spread, y-aligned
    - Validate: y-axis values should DECREASE as pixel-y increases
    - Validate: x-axis values should INCREASE as pixel-x increases
    """
    left_thresh = img_w * 0.20
    bottom_thresh = img_h * 0.65

    # Collect numeric regions
    y_cands = []  # potential y-axis labels
    x_cands = []  # potential x-axis labels

    for r in ocr_regions:
        val = parse_number(r["text"])
        if val is None:
            continue
        # Y-axis: left side
        if r["cx"] < left_thresh:
            y_cands.append({"px": r["cy"], "val": val, "cx": r["cx"], "text": r["text"]})
        # X-axis: bottom side
        if r["cy"] > bottom_thresh:
            x_cands.append({"px": r["cx"], "val": val, "cy": r["cy"], "text": r["text"]})

    # --- Y-axis: cluster by x-position (should all be at similar x) ---
    y_ticks = _filter_axis_ticks(y_cands, axis="y")
    # --- X-axis: cluster by y-position (should all be at similar y) ---
    x_ticks = _filter_axis_ticks(x_cands, axis="x")

    # Validate monotonicity
    # Y-axis: as pixel increases, value should decrease (image y is inverted)
    y_ticks = _validate_monotonic(y_ticks, decreasing=True)
    # X-axis: as pixel increases, value should increase
    x_ticks = _validate_monotonic(x_ticks, decreasing=False)

    # Extra validation: check for roughly uniform spacing in data values
    x_ticks = _validate_uniform_spacing(x_ticks)
    y_ticks = _validate_uniform_spacing(y_ticks)

    # Plot region: EXTEND beyond tick marks to capture full curve
    # The curve often starts before the first x-tick and extends below the last y-tick
    plot_region = None
    if len(x_ticks) >= 2 and len(y_ticks) >= 2:
        # Build preliminary transforms to extrapolate bounds
        x_tf = build_transform(x_ticks)
        y_tf = build_transform(y_ticks)

        if x_tf and y_tf:
            # Extrapolate: where would time=0 be? where would conc=0 be?
            # pixel = (value - offset) / scale
            x_at_zero = max(0, (0 - x_tf[1]) / x_tf[0]) if x_tf[0] != 0 else 0
            y_at_zero = min(img_h, (0 - y_tf[1]) / y_tf[0]) if y_tf[0] != 0 else img_h

            plot_region = (
                max(0, x_at_zero - 10),        # left: time=0 with margin
                max(0, min(t[0] for t in y_ticks) - 10),  # top: highest tick with margin
                min(img_w, max(t[0] for t in x_ticks) + 20),  # right: last tick with margin
                min(img_h * 0.85, y_at_zero + 10),  # bottom: conc=0 with margin
            )

    return {"x_ticks": x_ticks, "y_ticks": y_ticks, "plot_region": plot_region}


def _filter_axis_ticks(cands: list[dict], axis: str) -> list[tuple]:
    """
    Filter axis tick candidates by alignment clustering.
    Returns [(pixel_pos, data_value), ...] sorted by pixel position.
    """
    if len(cands) < 2:
        return [(c["px"], c["val"]) for c in cands]

    if axis == "y":
        # Y-axis labels should be x-aligned: cluster by cx
        align_key = "cx"
    else:
        # X-axis labels should be y-aligned: cluster by cy
        align_key = "cy"

    # Find the most common alignment position (mode)
    positions = [c[align_key] for c in cands]
    # Simple clustering: find the alignment value with most candidates within ±30px
    best_pos = None
    best_count = 0
    for p in positions:
        count = sum(1 for q in positions if abs(q - p) < 30)
        if count > best_count:
            best_count = count
            best_pos = p

    if best_pos is None:
        return []

    # Keep only candidates aligned with the best position
    aligned = [c for c in cands if abs(c[align_key] - best_pos) < 30]

    # Sort by pixel position
    ticks = [(c["px"], c["val"]) for c in aligned]
    ticks.sort(key=lambda t: t[0])

    # Dedup: remove ticks too close in pixel space
    return _dedup_ticks(ticks, min_gap=15)


def _dedup_ticks(ticks: list[tuple], min_gap: float = 15) -> list[tuple]:
    """Remove ticks too close in pixel space, keeping higher values."""
    if not ticks:
        return []
    result = [ticks[0]]
    for t in ticks[1:]:
        if abs(t[0] - result[-1][0]) > min_gap:
            result.append(t)
    return result


def _validate_monotonic(ticks: list[tuple], decreasing: bool) -> list[tuple]:
    """Remove ticks that break monotonicity."""
    if len(ticks) < 2:
        return ticks

    if decreasing:
        valid = [ticks[0]]
        for i in range(1, len(ticks)):
            if ticks[i][1] < valid[-1][1]:
                valid.append(ticks[i])
    else:
        valid = [ticks[0]]
        for i in range(1, len(ticks)):
            if ticks[i][1] > valid[-1][1]:
                valid.append(ticks[i])

    return valid


def _validate_uniform_spacing(ticks: list[tuple]) -> list[tuple]:
    """Keep only ticks with roughly uniform value spacing (arithmetic seq)."""
    if len(ticks) < 3:
        return ticks

    values = [t[1] for t in ticks]
    # Check if values form an arithmetic sequence
    diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
    if not diffs:
        return ticks

    median_diff = sorted(diffs)[len(diffs) // 2]
    if median_diff == 0:
        return ticks

    # Keep ticks where the diff to next is within 50% of median
    good = [ticks[0]]
    for i in range(1, len(ticks)):
        diff = ticks[i][1] - good[-1][1]
        if abs(diff - median_diff) < abs(median_diff) * 0.5:
            good.append(ticks[i])
        elif len(good) < 2:
            # If we haven't found 2 good ticks yet, accept with wider tolerance
            good.append(ticks[i])

    return good if len(good) >= 2 else ticks


# ---------------------------------------------------------------------------
# 4. Coordinate transform
# ---------------------------------------------------------------------------

def build_transform(ticks: list[tuple]) -> Optional[tuple]:
    """
    Linear least-squares fit: data_value = scale * pixel + offset.
    Returns (scale, offset) or None.
    """
    if len(ticks) < 2:
        return None
    pixels = np.array([t[0] for t in ticks], dtype=float)
    values = np.array([t[1] for t in ticks], dtype=float)
    A = np.vstack([pixels, np.ones(len(pixels))]).T
    result = np.linalg.lstsq(A, values, rcond=None)
    scale, offset = result[0]
    return (float(scale), float(offset))


def px_to_data(px: float, transform: tuple) -> float:
    """Convert pixel coordinate to data value."""
    return px * transform[0] + transform[1]


# ---------------------------------------------------------------------------
# 5. Curve tracing
# ---------------------------------------------------------------------------

def trace_curve_hough(image_path: str, plot_region: tuple,
                      x_tf: tuple, y_tf: tuple,
                      n_samples: int = 60) -> list[tuple]:
    """
    Trace curve by removing straight lines (axes/grid) via Hough transform.
    Returns [(time, concentration), ...].
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x0, y0, x1, y1 = [int(v) for v in plot_region]
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w - 1, x1)
    y1 = min(h - 1, y1)

    pw = x1 - x0
    ph = y1 - y0
    if pw < 30 or ph < 30:
        return []

    plot_gray = gray[y0:y1, x0:x1]

    # Step 1: Binary threshold — dark pixels
    _, binary = cv2.threshold(plot_gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Detect straight lines (axes, grid) using Hough
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180,
                            threshold=50, minLineLength=40, maxLineGap=5)

    # Step 3: Mask out straight lines (nearly horizontal or vertical)
    line_mask = np.zeros_like(binary)
    if lines is not None:
        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            dx = abs(lx2 - lx1)
            dy = abs(ly2 - ly1)
            if dx > 8 * dy or dy > 8 * dx:  # nearly straight
                cv2.line(line_mask, (lx1, ly1), (lx2, ly2), 255, 3)

    # Step 4: Curve = dark pixels minus straight lines
    curve_mask = binary & ~line_mask

    # Step 5: Morphological cleanup
    kernel = np.ones((2, 2), np.uint8)
    curve_mask = cv2.morphologyEx(curve_mask, cv2.MORPH_CLOSE, kernel)

    if np.sum(curve_mask > 0) < 10:
        return []

    # Step 6: Sample curve at regular x intervals
    step = max(1, pw // n_samples)
    points = []

    for lx in range(0, pw, step):
        col = curve_mask[:, lx]
        curve_ys = np.where(col > 0)[0]
        if len(curve_ys) == 0:
            continue

        # Use median of curve pixels (robust to thick lines)
        median_y = int(np.median(curve_ys))

        abs_x = x0 + lx
        abs_y = y0 + median_y
        data_x = px_to_data(abs_x, x_tf)
        data_y = px_to_data(abs_y, y_tf)

        if data_x >= -0.5 and data_y >= -10:
            points.append((round(max(0, data_x), 2), round(max(0, data_y), 1)))

    return points


# ---------------------------------------------------------------------------
# 6. Post-processing
# ---------------------------------------------------------------------------

def simplify_curve(points: list[tuple], n_output: int = 20) -> list[tuple]:
    """Resample curve to n_output uniform timepoints."""
    if len(points) <= n_output:
        return points
    if not points:
        return []

    times = np.array([p[0] for p in points])
    concs = np.array([p[1] for p in points])

    target_times = np.linspace(times.min(), times.max(), n_output)
    result = []
    seen_t = set()

    for t_target in target_times:
        idx = np.argmin(np.abs(times - t_target))
        t_val = round(float(times[idx]), 1)
        if t_val not in seen_t:
            seen_t.add(t_val)
            result.append((t_val, round(float(concs[idx]), 1)))

    return result


def detect_unit(ocr_regions: list[dict]) -> str:
    """Detect concentration unit from OCR text."""
    all_text = " ".join(r["text"] for r in ocr_regions).lower()
    patterns = [
        (r"nmol/l|nmol/1|\[nmol", "nmol/L"),
        (r"ng/ml|ng/m1|\[ng/ml", "ng/mL"),
        (r"ng/dl|ng/d1|\[ng/dl", "ng/dL"),
        (r"[uμµ]g/ml|mcg/ml", "μg/mL"),
        (r"mg/l|mg/1", "mg/L"),
        (r"pg/ml", "pg/mL"),
        (r"mg/dl", "mg/dL"),
        (r"[uμµ]g/dl", "μg/dL"),
        (r"[uμµ]mol/l", "μmol/L"),
    ]
    for pat, unit in patterns:
        if re.search(pat, all_text):
            return unit
    return "unknown"


# ---------------------------------------------------------------------------
# 7. Main digitization pipeline
# ---------------------------------------------------------------------------

def digitize_figure(image_path: str, caption: str = "") -> dict:
    """Full pipeline: OCR → axes → transform → trace → output."""
    path = Path(image_path)
    if not path.exists():
        return {"status": "error", "error": "file not found", "image_path": str(path)}

    img = cv2.imread(str(path))
    if img is None:
        return {"status": "error", "error": "cannot read image", "image_path": str(path)}

    img_h, img_w = img.shape[:2]

    # Step 1: OCR
    try:
        ocr_regions = ocr_image(str(path))
    except Exception as e:
        return {"status": "error", "error": f"OCR failed: {e}", "image_path": str(path)}

    # Step 2: Detect axes
    axes = detect_axes(ocr_regions, img_w, img_h)
    x_ticks = axes["x_ticks"]
    y_ticks = axes["y_ticks"]
    plot_region = axes["plot_region"]

    if len(x_ticks) < 2 or len(y_ticks) < 2 or not plot_region:
        return {
            "status": "failed",
            "error": "insufficient axis labels",
            "image_path": str(path),
            "x_ticks": len(x_ticks),
            "y_ticks": len(y_ticks),
            "ocr_count": len(ocr_regions),
        }

    # Step 3: Build transforms
    x_tf = build_transform(x_ticks)
    y_tf = build_transform(y_ticks)
    if not x_tf or not y_tf:
        return {"status": "failed", "error": "transform build failed", "image_path": str(path)}

    # Step 4: Trace curve (Hough line removal approach)
    points = trace_curve_hough(str(path), plot_region, x_tf, y_tf)
    if len(points) < 3:
        return {"status": "failed", "error": "curve trace found <3 points", "image_path": str(path)}

    # Step 5: Simplify
    simplified = simplify_curve(points, n_output=25)
    times = [p[0] for p in simplified]
    concs = [p[1] for p in simplified]
    cmax = max(concs)
    tmax = times[concs.index(cmax)]

    # Unit detection
    conc_unit = detect_unit(ocr_regions)

    return {
        "status": "success",
        "image_path": str(path),
        "image_size": [img_w, img_h],
        "concentration_unit": conc_unit,
        "x_range": [x_ticks[0][1], x_ticks[-1][1]],
        "y_range": [y_ticks[-1][1], y_ticks[0][1]],  # min, max (inverted axis)
        "x_ticks_n": len(x_ticks),
        "y_ticks_n": len(y_ticks),
        "timepoints_h": times,
        "concentrations": concs,
        "cmax": round(cmax, 1),
        "tmax": round(tmax, 2),
        "n_points": len(simplified),
    }


# ---------------------------------------------------------------------------
# 8. Batch processing
# ---------------------------------------------------------------------------

def batch_digitize(candidates_json: str, output_dir: str = "data/digitized/auto/") -> dict:
    """Digitize all C-t candidates from a JSON list."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(candidates_json) as f:
        candidates = json.load(f)

    results = []
    success = 0
    failed = 0

    for i, cand in enumerate(candidates):
        img_path = cand.get("image_path", "")
        caption = cand.get("caption_raw", "")
        result = digitize_figure(img_path, caption)
        result["source"] = {
            "pdf": cand.get("source_pdf", ""),
            "page": cand.get("page", 0),
            "caption": caption[:200],
        }
        results.append(result)

        if result["status"] == "success":
            success += 1
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(candidates)}] ok={success} fail={failed}", flush=True)

    # Save
    with open(outdir / "auto_digitized.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "total": len(candidates),
        "success": success,
        "failed": failed,
        "rate": round(success / max(1, len(candidates)) * 100, 1),
    }
    with open(outdir / "digitization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {success}/{len(candidates)} ({summary['rate']}%)")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Image path or candidates JSON")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--outdir", default="data/digitized/auto/")
    args = parser.parse_args()

    if args.batch:
        batch_digitize(args.path, args.outdir)
    else:
        result = digitize_figure(args.path)
        print(json.dumps(result, indent=2, default=str))
