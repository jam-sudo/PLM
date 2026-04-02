"""
PDF Figure Extractor

Extracts figure images from FDA Clinical Pharmacology Review PDFs.
Classifies figures as C-t profiles vs other chart types.

Usage:
    python -m pipeline.figure_extractor data/raw/NDA_021457_review_0.pdf --outdir data/figures/
"""

import io
import json
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("pip install PyMuPDF")

from PIL import Image


MIN_IMAGE_WIDTH = 200   # pixels
MIN_IMAGE_HEIGHT = 150  # pixels
MIN_IMAGE_AREA = 50000  # pixels^2


def extract_figures(pdf_path: Path, outdir: Path) -> list[dict]:
    """Extract all figures from a PDF, filtering by size."""
    outdir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    stem = pdf_path.stem

    figures = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            if not base_image:
                continue

            image_bytes = base_image["image"]
            ext = base_image["ext"]
            width = base_image["width"]
            height = base_image["height"]

            # Size filter
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue
            if width * height < MIN_IMAGE_AREA:
                continue

            # Save image
            fname = f"{stem}_p{page_num:03d}_img{img_idx:02d}.{ext}"
            fpath = outdir / fname
            with open(fpath, "wb") as f:
                f.write(image_bytes)

            # Extract surrounding text (potential caption)
            caption = _extract_nearby_text(page, img_info)

            figures.append({
                "source_pdf": str(pdf_path),
                "page": page_num,
                "image_path": str(fpath),
                "width": width,
                "height": height,
                "caption_raw": caption,
                "is_ct_profile": None,  # To be classified
            })

    doc.close()
    return figures


def _extract_nearby_text(page, img_info) -> str:
    """Extract text near an image that might be a caption."""
    # Heuristic: get all text from the page and look for "Figure" patterns
    text = page.get_text()
    # Find lines containing "Figure" or "Concentration" or "plasma"
    lines = text.split("\n")
    caption_lines = []
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in [
            "figure", "concentration", "plasma", "time",
            "mean", "cmax", "auc", "ng/ml", "μg/ml", "mg/l",
            "oral", "intravenous", "dose"
        ]):
            caption_lines.append(line.strip())
    return " ".join(caption_lines[:5])  # First 5 matching lines


def classify_ct_profile(figure: dict) -> bool:
    """
    Classify whether a figure is a concentration-time profile.
    
    Heuristic v1: keyword matching on caption.
    Future: CNN classifier or LLM-based classification.
    """
    caption = (figure.get("caption_raw") or "").lower()

    # Strong positive signals
    positive = [
        "concentration-time",
        "concentration–time",
        "plasma concentration",
        "mean plasma",
        "ng/ml",
        "μg/ml",
        "mg/l",
        "pharmacokinetic profile",
        "pk profile",
        "time (h)",
        "time (hr)",
    ]

    # Strong negative signals
    negative = [
        "bar chart",
        "forest plot",
        "kaplan-meier",
        "survival",
        "waterfall",
        "spider plot",
        "scatter plot",
        "correlation",
        "box plot",
        "histogram",
    ]

    pos_count = sum(1 for p in positive if p in caption)
    neg_count = sum(1 for n in negative if n in caption)

    return pos_count >= 2 and neg_count == 0


def process_pdf(pdf_path: Path, outdir: Path) -> list[dict]:
    """Full pipeline: extract figures, classify, return C-t candidates."""
    figures = extract_figures(pdf_path, outdir)
    for fig in figures:
        fig["is_ct_profile"] = classify_ct_profile(fig)

    ct_count = sum(1 for f in figures if f["is_ct_profile"])
    print(f"  {pdf_path.name}: {len(figures)} figures, {ct_count} C-t candidates")

    return figures


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=str, help="Path to FDA review PDF")
    parser.add_argument("--outdir", type=str, default="data/figures/")
    args = parser.parse_args()

    figures = process_pdf(Path(args.pdf), Path(args.outdir))
    
    # Save metadata
    meta_path = Path(args.outdir) / f"{Path(args.pdf).stem}_figures.json"
    with open(meta_path, "w") as f:
        json.dump(figures, f, indent=2)
    print(f"Metadata saved to {meta_path}")
