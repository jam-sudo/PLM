"""
PDF Figure Extractor

Extracts figure images from FDA Clinical Pharmacology Review PDFs.
Classifies figures as C-t profiles vs other chart types.

Usage:
    python -m pipeline.figure_extractor data/raw/NDA_021457_ClinPharmR.pdf --outdir data/figures/
    python -m pipeline.figure_extractor data/raw/ --outdir data/figures/ --batch
"""

import json
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("pip install PyMuPDF")


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

            # Detect scanned pages (single large image covering most of page)
            page_rect = page.rect
            is_scan = (
                len(image_list) <= 2
                and width > page_rect.width * 0.8
                and height > page_rect.height * 0.8
            )

            # Save image
            fname = f"{stem}_p{page_num:03d}_img{img_idx:02d}.{ext}"
            fpath = outdir / fname
            with open(fpath, "wb") as f:
                f.write(image_bytes)

            # Extract surrounding text (potential caption)
            caption = _extract_nearby_text(page)

            figures.append({
                "source_pdf": str(pdf_path),
                "page": page_num,
                "image_path": str(fpath),
                "width": width,
                "height": height,
                "caption_raw": caption,
                "is_scan": is_scan,
                "is_ct_profile": None,  # To be classified
            })

    doc.close()
    return figures


def _extract_nearby_text(page) -> str:
    """Extract text from page that might be caption or axis labels."""
    text = page.get_text()
    lines = text.split("\n")
    caption_lines = []
    for line in lines:
        lower = line.lower()
        if any(kw in lower for kw in [
            "figure", "concentration", "plasma", "time",
            "mean", "cmax", "auc", "ng/ml", "μg/ml", "mg/l",
            "oral", "intravenous", "dose", "nmol/l", "ng/dl",
            "pharmacokinetic", "profile", "pk", "subject",
            "hours", "conc", "log", "linear",
        ]):
            caption_lines.append(line.strip())
    return " ".join(caption_lines[:5])


def classify_ct_profile(figure: dict) -> bool:
    """
    Classify whether a figure is a concentration-time profile.

    Heuristic v2: improved keyword matching with stricter positive criteria.
    Precision ~59% on validation set; use LLM visual review for confirmation.
    """
    if figure.get("is_scan"):
        return False

    caption = (figure.get("caption_raw") or "").lower()

    # Strong positive signals
    positive = [
        "concentration-time", "concentration–time",
        "plasma concentration", "mean plasma",
        "ng/ml", "μg/ml", "mg/l", "nmol/l", "ng/dl",
        "pharmacokinetic profile", "pk profile",
        "time (h)", "time (hr)", "time (hours)",
        "conc-time", "conc–time",
        "plasma conc", "serum conc",
    ]

    # Strong negative signals
    negative = [
        "bar chart", "forest plot", "kaplan-meier", "survival",
        "waterfall", "spider plot", "scatter plot", "correlation",
        "box plot", "histogram", "metabolic pathway", "metabolism",
        "chemical structure", "molecular", "goodness-of-fit",
        "residual", "qq plot", "q-q plot", "diagnostic",
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
    scan_count = sum(1 for f in figures if f.get("is_scan"))
    print(f"  {pdf_path.name}: {len(figures)} figures ({scan_count} scans), {ct_count} C-t candidates")

    return figures


def process_batch(pdf_dir: Path, outdir: Path) -> dict:
    """Process all PDFs in a directory."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    all_figures = []

    for pdf_path in pdf_files:
        nda_outdir = outdir / pdf_path.stem
        figures = process_pdf(pdf_path, nda_outdir)
        all_figures.extend(figures)

    # Save metadata
    meta_path = outdir / "extraction_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_figures, f, indent=2)

    # Save C-t candidates
    ct_candidates = [f for f in all_figures if f["is_ct_profile"]]
    ct_path = outdir / "ct_candidates_heuristic.json"
    with open(ct_path, "w") as f:
        json.dump(ct_candidates, f, indent=2)

    print(f"\nTotal: {len(all_figures)} figures, {len(ct_candidates)} C-t candidates")
    print(f"Metadata: {meta_path}")
    return {"total": len(all_figures), "ct_candidates": len(ct_candidates)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to PDF or directory of PDFs")
    parser.add_argument("--outdir", type=str, default="data/figures/")
    parser.add_argument("--batch", action="store_true", help="Process all PDFs in directory")
    args = parser.parse_args()

    path = Path(args.path)
    outdir = Path(args.outdir)

    if args.batch or path.is_dir():
        process_batch(path, outdir)
    else:
        figures = process_pdf(path, outdir)
        meta_path = outdir / f"{path.stem}_figures.json"
        with open(meta_path, "w") as f:
            json.dump(figures, f, indent=2)
        print(f"Metadata saved to {meta_path}")
