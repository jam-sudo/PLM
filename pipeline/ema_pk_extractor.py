"""Extract PK parameters (Cmax, dose) from EMA EPAR PDFs.

Uses fitz (PyMuPDF) text extraction + regex to find PK tables.
Outputs structured JSON for each drug found.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from dataclasses import dataclass, asdict

import fitz

ROOT = Path("/home/jam/PLM")

# Regex patterns for PK extraction
CMAX_TABLE_RE = re.compile(
    r"C\s*max\s*\(?(?:ng[/.]mL|µg[/.]mL|mg[/.]L|ng\.mL\S*|ug\.mL\S*)?\)?"
    r"\s*[:\s]*(\d+[\d,.]*)",
    re.IGNORECASE,
)

# More targeted: find "Cmax (ng/mL)" followed by a number in a table row
CMAX_VALUE_RE = re.compile(
    r"C\s*max\s*\(?\s*(?:ng[/.]m[Ll]|ng\.mL\S*)\s*\)?\s*[a-z]?\s*"
    r"(\d+[\d,.]*)\s*(?:\([\d,.]+\))?",
    re.IGNORECASE,
)

# Dose patterns
DOSE_MG_RE = re.compile(
    r"(?:single\s+)?(?:oral\s+)?(?:dose\s+(?:of\s+)?)?(\d+(?:\.\d+)?)\s*[-–]?\s*mg\b(?!\s*/\s*kg)",
    re.IGNORECASE,
)

DOSE_FROM_TABLE_HEADER_RE = re.compile(
    r"(?:single|oral)?\s*(\d+)\s*[-–]?\s*mg\s*(?:tablet|capsule|dose|apremilast|oral)?",
    re.IGNORECASE,
)


@dataclass
class PKEntry:
    drug_name: str
    dose_mg: float
    cmax_ngml: float
    condition: str  # fasted/fed/etc
    source: str  # PDF filename + page
    n_subjects: int | None = None
    tmax_h: float | None = None
    half_life_h: float | None = None
    auc_nghr_ml: float | None = None


def extract_pk_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Find pages containing PK data."""
    doc = fitz.open(str(pdf_path))
    pk_pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        text_lower = text.lower()
        # Look for pages with PK tables
        has_cmax = "cmax" in text_lower or "c max" in text_lower or "c_max" in text_lower
        has_table_indicator = any(kw in text_lower for kw in [
            "pharmacokinetic parameter", "pk parameter", "summary of plasma",
            "geometric mean", "arithmetic mean", "mean (sd)", "mean (%cv)",
            "table", "auc", "tmax", "t max",
        ])
        if has_cmax and has_table_indicator:
            pk_pages.append((i + 1, text))
    doc.close()
    return pk_pages


def parse_pk_from_text(text: str, drug_name: str, pdf_name: str, page_num: int) -> list[PKEntry]:
    """Try to extract PK entries from a page of text."""
    entries = []
    lines = text.split("\n")

    # Strategy 1: Find lines with Cmax and extract nearby values
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not re.search(r"C\s*max", line_stripped, re.IGNORECASE):
            continue
        if "ng" not in line_stripped.lower() and "µg" not in line_stripped.lower():
            continue

        # Extract numeric values from this line
        numbers = re.findall(r"(\d+(?:\.\d+)?)", line_stripped)
        if not numbers:
            continue

        # Filter: Cmax should be a reasonable value (1-100000 ng/mL)
        cmax_candidates = []
        for n in numbers:
            val = float(n)
            if 1 < val < 200000:
                cmax_candidates.append(val)

        if not cmax_candidates:
            continue

        # Look for dose in surrounding context (10 lines before and after)
        context_start = max(0, i - 15)
        context_end = min(len(lines), i + 5)
        context = "\n".join(lines[context_start:context_end])

        dose_match = DOSE_MG_RE.search(context)
        dose = float(dose_match.group(1)) if dose_match else None

        # Skip if dose looks like mg/kg
        if dose and dose > 0:
            context_around_dose = context[max(0, dose_match.start()-5):dose_match.end()+10] if dose_match else ""
            if "mg/kg" in context_around_dose or "mg kg" in context_around_dose.lower():
                continue

        # Determine condition (fasted/fed)
        condition = "unknown"
        context_lower = context.lower()
        if "fasted" in context_lower or "fasting" in context_lower:
            condition = "fasted"
        elif "fed" in context_lower:
            condition = "fed"

        # Look for N subjects
        n_match = re.search(r"[Nn]\s*=\s*(\d+)", context)
        n_subjects = int(n_match.group(1)) if n_match else None

        # Use the first reasonable Cmax value
        for cmax in cmax_candidates[:2]:
            if dose and dose > 0:
                log_cd = math.log10(cmax / dose)
                if -2.5 < log_cd < 3.0:  # Sanity check
                    entries.append(PKEntry(
                        drug_name=drug_name,
                        dose_mg=dose,
                        cmax_ngml=cmax,
                        condition=condition,
                        source=f"{pdf_name}:p{page_num}",
                        n_subjects=n_subjects,
                    ))

    return entries


def extract_from_epar(pdf_path: Path, drug_name: str) -> list[PKEntry]:
    """Full extraction pipeline for one EPAR PDF."""
    print(f"\n{'='*60}")
    print(f"Processing: {drug_name} ({pdf_path.name})")
    print(f"{'='*60}")

    pk_pages = extract_pk_pages(pdf_path)
    print(f"Found {len(pk_pages)} pages with PK content")

    all_entries = []
    for page_num, text in pk_pages:
        entries = parse_pk_from_text(text, drug_name, pdf_path.name, page_num)
        if entries:
            for e in entries:
                print(f"  p{page_num}: {e.drug_name} {e.dose_mg}mg → Cmax {e.cmax_ngml} ng/mL ({e.condition})")
            all_entries.extend(entries)

    # Dedup by (dose, cmax, condition)
    seen = set()
    deduped = []
    for e in all_entries:
        key = (round(e.dose_mg, 1), round(e.cmax_ngml, 1), e.condition)
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    print(f"\nTotal entries: {len(all_entries)}, after dedup: {len(deduped)}")
    return deduped


def main():
    """Test on downloaded EPARs."""
    epar_dir = ROOT / "data/raw/ema_epars"
    if not epar_dir.exists():
        print("No EPARs downloaded yet")
        return

    pdfs = list(epar_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} EPAR PDFs")

    # Map PDF names to drug names
    drug_map = {
        "otezla_epar.pdf": "apremilast",
        "noxafil_scientific.pdf": "posaconazole",
    }

    all_entries = []
    for pdf in sorted(pdfs):
        drug_name = drug_map.get(pdf.name, pdf.stem.split("_")[0])
        entries = extract_from_epar(pdf, drug_name)
        all_entries.extend(entries)

    if not all_entries:
        print("\nNo entries extracted!")
        return

    # Save results
    out = {
        "source": "EMA EPAR extraction (pilot)",
        "n_entries": len(all_entries),
        "entries": [asdict(e) for e in all_entries],
    }
    out_path = ROOT / "data/curated/ema_epar_pk_pilot.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(all_entries)} PK entries from {len(pdfs)} EPARs")
    for e in all_entries:
        log_cd = math.log10(e.cmax_ngml / e.dose_mg) if e.dose_mg > 0 else None
        print(f"  {e.drug_name} {e.dose_mg}mg: Cmax={e.cmax_ngml} ng/mL log_cd={log_cd:.2f} [{e.condition}] {e.source}")


if __name__ == "__main__":
    main()
