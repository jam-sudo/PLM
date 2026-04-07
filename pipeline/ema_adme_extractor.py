"""
Extract ADME features from EMA EPAR PDFs for holdout drugs missing TDC data.

Downloads EPARs, extracts PK section text, parses ADME features via regex.
Fills TDC feature gaps to increase model information capture.

NOT extracting Cmax (that would be holdout contamination).
Extracting: t1/2, F, PPB, CL, Vd, CYP substrates, transporters.

Usage:
    python -m pipeline.ema_adme_extractor
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF


EMA_PDF_PATTERNS = [
    "https://www.ema.europa.eu/en/documents/assessment-report/{slug}-epar-public-assessment-report_en.pdf",
    "https://www.ema.europa.eu/en/documents/scientific-discussion/{slug}-epar-scientific-discussion_en.pdf",
]

DOWNLOAD_DIR = "/tmp/ema_epars"


def download_epar(slug: str) -> str | None:
    """Download EPAR PDF, return path or None."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    out_path = os.path.join(DOWNLOAD_DIR, f"{slug}_epar.pdf")

    if os.path.exists(out_path):
        return out_path

    for pattern in EMA_PDF_PATTERNS:
        url = pattern.format(slug=slug)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PLM-Research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(out_path, "wb") as f:
                    f.write(resp.read())
            return out_path
        except Exception:
            continue

    return None


def extract_pk_section(pdf_path: str) -> str:
    """Extract pharmacokinetics section text from EPAR PDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    pk_text = ""
    in_pk = False
    pk_start = None

    for page_num in range(doc.page_count):
        text = doc[page_num].get_text()
        full_text += text

        if re.search(r"2\.4\.2\s.*[Pp]harmacokinetics|[Pp]harmacokinetics\s*\n", text):
            if not in_pk:
                in_pk = True
                pk_start = page_num

        if in_pk:
            pk_text += text
            if pk_start and page_num > pk_start + 1:
                if re.search(r"2\.4\.3|[Pp]harmacodynamics\s*\n|2\.5\s", text):
                    in_pk = False

    doc.close()

    # Fallback: search entire text for PK keywords
    if len(pk_text) < 500:
        pk_text = full_text

    return pk_text


def parse_adme_features(pk_text: str) -> dict:
    """Extract ADME features from PK section text."""
    features = {}

    # Half-life (hours)
    thalf_matches = re.findall(
        r"(?:half[- ]?life|t½|t1/2|T-half)[:\s]+(?:approximately\s+|~\s*|is\s+)?(\d+(?:\.\d+)?)\s*(?:to\s+(\d+(?:\.\d+)?)\s*)?(?:hours?|h\b|hrs?)",
        pk_text, re.IGNORECASE,
    )
    if thalf_matches:
        vals = []
        for m in thalf_matches:
            v1 = float(m[0])
            if m[1]:
                v2 = float(m[1])
                vals.append((v1 + v2) / 2)
            else:
                vals.append(v1)
        if vals:
            features["half_life_h"] = round(float(max(set(vals), key=vals.count)), 2)

    # Bioavailability (%)
    f_matches = re.findall(
        r"(?:absolute\s+)?bioavailability[:\s]+(?:approximately\s+|~\s*|of\s+|is\s+|was\s+)?(\d+(?:\.\d+)?)\s*%",
        pk_text, re.IGNORECASE,
    )
    if f_matches:
        features["bioavailability_pct"] = float(f_matches[0])
        features["bioavailability_binary"] = 1.0 if float(f_matches[0]) >= 20 else 0.0

    # Protein binding (%)
    ppb_matches = re.findall(
        r"(?:protein|plasma|serum)\s+(?:binding|bound)[:\s]+(?:approximately\s+|~\s*)?(\d+(?:\.\d+)?)\s*%",
        pk_text, re.IGNORECASE,
    )
    if ppb_matches:
        features["ppb_pct"] = float(ppb_matches[0])

    # Clearance (L/h or mL/min)
    cl_lh = re.findall(
        r"(?:clearance|CL/F|CL)[:\s]+(?:approximately\s+|~\s*)?(\d+(?:\.\d+)?)\s*L/h",
        pk_text, re.IGNORECASE,
    )
    cl_mlmin = re.findall(
        r"(?:clearance|CL/F|CL)[:\s]+(?:approximately\s+|~\s*)?(\d+(?:\.\d+)?)\s*mL/min",
        pk_text, re.IGNORECASE,
    )
    if cl_lh:
        features["clearance_Lh"] = float(cl_lh[0])
    elif cl_mlmin:
        features["clearance_Lh"] = round(float(cl_mlmin[0]) * 0.06, 2)

    # Volume of distribution (L)
    vd_matches = re.findall(
        r"(?:volume\s+of\s+distribution|Vd|Vss|Vd/F|Vz/F)[:\s]+(?:approximately\s+|~\s*)?(\d+(?:\.\d+)?)\s*L\b",
        pk_text, re.IGNORECASE,
    )
    if vd_matches:
        features["vd_L"] = float(vd_matches[0])

    vd_kg = re.findall(
        r"(?:volume\s+of\s+distribution|Vd|Vss)[:\s]+(?:approximately\s+|~\s*)?(\d+(?:\.\d+)?)\s*L/kg",
        pk_text, re.IGNORECASE,
    )
    if vd_kg and "vd_L" not in features:
        features["vd_L_kg"] = float(vd_kg[0])

    # CYP substrates
    cyp_matches = re.findall(r"CYP\s*(\d[A-Z]\d+)", pk_text)
    if cyp_matches:
        features["cyp_enzymes"] = list(set(cyp_matches))
        # Binary flags for major CYPs
        cyps = set(cyp_matches)
        features["cyp3a4_substrate"] = 1.0 if "3A4" in cyps or "3A5" in cyps else 0.0
        features["cyp2d6_substrate"] = 1.0 if "2D6" in cyps else 0.0
        features["cyp2c9_substrate"] = 1.0 if "2C9" in cyps else 0.0
        features["cyp2c19_substrate"] = 1.0 if "2C19" in cyps else 0.0
        features["cyp1a2_substrate"] = 1.0 if "1A2" in cyps else 0.0

    # Transporters
    pgp = bool(re.search(r"P-?g(?:lyco)?p(?:rotein)?.*substrate|substrate.*P-?gp", pk_text, re.IGNORECASE))
    bcrp = bool(re.search(r"BCRP.*substrate|substrate.*BCRP", pk_text, re.IGNORECASE))
    oatp = bool(re.search(r"OATP.*substrate|substrate.*OATP", pk_text, re.IGNORECASE))
    features["pgp_substrate"] = 1.0 if pgp else 0.0
    features["bcrp_substrate"] = 1.0 if bcrp else 0.0
    features["oatp_substrate"] = 1.0 if oatp else 0.0

    return features


def main():
    print("=" * 70)
    print("EMA ADME FEATURE EXTRACTION")
    print("=" * 70)
    print("Evaluation integrity check:")
    print("  - Extracting ADME properties (F, CL, Vd, CYP, PPB)")
    print("  - NOT extracting Cmax (holdout target)")
    print("  - These are drug descriptors, not prediction targets")
    print()

    with open("/tmp/ema_holdout_matched.json") as f:
        matched = json.load(f)

    print(f"Holdout drugs with EMA EPARs: {len(matched)}")
    print()

    results = []
    for m in matched:
        slug = m["slug"]
        name = m["ho_name"]
        print(f"  [{name}] ({slug})...", end=" ", flush=True)

        pdf_path = download_epar(slug)
        if pdf_path is None:
            print("DOWNLOAD FAILED")
            results.append({"drug": name, "slug": slug, "status": "download_failed"})
            continue

        pk_text = extract_pk_section(pdf_path)
        if len(pk_text) < 200:
            print("PK SECTION NOT FOUND")
            results.append({"drug": name, "slug": slug, "status": "pk_not_found"})
            continue

        features = parse_adme_features(pk_text)
        n_features = sum(1 for k, v in features.items()
                        if k not in ("cyp_enzymes",) and v is not None)

        print(f"{n_features} features extracted")
        results.append({
            "drug": name,
            "slug": slug,
            "status": "ok",
            "n_features": n_features,
            "features": features,
        })

        time.sleep(0.5)  # polite rate limit

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{'='*70}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total drugs: {len(matched)}")
    print(f"  Downloaded: {len([r for r in results if r['status'] != 'download_failed'])}")
    print(f"  PK section found: {len(ok)}")
    print()
    print(f"  Feature coverage:")
    for feat in ["half_life_h", "bioavailability_pct", "ppb_pct", "clearance_Lh",
                 "vd_L", "cyp3a4_substrate", "pgp_substrate"]:
        n = sum(1 for r in ok if feat in r.get("features", {}))
        print(f"    {feat:<25s}: {n}/{len(ok)}")

    # Save
    out_path = "data/curated/ema_adme_features.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
