"""
FDA Clinical Pharmacology Review PDF Scraper

Downloads Clinical Pharmacology & Biopharmaceutics Review PDFs
from drugs@FDA for small molecule NDAs.

Usage:
    python -m pipeline.scraper --nda 021457 --outdir data/raw/
    python -m pipeline.scraper --list nda_list.txt --outdir data/raw/
"""

import json
import time
import requests
from pathlib import Path
from typing import Optional


# drugs@FDA API endpoint
DRUGS_FDA_API = "https://api.fda.gov/drug/drugsfda.json"

# Direct download base for review documents
DRUGS_FDA_DOCS = "https://www.accessdata.fda.gov/drugsatfda_docs"

# FDA blocks bare requests; use browser-like User-Agent
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Known URL patterns for ClinPharmR PDFs (year, NDA)
CLINPHARMR_PATTERNS = [
    "{docs}/nda/{year}/{nda}Orig1s000ClinPharmR.pdf",
    "{docs}/nda/{year}/{nda}Orig1s000ClinPharmR_0.pdf",
]

# Rate limiting
REQUEST_DELAY = 2.0  # seconds between FDA requests


def search_nda(application_number: str) -> Optional[dict]:
    """Search drugs@FDA API for an NDA/BLA."""
    params = {
        "search": f'openfda.application_number:"{application_number}"',
        "limit": 1,
    }
    try:
        resp = requests.get(DRUGS_FDA_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("results"):
            return data["results"][0]
    except Exception as e:
        print(f"  API error for {application_number}: {e}")
    return None


def build_pdf_url(nda: str, year: str) -> str:
    """Construct likely ClinPharmR PDF URL."""
    return f"{DRUGS_FDA_DOCS}/nda/{year}/{nda}Orig1s000ClinPharmR.pdf"


def download_pdf(url: str, outpath: Path) -> bool:
    """Download a PDF from FDA with proper headers."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        if "pdf" in content_type or url.endswith(".pdf"):
            with open(outpath, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            size_mb = outpath.stat().st_size / (1024 * 1024)
            print(f"  Downloaded: {outpath.name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  Not a PDF: content-type={content_type}")
    except requests.HTTPError as e:
        print(f"  HTTP error: {e.response.status_code}")
    except Exception as e:
        print(f"  Download failed: {e}")
    return False


def scrape_nda(nda: str, year: str, outdir: Path) -> Optional[Path]:
    """Download ClinPharmR PDF for an NDA given its approval year."""
    print(f"Processing NDA {nda} (year {year})...")
    outdir.mkdir(parents=True, exist_ok=True)

    outpath = outdir / f"NDA{nda}_ClinPharmR.pdf"
    if outpath.exists():
        print(f"  Already exists: {outpath.name}")
        return outpath

    url = build_pdf_url(nda, year)
    if download_pdf(url, outpath):
        return outpath

    # Try alternate URL pattern
    alt_url = f"{DRUGS_FDA_DOCS}/nda/{year}/{nda}Orig1s000ClinPharmR_0.pdf"
    if download_pdf(alt_url, outpath):
        return outpath

    print(f"  Failed to download NDA {nda}")
    return None


def scrape_batch(nda_list: list[dict], outdir: Path) -> dict:
    """Download ClinPharmR PDFs for a batch of NDAs.

    Args:
        nda_list: List of dicts with 'nda' and 'year' keys.
        outdir: Output directory for PDFs.

    Returns:
        Manifest dict with download status for each NDA.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {"download_date": time.strftime("%Y-%m-%d"), "results": []}

    for entry in nda_list:
        nda = entry["nda"]
        year = entry["year"]
        result = scrape_nda(nda, year, outdir)
        manifest["results"].append({
            "nda": nda,
            "year": year,
            "success": result is not None,
            "path": str(result) if result else None,
        })
        time.sleep(REQUEST_DELAY)

    # Save manifest
    manifest_path = outdir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")

    n_ok = sum(1 for r in manifest["results"] if r["success"])
    print(f"Downloaded {n_ok}/{len(nda_list)} PDFs")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nda", type=str, help="Single NDA number")
    parser.add_argument("--year", type=str, help="Approval year for --nda")
    parser.add_argument("--list", type=str, help="JSON file with [{nda, year}, ...]")
    parser.add_argument("--outdir", type=str, default="data/raw/")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.nda and args.year:
        scrape_nda(args.nda, args.year, outdir)
    elif args.list:
        with open(args.list) as f:
            nda_list = json.load(f)
        scrape_batch(nda_list, outdir)
    else:
        parser.print_help()
