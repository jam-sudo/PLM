"""
FDA Clinical Pharmacology Review PDF Scraper

Downloads Clinical Pharmacology & Biopharmaceutics Review PDFs
from drugs@FDA for small molecule NDAs.

Usage:
    python -m pipeline.scraper --nda 021457 --outdir data/raw/
    python -m pipeline.scraper --list nda_list.txt --outdir data/raw/
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from typing import Optional


# drugs@FDA API endpoint
DRUGS_FDA_API = "https://api.fda.gov/drug/drugsfda.json"

# Direct download base for review documents
DRUGS_FDA_DOCS = "https://www.accessdata.fda.gov/drugsatfda_docs"


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


def get_review_urls(nda_result: dict) -> list[str]:
    """Extract Clinical Pharmacology review PDF URLs from API result."""
    urls = []
    for submission in nda_result.get("submissions", []):
        for doc in submission.get("review_priority", []):
            # TODO: Parse actual review document links
            pass
    # Fallback: construct likely URL patterns
    # FDA review docs follow patterns like:
    # /nda/{NDA_NUMBER}/{REVIEW_TYPE}/...
    return urls


def download_pdf(url: str, outpath: Path) -> bool:
    """Download a PDF from FDA."""
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        if "pdf" in resp.headers.get("content-type", "").lower():
            with open(outpath, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"  Download failed: {e}")
    return False


def scrape_nda(nda: str, outdir: Path) -> list[Path]:
    """Download all Clinical Pharmacology review PDFs for an NDA."""
    print(f"Processing NDA {nda}...")
    outdir.mkdir(parents=True, exist_ok=True)

    result = search_nda(nda)
    if not result:
        print(f"  Not found in drugs@FDA API")
        return []

    urls = get_review_urls(result)
    downloaded = []
    for i, url in enumerate(urls):
        outpath = outdir / f"{nda}_review_{i}.pdf"
        if outpath.exists():
            print(f"  Already exists: {outpath.name}")
            downloaded.append(outpath)
            continue
        if download_pdf(url, outpath):
            print(f"  Downloaded: {outpath.name}")
            downloaded.append(outpath)
        time.sleep(1)  # Rate limiting

    return downloaded


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nda", type=str, help="Single NDA number")
    parser.add_argument("--list", type=str, help="File with NDA numbers, one per line")
    parser.add_argument("--outdir", type=str, default="data/raw/")
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.nda:
        scrape_nda(args.nda, outdir)
    elif args.list:
        with open(args.list) as f:
            ndas = [line.strip() for line in f if line.strip()]
        for nda in ndas:
            scrape_nda(nda, outdir)
            time.sleep(2)
