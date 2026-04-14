"""Download EMA SmPC PDFs and extract Cmax from Section 5.2.

Strategy:
  1. Pick oral small molecules from EMA catalog not in v12/holdout
  2. Download SmPC PDFs from EMA website
  3. Extract Section 5.2 text (Pharmacokinetic properties)
  4. Parse Cmax values with regex
  5. Look up SMILES from PubChem
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import requests

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

ROOT = Path("/home/jam/PLM")
OUT_DIR = ROOT / "data/raw/ema_smpc"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = ROOT / "data/curated/ema_smpc_extracted.json"


def load_existing_drugs():
    """Load drug names/IKs from v12 + holdout to check novelty."""
    v12 = json.load(open(ROOT / "data/curated/plm_dataset_v12_chembl.json"))
    ho = json.load(open(ROOT / "data/validation/holdout_definition.json"))

    # Collect all InChIKey14s
    iks = set()
    for r in v12:
        ik = (r.get("ik") or "")[:14]
        if ik:
            iks.add(ik)
    for ik in ho.get("holdout_inchikeys", []):
        iks.add((ik or "")[:14])

    # Also collect drug names (lowercase) for fuzzy matching
    names = set()
    for d in ho.get("holdout_drugs", []):
        name = (d.get("name") or "").lower().strip()
        if name:
            names.add(name)

    return iks, names


def get_ema_candidates():
    """Get oral small molecule candidates from EMA catalog."""
    ema = json.load(open(ROOT / "data/raw/ema_medicines.json"))
    data = ema["data"]

    candidates = []
    for x in data:
        if x.get("category") != "Human":
            continue
        if x.get("biosimilar") == "Yes" or x.get("generic") == "Yes":
            continue

        subst = x.get("active_substance", "")
        name = x.get("name_of_medicine", "")
        url = x.get("medicine_url", "")

        # Skip obvious biologics
        bio_words = ["mab", "nib$", "cept$", "umab", "zumab",
                     "messenger rna", "virus", "vaccine", "insulin",
                     "interferon", "immunoglobulin", "factor viii",
                     "factor ix", "blood", "antithrombin",
                     "pegylated protein", "antibody", "alfa", "beta"]
        subst_lower = subst.lower()
        if any(w in subst_lower for w in bio_words if not w.endswith("$")):
            continue
        # Check suffix patterns
        if re.search(r"(mab|cept|alfa|beta)\b", subst_lower):
            continue

        # Must have semicolon-free substance (single active, not combo)
        if ";" in subst:
            continue

        if url:
            candidates.append({
                "name": name,
                "substance": subst,
                "url": url,
            })

    return candidates


def download_smpc(medicine_url, name):
    """Try to download SmPC PDF from EMA website."""
    # EMA SmPC URL pattern
    slug = medicine_url.rstrip("/").split("/")[-1]
    smpc_url = f"https://www.ema.europa.eu/en/documents/product-information/{slug}-epar-product-information_en.pdf"

    out_path = OUT_DIR / f"{slug}_smpc.pdf"
    if out_path.exists():
        return out_path

    try:
        resp = requests.get(smpc_url, timeout=30,
                           headers={"User-Agent": "Mozilla/5.0 PLM-Research"})
        if resp.status_code == 200 and len(resp.content) > 10000:
            out_path.write_bytes(resp.content)
            return out_path
        else:
            return None
    except Exception as e:
        print(f"    Download failed: {e}", flush=True)
        return None


def extract_section_52(pdf_path):
    """Extract Section 5.2 Pharmacokinetic properties text from SmPC PDF."""
    if fitz is None:
        return None

    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None

    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Find Section 5.2
    patterns = [
        r"5\.2\s+Pharmacokinetic properties(.*?)(?:5\.3\s+|6\.\s+)",
        r"5\.2\s+Pharmacokinetic\s+properties(.*?)(?:5\.3|6\.)",
        r"Pharmacokinetic properties\s*\n(.*?)(?:Preclinical safety|5\.3)",
    ]

    for pat in patterns:
        m = re.search(pat, full_text, re.DOTALL | re.IGNORECASE)
        if m:
            section = m.group(1).strip()
            # Join mid-sentence line breaks from PDF extraction
            section = re.sub(r'\n(?=[a-z])', ' ', section)
            section = re.sub(r'\n\s+', ' ', section)
            section = re.sub(r'\s{2,}', ' ', section)
            return section

    return None


def parse_cmax(text, substance_name):
    """Parse Cmax value from Section 5.2 text."""
    if not text:
        return []

    results = []

    # Broader Cmax patterns covering SmPC formats
    cmax_patterns = [
        # "Cmax was 52.86 ng/mL" or "Cmax of 340 ng/mL"
        r"C\s*max\s*(?:of|was|is|=|:)\s*(?:approximately\s+)?(\d[\d,. ]*)\s*(ng/m[Ll]|µg/m[Ll]|mg/[Ll]|ug/m[Ll]|mcg/m[Ll]|nmol/[Ll]|μg/m[Ll])",
        # "Cmax and AUC values were 6 µg/mL (28%) and 100 µg.h/mL"
        r"C\s*max\s*(?:and\s+AUC)?\s*(?:values?\s*)?(?:were|was|of|=|:)\s*(\d[\d,. ]*)\s*(ng/m[Ll]|µg/m[Ll]|mg/[Ll]|µg/L|nmol/[Ll]|μg/m[Ll])",
        # "mean Cmax was 52 ng/mL" or "geometric mean Cmax was 52 ng/mL"
        r"(?:mean|geometric\s+mean|median)\s*C\s*max\s*(?:was|of|=|:)\s*(\d[\d,. ]*)\s*(ng/m[Ll]|µg/m[Ll]|mg/[Ll]|nmol/[Ll]|μg/m[Ll])",
        # "peak plasma concentrations... of 259 nmol/L" or "maximum concentration of 340 ng/mL"
        r"(?:peak|maximum)\s+(?:plasma\s+)?concentration[s]?\s*(?:\([^)]*\)\s*)?(?:of|was|is|=|reached|achieved)?\s*(?:approximately\s+)?(\d[\d,. ]*)\s*(ng/m[Ll]|µg/m[Ll]|nmol/[Ll]|μg/m[Ll])",
        # "Cmax 259 nmol/L" (no verb)
        r"C\s*max\s*[\s,]*(\d[\d,. ]*)\s*(ng/m[Ll]|µg/m[Ll]|nmol/[Ll]|μg/m[Ll])",
        # "XXX ng/mL (Cmax)" or "XXX ng/mL for Cmax"
        r"(\d[\d,. ]*)\s*(ng/m[Ll]|µg/m[Ll]|nmol/[Ll]|μg/m[Ll])\s*(?:\([^)]*\)\s*)?(?:for\s+)?C\s*max",
        # Broad: "Cmax ... were/was X unit" (allows text between)
        r"C\s*max.*?(?:were|was)\s+(\d[\d,.]*)\s*(µg/m[Ll]|ng/m[Ll]|mg/[Ll]|μg/m[Ll])",
    ]

    # Dose patterns (broader)
    dose_patterns = [
        r"(\d+[\d,.]*)\s*mg\s+(?:oral|single|dose|tablet|capsule|once|twice|daily|QD|BID)",
        r"(?:single|oral)\s+(?:dose\s+(?:of\s+)?)?(\d+[\d,.]*)\s*mg",
        r"(\d+)\s*mg\s+(?:film-coated\s+)?(?:tablet|capsule)",
        r"dose[s]?\s+(?:of\s+)?(\d+[\d,.]*)\s*mg",
        r"(\d+)\s*mg\s+(?:was|were|and|to)\b",
        r"(?:at|of)\s+(\d+)\s*mg\b",
    ]

    for pat in cmax_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val_str = m.group(1).replace(",", "").replace(" ", "").strip()
            try:
                cmax_val = float(val_str)
            except ValueError:
                continue
            unit = m.group(2).lower().replace("μ", "µ")

            # Convert to ng/mL
            if "nmol" in unit:
                continue  # Skip nmol — need MW for conversion
            elif "µg" in unit or "ug" in unit or "mcg" in unit:
                cmax_ngml = cmax_val * 1000
            elif "mg" in unit:
                cmax_ngml = cmax_val * 1e6
            else:  # ng/mL
                cmax_ngml = cmax_val

            if cmax_ngml <= 0 or cmax_ngml > 1e8:
                continue

            # Try to find dose near this match
            context_start = max(0, m.start() - 800)
            context = text[context_start:m.end() + 300]

            dose = None
            for dpat in dose_patterns:
                dm = re.search(dpat, context, re.IGNORECASE)
                if dm:
                    d = float(dm.group(1).replace(",", ""))
                    if 0.1 <= d <= 2000:  # reasonable oral dose range
                        dose = d
                        break

            if cmax_ngml > 0 and dose and dose > 0:
                results.append({
                    "drug_name": substance_name.lower(),
                    "dose_mg": dose,
                    "cmax_ngml": round(cmax_ngml, 2),
                    "unit_original": m.group(2),
                    "source": "EMA_SmPC",
                })

    # Deduplicate within this drug
    seen = set()
    unique = []
    for r in results:
        key = (r["dose_mg"], r["cmax_ngml"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


def get_smiles_pubchem(drug_name):
    """Look up SMILES from PubChem by drug name. Try multiple name variants."""
    # Clean up name: remove salt forms
    clean_names = [drug_name]
    for suffix in [" hydrochloride", " mesilate", " mesylate", " monohydrate",
                   " dihydrate", " acetate", " phosphate", " sodium",
                   " potassium", " calcium", " fumarate", " maleate",
                   " tartrate", " sulfate", " hydrobromide"]:
        if drug_name.lower().endswith(suffix):
            clean_names.insert(0, drug_name[:len(drug_name)-len(suffix)])

    for name in clean_names:
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            props = data["PropertyTable"]["Properties"][0]
            smiles = (props.get("IsomericSMILES") or props.get("CanonicalSMILES")
                      or props.get("SMILES") or props.get("ConnectivitySMILES"))
            ik = props.get("InChIKey", "")[:14]
            if smiles:
                return smiles, ik
        except Exception:
            continue
        time.sleep(0.3)
    return None, None


def main():
    print("EMA SmPC Cmax Extraction Pipeline", flush=True)
    print("=" * 50, flush=True)

    existing_iks, existing_names = load_existing_drugs()
    print(f"Existing: {len(existing_iks)} IK14s, {len(existing_names)} holdout names", flush=True)

    candidates = get_ema_candidates()
    print(f"EMA candidates: {len(candidates)}", flush=True)

    extracted = []
    attempted = 0
    MAX_ATTEMPTS = 200  # Process in batches

    for cand in candidates[:MAX_ATTEMPTS]:
        name = cand["name"]
        subst = cand["substance"]

        # Quick check: skip if substance name matches known holdout
        if subst.lower() in existing_names:
            continue

        attempted += 1
        print(f"\n[{attempted}] {name} ({subst})", flush=True)

        # Download SmPC
        pdf_path = download_smpc(cand["url"], name)
        if not pdf_path:
            print("    No SmPC PDF", flush=True)
            time.sleep(0.5)
            continue

        print(f"    PDF: {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)", flush=True)

        # Extract Section 5.2
        section = extract_section_52(pdf_path)
        if not section:
            print("    No Section 5.2 found", flush=True)
            continue

        print(f"    Section 5.2: {len(section)} chars", flush=True)

        # Parse Cmax
        results = parse_cmax(section, subst)
        if not results:
            print("    No Cmax parsed", flush=True)
            continue

        # Look up SMILES
        smiles, ik14 = get_smiles_pubchem(subst)
        if not smiles:
            print(f"    No SMILES for {subst}", flush=True)
            continue

        # Check novelty
        if ik14 in existing_iks:
            print(f"    SKIP: {subst} already in v12/holdout (IK={ik14})", flush=True)
            continue

        for r in results:
            r["smiles"] = smiles
            r["ik"] = ik14
            r["ema_name"] = name
            extracted.append(r)
            print(f"    EXTRACTED: {r['drug_name']} {r['dose_mg']}mg "
                  f"Cmax={r['cmax_ngml']} ng/mL", flush=True)

        time.sleep(1)  # Rate limit

    print(f"\n{'='*50}", flush=True)
    print(f"Attempted: {attempted}", flush=True)
    print(f"Extracted: {len(extracted)} entries", flush=True)

    # Deduplicate
    seen = set()
    unique = []
    for r in extracted:
        key = (r["ik"], r["dose_mg"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"Unique (by IK+dose): {len(unique)}", flush=True)

    # Save
    RESULTS_FILE.write_text(json.dumps(unique, indent=2))
    print(f"Wrote {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
