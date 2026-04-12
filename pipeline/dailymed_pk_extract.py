"""DailyMed PK extraction pipeline — full version.

1. Takes a list of drug names
2. Queries DailyMed for Section 12.3
3. Parses Cmax with improved regex (handles "C max" subscript spacing)
4. Gets SMILES from PubChem
5. Checks overlap against v11 by IK14
6. Outputs structured JSON for training
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import requests
import xml.etree.ElementTree as ET

ROOT = Path("/home/jam/PLM")

DAILYMED_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PK_LOINC = "43682-4"

# Unit pattern fragment — reused across patterns
_UNIT = r"(ng/mL|ng/ml|µg/mL|ug/mL|mcg/mL|mg/L|ng\.mL\S*|ug\.mL\S*|mcg/ml)"
# Optional CV% in parens between value and unit: "563 (29%) ng/mL"
_CV = r"(?:\s*\(\s*\d+[\d.]*\s*%?\s*\)\s*)"

# Improved Cmax regex: handles "C max", "Cmax", "C_max" with various spacing
# Key fix: allows (CV%) between number and unit
CMAX_PATTERNS = [
    # "C max ) is/was/of/: 1234 (CV%) ng/mL"
    re.compile(
        r"C\s*_?\s*max\s*\)?\s*(?:,?\s*ss)?\s*(?:is|was|of|=|:)\s*(?:approximately\s+|~)?"
        r"(\d+[\d,.]*)" + _CV + r"?\s*" + _UNIT,
        re.IGNORECASE,
    ),
    # "mean/geometric mean C max of/is 1234 (CV%) ng/mL"
    re.compile(
        r"(?:mean|geometric mean|median|average)\s+(?:peak\s+)?(?:plasma\s+)?(?:concentration\s+)?"
        r"\(?C\s*_?\s*max\s*\)?\s*(?:is|was|of|=|:)?\s*(?:approximately\s+|~)?"
        r"(\d+[\d,.]*)" + _CV + r"?\s*" + _UNIT,
        re.IGNORECASE,
    ),
    # "C max 1234 (CV%) ng/mL" — direct juxtaposition
    re.compile(
        r"C\s*_?\s*max\s*\)?\s+(\d+[\d,.]*)" + _CV + r"?\s*" + _UNIT,
        re.IGNORECASE,
    ),
    # "X (CV%) ng/mL and Y (CV%) ng/mL" after AUC — catch second value
    # e.g., "were 1843 (38%) ng•h/mL and 563 (29%) ng/mL"
    re.compile(
        r"and\s+(\d+[\d,.]*)" + _CV + r"?\s*" + _UNIT,
        re.IGNORECASE,
    ),
    # "peak plasma concentration(s) (C max ) is/was 1234 (CV%) ng/mL"
    re.compile(
        r"peak\s+(?:plasma\s+)?concentrations?\s*\(\s*C\s*_?\s*max\s*\)\s*"
        r"(?:is|was|of|=|:)?\s*(?:approximately\s+|~)?(\d+[\d,.]*)"
        + _CV + r"?\s*" + _UNIT,
        re.IGNORECASE,
    ),
    # "C max,ss (ng/mL) ... 747 (45)" — table-like header then value
    re.compile(
        r"C\s*_?\s*max\s*(?:,?\s*ss)?\s*\(\s*" + _UNIT[1:-1] + r"\s*\)\s*"
        r".*?(\d+[\d,.]*)" + _CV + r"?",
        re.IGNORECASE,
    ),
]

# Dose patterns
DOSE_RE = re.compile(
    r"(?:single\s+(?:oral\s+)?dose\s+(?:of\s+)?|oral\s+dose\s+(?:of\s+)?|"
    r"(?:administered|given)\s+(?:a\s+)?(?:single\s+)?|"
    r"(?:following|after)\s+(?:a\s+)?(?:single\s+)?(?:oral\s+)?(?:dose\s+of\s+)?|"
    r"(?:at\s+)?(?:a\s+)?(?:dose\s+of\s+)?|"
    r"(?:recommended\s+)?(?:daily\s+)?(?:dose\s+(?:of\s+)?))"
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*mg\b(?!\s*/\s*(?:kg|m\s*2|m²))",
    re.IGNORECASE,
)

# Also catch "X mg once daily" or "X mg tablet" or "X-mg" or "following X mg"
DOSE_SIMPLE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*mg\s+(?:once|twice|tablet|capsule|oral|daily|dose|administered|BID|QD|of\s+\w+\s+(?:once|twice|daily))",
    re.IGNORECASE,
)

# "following 160 mg twice daily" / "at 100 mg daily"
DOSE_FOLLOWING_RE = re.compile(
    r"(?:following|at|of)\s+(\d+(?:\.\d+)?)\s*[-–]?\s*mg\s*(?:once|twice|BID|QD|daily|orally)?",
    re.IGNORECASE,
)

# Fixed-dose from formulation: "100 mg tablets"
DOSE_FORM_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*mg\s+(?:tablet|capsule|film)",
    re.IGNORECASE,
)


def cmax_to_ngml(value: float, unit: str) -> float:
    """Convert Cmax to ng/mL."""
    u = unit.lower().replace(".", "/").replace(" ", "")
    if "mcg" in u or "ug" in u or "µg" in u:
        return value * 1000
    elif "mg/l" in u:
        return value * 1000
    elif "ng" in u:
        return value
    return value  # Assume ng/mL if unclear


def get_spl(drug_name: str) -> tuple[str | None, str]:
    """Get SPL XML for drug. Returns (xml, setid)."""
    url = f"{DAILYMED_API}/spls.json"
    r = requests.get(url, params={"drug_name": drug_name, "pagesize": 3}, timeout=15)
    if r.status_code != 200:
        return None, ""
    data = r.json().get("data", [])
    if not data:
        return None, ""

    setid = data[0]["setid"]
    xml_r = requests.get(f"{DAILYMED_API}/spls/{setid}.xml", timeout=15)
    if xml_r.status_code != 200:
        return None, setid
    return xml_r.text, setid


def extract_pk_text(xml_text: str) -> str | None:
    """Extract Section 12.3 text."""
    try:
        root = ET.fromstring(xml_text)
        for section in root.iter("{urn:hl7-org:v3}section"):
            code = section.find("{urn:hl7-org:v3}code")
            if code is not None and code.get("code") == PK_LOINC:
                texts = []
                for elem in section.iter():
                    if elem.text:
                        texts.append(elem.text.strip())
                    if elem.tail:
                        texts.append(elem.tail.strip())
                return " ".join(t for t in texts if t)
    except ET.ParseError:
        pass
    return None


def parse_cmax(pk_text: str) -> list[dict]:
    """Extract Cmax values with improved regex."""
    results = []
    seen = set()

    for pi, pattern in enumerate(CMAX_PATTERNS):
        for match in pattern.finditer(pk_text):
            groups = match.groups()

            # For "and X (CV%) ng/mL" pattern, only valid if "C max" is nearby
            if pi == 3:  # "and X ... ng/mL" pattern
                lookback = pk_text[max(0, match.start()-200):match.start()]
                if not re.search(r"C\s*_?\s*max", lookback, re.IGNORECASE):
                    continue
                # Also skip if context is about AUC change, DDI, etc
                if any(kw in lookback.lower() for kw in ["decreased", "increased", "fold", "reduced"]):
                    continue

            # Find cmax_str and unit from captured groups
            unit_candidates = {"ng/mL", "ng/ml", "µg/mL", "ug/mL", "mcg/mL", "mg/L",
                             "mcg/ml", "ng.mL", "ug.mL"}
            cmax_str, unit = None, None
            for g in groups:
                if g is None:
                    continue
                g_clean = g.lower().replace(".", "/").rstrip("-1")
                if any(u.lower() in g_clean for u in unit_candidates) or g_clean.startswith("ng") or g_clean.startswith("mcg") or g_clean.startswith("ug"):
                    unit = g
                elif cmax_str is None:
                    cmax_str = g

            if not cmax_str or not unit:
                continue

            cmax_str = cmax_str.replace(",", "")
            try:
                cmax = float(cmax_str)
            except ValueError:
                continue

            cmax_ngml = cmax_to_ngml(cmax, unit)
            if cmax_ngml <= 0 or cmax_ngml > 1e7:
                continue

            # Find dose in context
            start = max(0, match.start() - 500)
            end = min(len(pk_text), match.end() + 200)
            context = pk_text[start:end]

            dose = None
            for dose_pat in [DOSE_RE, DOSE_SIMPLE_RE, DOSE_FOLLOWING_RE, DOSE_FORM_RE]:
                dm = dose_pat.search(context)
                if dm:
                    dose_val = float(dm.group(1))
                    # Check for mg/kg
                    around = context[max(0, dm.start()-5):dm.end()+15]
                    if "mg/kg" in around or "mg/m" in around or "mg kg" in around.lower():
                        continue
                    if 0.1 <= dose_val <= 5000:
                        dose = dose_val
                        break

            key = (round(cmax_ngml, 1), dose)
            if key in seen:
                continue
            seen.add(key)

            results.append({
                "cmax_ngml": cmax_ngml,
                "dose_mg": dose,
                "unit_raw": unit,
                "context_snippet": pk_text[max(0, match.start()-50):match.end()+50].strip()[:200],
            })

    return results


def get_smiles_from_pubchem(drug_name: str) -> str | None:
    """Look up canonical SMILES from PubChem."""
    try:
        url = f"{PUBCHEM_API}/compound/name/{drug_name}/property/CanonicalSMILES,InChIKey/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            return {
                "smiles": props[0].get("CanonicalSMILES"),
                "inchikey": props[0].get("InChIKey"),
            }
    except Exception:
        pass
    return None


def main():
    # Load v11 IK14s for overlap check
    v11 = json.loads((ROOT / "data/curated/plm_dataset_v11_llm.json").read_text())
    v11_iks = set()
    for r in v11:
        ik = (r.get("ik") or "")[:14]
        if ik:
            v11_iks.add(ik)
    print(f"v11: {len(v11_iks)} unique IK14s")

    # Also load v12 ChEMBL additions
    v12 = json.loads((ROOT / "data/curated/plm_dataset_v12_chembl.json").read_text())
    v12_iks = set()
    for r in v12:
        ik = (r.get("ik") or "")[:14]
        if ik:
            v12_iks.add(ik)

    # Holdout IK14s — must exclude
    ho = json.loads((ROOT / "data/validation/holdout_definition.json").read_text())
    ho_iks = set((k or "")[:14] for k in ho["holdout_inchikeys"])

    # Large list of FDA-approved oral drugs to scout
    # Focus on drugs likely NOT in v11 (newer approvals, oncology, rare disease)
    drugs = [
        # Oncology oral
        "acalabrutinib", "zanubrutinib", "ibrutinib", "pirtobrutinib",
        "fedratinib", "glasdegib", "gilteritinib", "midostaurin",
        "enasidenib", "ivosidenib", "selinexor", "duvelisib",
        "idelalisib", "copanlisib", "alpelisib", "umbralisib",
        "avapritinib", "ripretinib", "tucatinib", "neratinib",
        "capmatinib", "tepotinib", "mobocertinib", "adagrasib",
        "sotorasib", "infigratinib", "erdafitinib", "futibatinib",
        "pemigatinib", "entrectinib", "larotrectinib", "repotrectinib",
        "tivozanib", "cabozantinib", "lenvatinib", "axitinib",
        "pazopanib", "sunitinib", "sorafenib", "vemurafenib",
        "dabrafenib", "encorafenib", "trametinib", "cobimetinib",
        "binimetinib", "sonidegib", "vismodegib", "olaparib",
        "rucaparib", "niraparib", "talazoparib", "ixazomib",
        "panobinostat", "vorinostat", "venetoclax", "idelalisib",
        # Immunology / rheumatology oral
        "apremilast", "baricitinib", "tofacitinib", "upadacitinib",
        "filgotinib", "deucravacitinib", "abrocitinib",
        # Hematology oral
        "eltrombopag", "avatrombopag", "luspatercept",
        "avacopan", "voclosporin",
        # Cardiology/metabolic newer oral
        "mavacamten", "aficamten", "vericiguat", "sacubitril",
        "bempedoic acid", "inclisiran",
        # Neurology oral
        "elagolix", "relugolix", "orilissa", "rinvoq",
        "ozanimod", "siponimod", "ponesimod", "fingolimod",
        # Anti-infective oral
        "tedizolid", "delafloxacin", "lefamulin", "omadacycline",
        "pretomanid", "bedaquiline",
        # Rare disease oral
        "miglustat", "eliglustat", "tafamidis",
        "givosiran", "lumasiran",
    ]

    # Remove duplicates
    drugs = list(dict.fromkeys(drugs))
    print(f"Scouting {len(drugs)} drugs")

    all_entries = []
    failed = []
    no_pk = []
    no_cmax = []

    for i, drug in enumerate(drugs):
        if i > 0 and i % 10 == 0:
            print(f"  Progress: {i}/{len(drugs)} ({len(all_entries)} entries so far)")

        xml, setid = get_spl(drug)
        if not xml:
            failed.append(drug)
            time.sleep(0.3)
            continue

        pk_text = extract_pk_text(xml)
        if not pk_text:
            no_pk.append(drug)
            time.sleep(0.3)
            continue

        entries = parse_cmax(pk_text)
        if not entries:
            no_cmax.append(drug)
            time.sleep(0.3)
            continue

        # Get SMILES
        chem = get_smiles_from_pubchem(drug)
        smiles = chem["smiles"] if chem else None
        ik_full = chem["inchikey"] if chem else None
        ik14 = ik_full[:14] if ik_full else None

        in_v11 = ik14 in v11_iks if ik14 else False
        in_v12 = ik14 in v12_iks if ik14 else False
        in_ho = ik14 in ho_iks if ik14 else False

        for e in entries:
            if e["dose_mg"] is None:
                continue
            log_cd = math.log10(e["cmax_ngml"] / e["dose_mg"]) if e["dose_mg"] > 0 else None
            if log_cd is not None and (log_cd < -2.5 or log_cd > 3.5):
                continue  # Sanity check

            entry = {
                "drug_name": drug,
                "smiles": smiles,
                "ik14": ik14,
                "dose_mg": e["dose_mg"],
                "cmax_ngml": e["cmax_ngml"],
                "log_cd": log_cd,
                "in_v11": in_v11,
                "in_v12": in_v12,
                "in_holdout": in_ho,
                "src": f"DailyMed:{setid[:8]}",
            }
            all_entries.append(entry)
            status = "HO" if in_ho else ("v11" if in_v11 else ("v12" if in_v12 else "NEW"))
            print(f"  {drug} {e['dose_mg']}mg → {e['cmax_ngml']} ng/mL (log_cd={log_cd:.2f}) [{status}]")

        time.sleep(0.5)  # Rate limit

    # Summary
    print(f"\n{'='*60}")
    print(f"DailyMed PK Extraction Summary")
    print(f"{'='*60}")
    print(f"Drugs queried: {len(drugs)}")
    print(f"Failed (no DailyMed): {len(failed)}")
    print(f"No PK section: {len(no_pk)}")
    print(f"No Cmax parsed: {len(no_cmax)}")
    print(f"Total entries with Cmax+dose: {len(all_entries)}")

    novel = [e for e in all_entries if not e["in_v11"] and not e["in_v12"] and not e["in_holdout"] and e["smiles"]]
    overlap_v11 = [e for e in all_entries if e["in_v11"]]
    holdout = [e for e in all_entries if e["in_holdout"]]

    print(f"\nNovel (not in v11/v12/holdout, with SMILES): {len(novel)}")
    print(f"Already in v11: {len(overlap_v11)}")
    print(f"In holdout (excluded): {len(holdout)}")

    for e in novel:
        print(f"  {e['drug_name']} {e['dose_mg']}mg → {e['cmax_ngml']} ng/mL (log_cd={e['log_cd']:.2f})")

    # Failed drugs for debugging
    if no_cmax:
        print(f"\nDrugs with PK section but no Cmax parsed ({len(no_cmax)}):")
        for d in no_cmax[:20]:
            print(f"  {d}")

    # Save
    out = {
        "extraction_date": "2026-04-12",
        "drugs_queried": len(drugs),
        "entries": all_entries,
        "novel_count": len(novel),
        "overlap_v11": len(overlap_v11),
        "holdout_excluded": len(holdout),
        "failed_drugs": failed,
        "no_pk_section": no_pk,
        "no_cmax_parsed": no_cmax,
    }
    out_path = ROOT / "data/curated/dailymed_pk_extracted.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
