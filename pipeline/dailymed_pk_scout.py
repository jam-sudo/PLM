"""Scout DailyMed for PK data (Section 12.3) not in v11.

DailyMed provides FDA drug labels in structured XML (SPL format).
Section 12.3 "Pharmacokinetics" often contains Cmax, Tmax, AUC values.
No API key required.
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path("/home/jam/PLM")

# DailyMed API base
DAILYMED_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# Section 12.3 LOINC code
PK_LOINC = "43682-4"  # PHARMACOKINETICS section


def search_drugs(name: str) -> list[dict]:
    """Search DailyMed for a drug by name."""
    url = f"{DAILYMED_API}/spls.json"
    params = {"drug_name": name, "page": 1, "pagesize": 5}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("data", [])


def get_spl_sections(setid: str) -> str | None:
    """Get the full SPL XML for a drug label."""
    url = f"{DAILYMED_API}/spls/{setid}.xml"
    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        return None
    return r.text


def extract_pk_section(xml_text: str) -> str | None:
    """Extract Section 12.3 (Pharmacokinetics) text from SPL XML."""
    try:
        # SPL uses HL7 namespace
        ns = {"hl7": "urn:hl7-org:v3"}
        root = ET.fromstring(xml_text)

        # Find section with code = PK LOINC
        for section in root.iter("{urn:hl7-org:v3}section"):
            code = section.find("{urn:hl7-org:v3}code")
            if code is not None and code.get("code") == PK_LOINC:
                # Extract all text from this section
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


def parse_cmax_from_pk(pk_text: str) -> list[dict]:
    """Parse Cmax values from PK section text."""
    results = []

    # Pattern: "Cmax of/was/is X ng/mL" or "Cmax (ng/mL): X" or "mean Cmax X ng/mL"
    cmax_patterns = [
        # "Cmax of 340 ng/mL" or "Cmax was 340 ng/mL"
        re.compile(
            r"C\s*max\s+(?:of|was|is|=|:)\s*(?:approximately\s+)?(\d+[\d,.]*)\s*"
            r"(ng/mL|ng/ml|µg/mL|ug/mL|mg/L|mcg/mL|ng\.mL)",
            re.IGNORECASE,
        ),
        # "mean Cmax 340 ng/mL"
        re.compile(
            r"(?:mean|median|geometric mean|average)\s+C\s*max\s+(?:of\s+)?(?:approximately\s+)?(\d+[\d,.]*)\s*"
            r"(ng/mL|ng/ml|µg/mL|ug/mL|mg/L|mcg/mL|ng\.mL)",
            re.IGNORECASE,
        ),
        # "Cmax 340 ng/mL" (direct)
        re.compile(
            r"C\s*max\s*\(?(?:ng/mL|ng/ml)?\)?\s*(?:of\s+)?(\d+[\d,.]*)\s*"
            r"(ng/mL|ng/ml|µg/mL|ug/mL|mg/L|mcg/mL)",
            re.IGNORECASE,
        ),
    ]

    # Dose patterns in surrounding context
    dose_re = re.compile(
        r"(\d+(?:\.\d+)?)\s*[-–]?\s*mg\b(?!\s*/\s*kg)",
        re.IGNORECASE,
    )

    for pattern in cmax_patterns:
        for match in pattern.finditer(pk_text):
            cmax_str = match.group(1).replace(",", "")
            unit = match.group(2)
            try:
                cmax = float(cmax_str)
            except ValueError:
                continue

            # Convert units to ng/mL
            unit_lower = unit.lower().replace(".", "/").replace(" ", "")
            if "ug" in unit_lower or "µg" in unit_lower or "mcg" in unit_lower:
                cmax *= 1000  # µg/mL → ng/mL
            elif "mg/l" in unit_lower:
                cmax *= 1000  # mg/L → ng/mL

            if cmax <= 0:
                continue

            # Find dose in context (200 chars around match)
            start = max(0, match.start() - 300)
            end = min(len(pk_text), match.end() + 100)
            context = pk_text[start:end]

            dose_match = dose_re.search(context)
            dose = float(dose_match.group(1)) if dose_match else None

            # Skip mg/kg
            if dose_match:
                around = context[max(0, dose_match.start()-5):dose_match.end()+10]
                if "mg/kg" in around or "mg kg" in around.lower():
                    dose = None

            results.append({
                "cmax_ngml": cmax,
                "unit_raw": unit,
                "dose_mg": dose,
                "context": context[:200].strip(),
            })

    return results


def main():
    # Load v11 drug names for overlap check
    v11 = json.loads((ROOT / "data/curated/plm_dataset_v11_llm.json").read_text())
    v11_names = set()
    for r in v11:
        dn = (r.get("drug_name") or "").lower().strip()
        if dn:
            v11_names.add(dn)

    # Test drugs - mix of likely-in-v11 and likely-novel
    test_drugs = [
        "apremilast",       # oral, PsA/psoriasis
        "baricitinib",      # oral JAK inhibitor
        "tofacitinib",      # oral JAK inhibitor
        "eltrombopag",      # oral TPO agonist
        "avatrombopag",     # oral TPO agonist
        "acalabrutinib",    # oral BTK inhibitor
        "zanubrutinib",     # oral BTK inhibitor
        "fedratinib",       # oral JAK2 inhibitor
        "glasdegib",        # oral hedgehog inhibitor
        "duvelisib",        # oral PI3K inhibitor
        "gilteritinib",     # oral FLT3 inhibitor
        "midostaurin",      # oral FLT3 inhibitor
        "enasidenib",       # oral IDH2 inhibitor (actually oral? check)
        "ivosidenib",       # oral IDH1 inhibitor
        "selinexor",        # oral XPO1 inhibitor
    ]

    print("=== DailyMed PK Scout ===")
    print(f"v11 has {len(v11_names)} unique drug names")

    all_results = []
    for drug in test_drugs:
        in_v11 = drug.lower() in v11_names
        print(f"\n--- {drug} {'(IN v11)' if in_v11 else '(NOVEL)'} ---")

        results = search_drugs(drug)
        if not results:
            print(f"  No DailyMed results")
            continue

        # Take the first (usually most current) result
        setid = results[0].get("setid")
        title = results[0].get("title", "")[:80]
        print(f"  Found: {title}")
        print(f"  SetID: {setid}")

        xml = get_spl_sections(setid)
        if not xml:
            print(f"  Failed to get SPL XML")
            continue

        pk_text = extract_pk_section(xml)
        if not pk_text:
            print(f"  No Section 12.3 found")
            continue

        print(f"  PK section: {len(pk_text)} chars")

        # Parse Cmax
        pk_entries = parse_cmax_from_pk(pk_text)
        if pk_entries:
            for e in pk_entries:
                print(f"  → Cmax={e['cmax_ngml']} ng/mL, dose={e['dose_mg']}mg")
                e["drug_name"] = drug
                e["in_v11"] = in_v11
                all_results.append(e)
        else:
            # Show first 200 chars of PK text for debugging
            print(f"  No Cmax parsed. PK text preview:")
            print(f"  {pk_text[:300]}")

        time.sleep(0.5)  # Be nice to DailyMed

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(all_results)} Cmax entries from {len(test_drugs)} drugs")
    novel_with_cmax = [r for r in all_results if not r["in_v11"] and r["dose_mg"]]
    print(f"Novel drugs with Cmax+dose: {len(novel_with_cmax)}")
    for r in novel_with_cmax:
        print(f"  {r['drug_name']} {r['dose_mg']}mg → {r['cmax_ngml']} ng/mL")

    # Save
    out_path = ROOT / "data/curated/dailymed_pk_scout.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
