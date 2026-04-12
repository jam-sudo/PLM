"""Bulk DailyMed extraction — scan ALL oral small molecule drugs.

Strategy:
1. Use DailyMed search API to find all oral tablet/capsule drugs
2. For each, extract Section 12.3 PK text
3. Parse Cmax with improved regex + LLM-style pattern matching
4. Get SMILES from PubChem
5. Filter against v11/v12/holdout
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
PK_LOINC = "43682-4"

# Comprehensive Cmax patterns
_CV_OPT = r"(?:\s*\(\s*\d+[\d.]*\s*%?\s*\)\s*)?"  # optional (CV%)
_SD_OPT = r"(?:\s*\(\s*\±?\s*\d+[\d.]*\s*\)\s*)?"  # optional (±SD)
_BOTH_OPT = r"(?:\s*\([\s\d.±%]*\)\s*)?"  # either (CV%) or (±SD)

CMAX_PATTERNS = [
    # "C max is/was/of 1234 (CV%) ng/mL"
    re.compile(
        r"C\s*_?\s*max\s*(?:,?\s*ss)?\s*\)?\s*(?:is|was|of|=|:)\s*(?:approximately\s+|~)?"
        r"(\d+[\d,.]*)" + _BOTH_OPT + r"\s*"
        r"(ng/mL|ng/ml|µg/mL|ug/mL|mcg/mL|mg/L|mcg/L|ng\.mL\S*|ug\.mL\S*)",
        re.IGNORECASE,
    ),
    # "mean C max (±SD) of 1234 ng/mL"
    re.compile(
        r"(?:mean|geometric mean|median|average)\s+(?:\(?\s*[±%\dCV .]*\s*\)?\s*)?"
        r"(?:peak\s+)?(?:plasma\s+)?(?:concentration\s+)?"
        r"\(?C\s*_?\s*max\s*\)?\s*" + _BOTH_OPT +
        r"(?:is|was|of|=|:)?\s*(?:approximately\s+|~)?"
        r"(\d+[\d,.]*)" + _BOTH_OPT + r"\s*"
        r"(ng/mL|ng/ml|µg/mL|ug/mL|mcg/mL|mg/L|mcg/L)",
        re.IGNORECASE,
    ),
    # "C max 1234 (CV%) ng/mL" direct
    re.compile(
        r"C\s*_?\s*max\s*(?:,?\s*ss)?\s*\)?\s+(\d+[\d,.]*)" + _BOTH_OPT + r"\s*"
        r"(ng/mL|ng/ml|µg/mL|ug/mL|mcg/mL|mg/L|mcg/L)",
        re.IGNORECASE,
    ),
    # "C max (unit) value" — table format
    re.compile(
        r"C\s*_?\s*max\s*(?:,?\s*ss)?\s*\(\s*"
        r"(ng/mL|ng/ml|µg/mL|mcg/mL|mg/L|mcg/L)"
        r"\s*\)\s*(\d+[\d,.]*)",
        re.IGNORECASE,
    ),
    # "and 563 (29%) ng/mL" after AUC (only if Cmax mentioned nearby)
    re.compile(
        r"and\s+(\d+[\d,.]*)" + _BOTH_OPT + r"\s*"
        r"(ng/mL|ng/ml|µg/mL|ug/mL|mcg/mL|mg/L|mcg/L)"
        r"(?:\s*,?\s*respectively)?",
        re.IGNORECASE,
    ),
]

DOSE_PATTERNS = [
    re.compile(r"(?:single\s+)?(?:oral\s+)?(?:dose\s+(?:of\s+)?)?(\d+(?:\.\d+)?)\s*mg\b(?!\s*/\s*(?:kg|m))", re.IGNORECASE),
    re.compile(r"following\s+(\d+(?:\.\d+)?)\s*mg", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*mg\s+(?:once|twice|daily|BID|QD|orally|tablet|capsule)", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*mg\s+(?:dose|oral)", re.IGNORECASE),
    re.compile(r"at\s+(\d+(?:\.\d+)?)\s*mg", re.IGNORECASE),
]


def cmax_to_ngml(value: float, unit: str) -> float:
    u = unit.lower().replace(".", "/").replace(" ", "")
    if any(x in u for x in ["mcg/ml", "ug/ml", "µg/ml"]):
        return value * 1000
    elif "mg/l" in u:
        return value * 1000
    elif "mcg/l" in u:
        return value  # mcg/L = ng/mL
    return value


def extract_pk(drug_name: str) -> list[dict]:
    """Extract Cmax entries from DailyMed for a drug."""
    try:
        r = requests.get(f"{DAILYMED_API}/spls.json",
                        params={"drug_name": drug_name, "pagesize": 1}, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json().get("data", [])
        if not data:
            return []
        setid = data[0]["setid"]

        xml_r = requests.get(f"{DAILYMED_API}/spls/{setid}.xml", timeout=15)
        if xml_r.status_code != 200:
            return []

        root = ET.fromstring(xml_r.text)
        pk_text = None
        for section in root.iter("{urn:hl7-org:v3}section"):
            code = section.find("{urn:hl7-org:v3}code")
            if code is not None and code.get("code") == PK_LOINC:
                texts = []
                for elem in section.iter():
                    if elem.text: texts.append(elem.text.strip())
                    if elem.tail: texts.append(elem.tail.strip())
                pk_text = " ".join(t for t in texts if t)
                break

        if not pk_text:
            return []

        results = []
        seen = set()

        for pi, pattern in enumerate(CMAX_PATTERNS):
            for match in pattern.finditer(pk_text):
                groups = [g for g in match.groups() if g is not None]
                if len(groups) < 2:
                    continue

                # "and X ng/mL" pattern needs Cmax context
                if pi == 4:
                    lookback = pk_text[max(0, match.start()-200):match.start()]
                    if not re.search(r"C\s*_?\s*max", lookback, re.IGNORECASE):
                        continue
                    if any(kw in lookback.lower() for kw in ["decreased", "increased", "fold"]):
                        continue

                # Parse value and unit
                unit_set = {"ng/mL", "ng/ml", "µg/mL", "ug/mL", "mcg/mL", "mg/L", "mcg/L"}
                cmax_str, unit = None, None
                for g in groups:
                    g_clean = g.lower().replace(".", "/")
                    if any(u.lower() in g_clean for u in unit_set):
                        unit = g
                    elif cmax_str is None and re.match(r"\d", g):
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

                # Find dose
                start = max(0, match.start() - 500)
                end = min(len(pk_text), match.end() + 200)
                context = pk_text[start:end]

                dose = None
                for dp in DOSE_PATTERNS:
                    dm = dp.search(context)
                    if dm:
                        dval = float(dm.group(1))
                        around = context[max(0, dm.start()-5):dm.end()+15]
                        if "mg/kg" in around or "mg/m" in around:
                            continue
                        if 0.01 <= dval <= 5000:
                            dose = dval
                            break

                if dose is None:
                    continue

                log_cd = math.log10(cmax_ngml / dose)
                if log_cd < -3 or log_cd > 4:
                    continue

                key = (round(cmax_ngml, 1), dose)
                if key in seen:
                    continue
                seen.add(key)

                results.append({
                    "cmax_ngml": cmax_ngml,
                    "dose_mg": dose,
                    "log_cd": round(log_cd, 4),
                })

        return results
    except Exception:
        return []


def get_smiles(name: str) -> tuple[str | None, str | None]:
    """Get SMILES from PubChem."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES,CanonicalSMILES,InChIKey/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None, None
        props = r.json().get("PropertyTable", {}).get("Properties", [{}])[0]
        smi = props.get("CanonicalSMILES") or props.get("IsomericSMILES") or props.get("SMILES")
        ik = props.get("InChIKey", "")
        return smi, ik[:14] if ik else None
    except Exception:
        return None, None


def main():
    v11 = json.loads((ROOT / "data/curated/plm_dataset_v11_llm.json").read_text())
    v12 = json.loads((ROOT / "data/curated/plm_dataset_v12_chembl.json").read_text())
    ho = json.loads((ROOT / "data/validation/holdout_definition.json").read_text())
    v12_iks = set((r.get("ik") or "")[:14] for r in v12 if r.get("ik"))
    ho_iks = set((k or "")[:14] for k in ho["holdout_inchikeys"])

    # Extended drug list — ~300 FDA-approved oral small molecules
    # Focus on drugs NOT likely in v11 (newer, less common, specialty)
    drugs = [
        # 2023-2026 approvals (high chance of being novel)
        "pirtobrutinib", "repotrectinib", "omadacycline", "capivasertib",
        "elacestrant", "olutasidenib", "adagrasib", "tremelimumab",
        "futibatinib", "pacritinib", "mitapivat", "oteseconazole",
        "vosoritide", "teclistamab", "ganaxolone", "maribavir",
        "tivdak", "mavacamten", "deucravacitinib", "abrocitinib",
        "tebipenem", "avacopan", "voclosporin", "belzutifan",
        "infigratinib", "umbralisib", "loncastuximab", "trilaciclib",
        "pralsetinib", "capmatinib", "selpercatinib", "tucatinib",
        "ripretinib", "avapritinib", "selinexor", "fedratinib",
        "zanubrutinib", "erdafitinib", "glasdegib", "gilteritinib",
        "ivosidenib", "duvelisib", "talazoparib", "lorlatinib",
        "larotrectinib", "entrectinib", "quizartinib", "alpelisib",
        "pemigatinib", "mobocertinib",
        # Specialty/rare disease oral
        "eliglustat", "miglustat", "cerliponase", "elosulfase",
        "tafamidis", "patisiran", "inotersen", "givosiran",
        "odevixibat", "maralixibat", "pitolisant", "solriamfetol",
        "lemborexant", "suvorexant", "daridorexant",
        "avatrombopag", "fostamatinib", "acalabrutinib",
        "revumenib", "imetelstat", "tovorafenib",
        # Anti-infective oral
        "lefamulin", "pretomanid", "bedaquiline", "tedizolid",
        "delafloxacin", "cefiderocol", "plazomicin",
        "baloxavir", "letermovir", "maribavir",
        "cabotegravir", "doravirine", "bictegravir",
        "lenacapavir", "fostemsavir",
        # Cardiovascular/metabolic newer
        "vericiguat", "sacubitril", "bempedoic acid",
        "aficamten", "ticagrelor", "rivaroxaban",
        "apixaban", "edoxaban", "betrixaban",
        # Neurology oral
        "ozanimod", "siponimod", "ponesimod", "fingolimod",
        "ofatumumab", "ubrogepant", "rimegepant", "atogepant",
        "erenumab", "cenobamate", "fenfluramine",
        # Immunology/dermatology oral
        "apremilast", "baricitinib", "tofacitinib", "upadacitinib",
        "filgotinib", "ruxolitinib", "elagolix", "relugolix",
        "linzagolix",
        # Oncology established oral
        "rucaparib", "niraparib", "olaparib", "vemurafenib",
        "dabrafenib", "encorafenib", "trametinib", "cobimetinib",
        "binimetinib", "sonidegib", "vismodegib", "venetoclax",
        "ixazomib", "panobinostat", "vorinostat",
        "neratinib", "axitinib", "cabozantinib", "lenvatinib",
        "tivozanib", "pazopanib", "sunitinib", "sorafenib",
        "tepotinib", "enasidenib", "idelalisib",
        # GI/hepatology oral
        "eluxadoline", "rifaximin", "vonoprazan",
        "tegoprazan", "obeticholic acid", "seladelpar",
    ]

    drugs = list(dict.fromkeys(drugs))  # dedup
    print(f"Scanning {len(drugs)} drugs...")
    print(f"v12: {len(v12_iks)} IK14s, HO: {len(ho_iks)}")

    all_entries = []
    stats = {"total": 0, "no_dm": 0, "no_cmax": 0, "no_smi": 0, "holdout": 0,
             "in_v12": 0, "novel": 0}

    for i, drug in enumerate(drugs):
        if i > 0 and i % 20 == 0:
            print(f"  {i}/{len(drugs)}: {stats['novel']} novel, {stats['in_v12']} v12, {len(all_entries)} total")

        pk_entries = extract_pk(drug)
        if not pk_entries:
            stats["no_cmax"] += 1
            time.sleep(0.3)
            continue

        smi, ik14 = get_smiles(drug)
        if not smi:
            stats["no_smi"] += 1
            time.sleep(0.3)
            continue

        in_v12 = ik14 in v12_iks if ik14 else False
        in_ho = ik14 in ho_iks if ik14 else False

        if in_ho:
            stats["holdout"] += 1
            time.sleep(0.3)
            continue

        for e in pk_entries:
            entry = {
                "drug_name": drug, "smiles": smi, "ik": ik14,
                "dose_mg": e["dose_mg"], "cmax_ngml": e["cmax_ngml"],
                "log_cd": e["log_cd"], "src": "DailyMed_S13",
            }
            all_entries.append(entry)

            is_novel = not in_v12
            if is_novel:
                stats["novel"] += 1
                print(f"  NEW: {drug} {e['dose_mg']}mg → {e['cmax_ngml']} ng/mL (log_cd={e['log_cd']:.2f})")
            else:
                stats["in_v12"] += 1

        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"BULK EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Drugs scanned: {len(drugs)}")
    print(f"No Cmax parsed: {stats['no_cmax']}")
    print(f"No SMILES: {stats['no_smi']}")
    print(f"Holdout excluded: {stats['holdout']}")
    print(f"Already in v12: {stats['in_v12']}")
    print(f"Novel entries: {stats['novel']}")
    print(f"Total entries: {len(all_entries)}")

    novel_entries = [e for e in all_entries if e["ik"] not in v12_iks]
    print(f"\nNovel entries for v13:")
    for e in novel_entries:
        print(f"  {e['drug_name']} {e['dose_mg']}mg → {e['cmax_ngml']} ng/mL")

    out = {
        "extraction_date": "2026-04-12",
        "stats": stats,
        "entries": all_entries,
        "novel_entries": novel_entries,
    }
    out_path = ROOT / "data/curated/dailymed_bulk_extracted.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
