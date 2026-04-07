"""
Extract ADME features from DailyMed FDA drug labels for holdout drugs.

Uses DailyMed API (no auth needed) → SPL XML → Clinical Pharmacology text → regex ADME parsing.
Targets holdout drugs missing TDC ADME features.

NOT extracting Cmax (holdout contamination). Only drug properties: F, PPB, t1/2, CL, Vd, CYP, transporters.

Usage:
    python -m pipeline.dailymed_adme_extractor
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


def get_spl_text(drug_name: str) -> str | None:
    """Fetch Clinical Pharmacology section from DailyMed for a drug."""
    # Search for drug
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name={drug_name}&page=1&pagesize=3"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PLM/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None

    if not data.get("data"):
        return None

    setid = data["data"][0]["setid"]

    # Get SPL XML
    xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml"
    try:
        req = urllib.request.Request(xml_url, headers={"User-Agent": "PLM/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_text = resp.read().decode("utf-8")
    except Exception:
        return None

    # Parse XML, extract Clinical Pharmacology + Pharmacokinetics sections
    ns = {"hl7": "urn:hl7-org:v3"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    pk_text = ""
    for section in root.findall(".//hl7:component/hl7:section", ns):
        title_elem = section.find("hl7:title", ns)
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip().upper()
            if "CLINICAL PHARMACOLOGY" in title or "PHARMACOKINETICS" in title:
                for elem in section.iter():
                    if elem.text:
                        pk_text += elem.text + " "
                    if elem.tail:
                        pk_text += elem.tail + " "

    pk_text = re.sub(r"\s+", " ", pk_text).strip()
    return pk_text if len(pk_text) > 100 else None


def parse_adme(text: str) -> dict:
    """Parse ADME features from Clinical Pharmacology text."""
    features = {}

    # Bioavailability
    for pattern in [
        r"(?:absolute\s+)?bioavailability\s+(?:of\s+\w+\s+)?(?:is\s+)?(?:approximately\s+|about\s+|~)?(\d+(?:\.\d+)?)\s*%",
        r"bioavailability\s+(?:was\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?)\s*%",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            features["bioavailability_pct"] = float(m.group(1))
            features["bioavailability_binary"] = 1.0 if float(m.group(1)) >= 20 else 0.0
            break

    # Protein binding
    for pattern in [
        r"(?:plasma\s+)?protein\s+binding\s+(?:in\s+humans\s+)?(?:is\s+)?(?:approximately\s+|about\s+|~)?(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s+(?:bound\s+to|protein\s+bound)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            features["ppb_pct"] = float(m.group(1))
            break

    # Half-life
    for pattern in [
        r"(?:terminal\s+)?(?:elimination\s+)?half[- ]?life\s+(?:of\s+\w+\s+)?(?:is\s+|of\s+)?(?:approximately\s+|about\s+|~)?(\d+(?:\.\d+)?)\s*(?:to\s+(\d+(?:\.\d+)?)\s*)?(?:hours?|h\b)",
        r"(?:apparent\s+)?half[- ]?life\s+(?:of\s+)?(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*(?:to\s+(\d+(?:\.\d+)?)\s*)?(?:hours?|h\b)",
        r"t½\s+(?:is\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?)\s*(?:hours?|h\b)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            v1 = float(m.group(1))
            v2 = float(m.group(2)) if m.group(2) else v1
            features["half_life_h"] = round((v1 + v2) / 2, 2)
            break

    # Volume of distribution
    for pattern in [
        r"(?:volume\s+of\s+distribution|Vss|Vd|Vz)\s+(?:\(Vss\)\s+)?(?:is\s+|of\s+)?(?:approximately\s+|about\s+|~)?(\d+(?:\.\d+)?)\s*(?:liters?|L\b)",
        r"(?:volume\s+of\s+distribution|Vd)\s+(?:is\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?)\s*L/kg",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if "L/kg" in pattern:
                features["vd_L_kg"] = val
            else:
                features["vd_L"] = val
            break

    # Clearance
    for pattern in [
        r"(?:total\s+body\s+|apparent\s+|oral\s+)?(?:clearance|CL/F|CL)\s+(?:is\s+|of\s+)?(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*L/(?:h|hr|hour)",
        r"(?:clearance|CL/F)\s+(?:is\s+)?(?:approximately\s+)?(\d+(?:\.\d+)?)\s*mL/min",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if "mL/min" in pattern:
                features["clearance_Lh"] = round(val * 0.06, 2)
            else:
                features["clearance_Lh"] = val
            break

    # CYP enzymes
    cyp_matches = re.findall(r"CYP\s*(\d[A-Z]\d+)", text)
    if cyp_matches:
        cyps = set(cyp_matches)
        features["cyp_enzymes"] = sorted(cyps)
        features["cyp3a4_substrate"] = 1.0 if "3A4" in cyps or "3A5" in cyps else 0.0
        features["cyp2d6_substrate"] = 1.0 if "2D6" in cyps else 0.0
        features["cyp2c9_substrate"] = 1.0 if "2C9" in cyps else 0.0
        features["cyp2c19_substrate"] = 1.0 if "2C19" in cyps else 0.0
        features["cyp1a2_substrate"] = 1.0 if "1A2" in cyps else 0.0

    # Transporters
    pgp = bool(re.search(r"(?:substrate\s+of|transported\s+by).*?P-?g(?:lyco)?p|P-?gp.*substrate", text, re.IGNORECASE))
    bcrp = bool(re.search(r"(?:substrate|transported).*?BCRP|BCRP.*substrate|breast\s+cancer\s+resistance\s+protein", text, re.IGNORECASE))
    features["pgp_substrate"] = 1.0 if pgp else 0.0
    features["bcrp_substrate"] = 1.0 if bcrp else 0.0

    return features


def main():
    print("=" * 70)
    print("DAILYMED ADME FEATURE EXTRACTION")
    print("=" * 70)
    print("Extracting ADME properties (NOT Cmax) from FDA drug labels")
    print()

    # Load holdout drugs needing features
    with open("data/validation/holdout_definition.json") as f:
        ho = json.load(f)
    with open("data/curated/tdc_adme_data.json") as f:
        tdc = json.load(f)

    # Target all 97 holdout drugs (fill any missing features)
    results = []
    for d in ho["holdout_drugs"]:
        name = d["name"]
        ik = d["inchikey14"]

        # Check current TDC coverage
        e = tdc.get(ik, {})
        n_existing = sum(1 for k in ["bioavailability_binary", "half_life_h", "caco2_logPapp", "ppb_pct",
                                      "clearance_ul_min_mg", "clearance_ul_min_million_cells", "vd_L_kg"]
                        if e.get(k) is not None)

        print(f"  {name:<25s} (TDC features: {n_existing})...", end=" ", flush=True)

        pk_text = get_spl_text(name)
        if pk_text is None:
            print("NOT FOUND")
            results.append({"drug": name, "ik": ik, "status": "not_found", "tdc_existing": n_existing})
            time.sleep(0.3)
            continue

        features = parse_adme(pk_text)
        n_new = sum(1 for k, v in features.items()
                   if k not in ("cyp_enzymes",) and v is not None)

        print(f"{n_new} features extracted")
        results.append({
            "drug": name,
            "ik": ik,
            "status": "ok",
            "tdc_existing": n_existing,
            "n_new_features": n_new,
            "features": features,
        })
        time.sleep(0.3)

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{'='*70}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total holdout drugs: 97")
    print(f"  DailyMed found: {len(ok)}")
    print(f"  Not found: {len(results) - len(ok)}")
    print()

    feat_counts = {}
    for feat in ["bioavailability_pct", "ppb_pct", "half_life_h", "clearance_Lh",
                 "vd_L", "cyp3a4_substrate", "pgp_substrate", "bcrp_substrate"]:
        n = sum(1 for r in ok if feat in r.get("features", {}))
        feat_counts[feat] = n
        print(f"    {feat:<25s}: {n}/{len(ok)}")

    # Count NEW fills (features that were NaN in TDC but now have values)
    new_fills = 0
    for r in ok:
        ik = r["ik"]
        e = tdc.get(ik, {})
        feats = r.get("features", {})
        if feats.get("half_life_h") and e.get("half_life_h") is None:
            new_fills += 1
        if feats.get("ppb_pct") and e.get("ppb_pct") is None:
            new_fills += 1
        if feats.get("bioavailability_pct") and e.get("bioavailability_binary") is None:
            new_fills += 1
        if feats.get("vd_L") and e.get("vd_L_kg") is None:
            new_fills += 1

    print(f"\n  New TDC NaN fills: {new_fills}")

    # Save
    out_path = "data/curated/dailymed_adme_features.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
