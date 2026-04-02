"""
Caption & Metadata Parser

Extracts PK metadata from figure captions and surrounding PDF text.
Looks up SMILES via PubChem API.

Usage:
    python -m pipeline.caption_parser --caption "Mean plasma concentration of empagliflozin 25mg oral"
    python -m pipeline.caption_parser --drug empagliflozin --lookup-smiles
"""

import re
import time
import requests
from typing import Optional


PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Route keywords
ROUTE_KEYWORDS = {
    "oral": ["oral", "po", "tablet", "capsule", "suspension", "solution"],
    "IV_bolus": ["iv bolus", "intravenous bolus"],
    "IV_infusion": ["iv infusion", "intravenous infusion", "iv drip"],
    "SC_injection": ["subcutaneous", "sc injection", "sc dose", "s.c."],
    "IM_injection": ["intramuscular", "im injection", "i.m."],
    "transdermal": ["transdermal", "patch", "topical"],
    "sublingual": ["sublingual"],
}

# Formulation keywords
FORMULATION_KEYWORDS = {
    "IR_tablet": ["immediate release tablet", "ir tablet", "tablet"],
    "IR_capsule": ["capsule", "softgel"],
    "ER_tablet": ["extended release", "er tablet", "xr tablet", "modified release"],
    "ER_capsule": ["er capsule", "xr capsule"],
    "solution": ["solution", "oral solution"],
    "suspension": ["suspension"],
    "IV_bolus": ["iv bolus"],
    "IV_infusion": ["iv infusion"],
    "SC_injection": ["subcutaneous", "sc injection", "prefilled syringe"],
    "IM_injection": ["intramuscular"],
    "transdermal": ["transdermal", "patch"],
}

# Food effect keywords
FOOD_KEYWORDS = {
    "fasted": ["fasted", "fasting", "empty stomach"],
    "fed": ["fed", "with food", "after meal", "high-fat", "low-fat", "with breakfast"],
}


def extract_dose(text: str) -> Optional[float]:
    """Extract dose in mg from caption text."""
    patterns = [
        r'(\d+(?:\.\d+)?)\s*mg(?:\s+(?:oral|dose|single|tablet|capsule))?',
        r'dose[:\s]+(\d+(?:\.\d+)?)\s*mg',
        r'(\d+(?:\.\d+)?)\s*mg/(?:kg|m2)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_route(text: str) -> str:
    """Extract route of administration from text."""
    lower = text.lower()
    for route, keywords in ROUTE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return route
    return "not_specified"


def extract_formulation(text: str) -> str:
    """Extract formulation type from text."""
    lower = text.lower()
    for form, keywords in FORMULATION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return form
    return "other"


def extract_food_effect(text: str) -> str:
    """Extract food effect condition from text."""
    lower = text.lower()
    for effect, keywords in FOOD_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return effect
    return "not_specified"


def extract_n_subjects(text: str) -> Optional[int]:
    """Extract number of subjects from text."""
    patterns = [
        r'[Nn]\s*=\s*(\d+)',
        r'(\d+)\s+(?:subjects|patients|volunteers|participants)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def extract_concentration_unit(text: str) -> str:
    """Extract concentration unit from text."""
    units = ["ng/mL", "μg/mL", "mg/L", "nmol/L", "ng/dL", "pg/mL"]
    lower = text.lower()
    for unit in units:
        if unit.lower() in lower:
            return unit
    return "ng/mL"


def lookup_smiles(drug_name: str) -> Optional[str]:
    """Look up SMILES from PubChem by drug name."""
    url = f"{PUBCHEM_API}/compound/name/{drug_name}/JSON"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for prop in data["PC_Compounds"][0].get("props", []):
                label = prop.get("urn", {}).get("label", "")
                name = prop.get("urn", {}).get("name", "")
                if label == "SMILES" and name == "Isomeric":
                    return prop["value"]["sval"]
            # Fallback: any SMILES
            for prop in data["PC_Compounds"][0].get("props", []):
                if prop.get("urn", {}).get("label") == "SMILES":
                    return prop["value"]["sval"]
    except Exception:
        pass
    return None


def parse_caption(caption: str) -> dict:
    """Extract all metadata from a figure caption."""
    return {
        "dose_mg": extract_dose(caption),
        "route": extract_route(caption),
        "formulation": extract_formulation(caption),
        "food_effect": extract_food_effect(caption),
        "n_subjects": extract_n_subjects(caption),
        "concentration_unit": extract_concentration_unit(caption),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--caption", type=str, help="Caption text to parse")
    parser.add_argument("--drug", type=str, help="Drug name for SMILES lookup")
    parser.add_argument("--lookup-smiles", action="store_true")
    args = parser.parse_args()

    if args.caption:
        result = parse_caption(args.caption)
        print("Parsed metadata:")
        for k, v in result.items():
            print(f"  {k}: {v}")

    if args.drug and args.lookup_smiles:
        smiles = lookup_smiles(args.drug)
        print(f"\nSMILES for {args.drug}: {smiles or 'Not found'}")
