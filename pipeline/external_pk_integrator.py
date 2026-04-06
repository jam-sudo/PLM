"""
External PK Database Integrator — Fetch and merge PK data from public sources.

Data Sources:
  1. PK-DB (pk-db.org) — Curated PK parameter database via REST API
  2. ChEMBL — Bioactivity DB with clinical PK assays (already partially done)
  3. DailyMed / FDA Labels — Structured PK sections from drug labels
  4. PubChem — SMILES validation and cross-referencing

Strategy:
  - Fetch Cmax, AUC, Tmax, t1/2 from each source
  - Normalize units to canonical (ng/mL, ng·h/mL, h)
  - Match to existing dataset via InChIKey14 (first 14 chars)
  - Identify NEW drugs not in current training set
  - Validate against existing data where overlap exists

Output: data/curated/external_pk_integrated.json
"""

import json
import time
import math
import re
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, inchi
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ─── Unit Conversion ─────────────────────────────────────────────

CONC_CONVERSIONS = {
    'ng/ml': 1.0, 'ng/mL': 1.0,
    'ug/ml': 1000.0, 'ug/mL': 1000.0,
    'µg/ml': 1000.0, 'µg/mL': 1000.0,
    'mcg/ml': 1000.0, 'mcg/mL': 1000.0,
    'mg/l': 1000.0, 'mg/L': 1000.0,
    'mg/ml': 1e6, 'mg/mL': 1e6,
    'pg/ml': 0.001, 'pg/mL': 0.001,
    'nmol/l': None,  # needs MW
    'µmol/l': None,
}


def smiles_to_inchikey14(smiles: str) -> Optional[str]:
    """Convert SMILES to first 14 chars of InChIKey."""
    if not HAS_RDKIT or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        ik = inchi.InchiToInchiKey(inchi.MolToInchi(mol))
        return ik[:14] if ik else None
    except:
        return None


def convert_concentration(value: float, unit: str, mw: float = None) -> Optional[float]:
    """Convert concentration to ng/mL."""
    unit_lower = unit.strip().lower()
    for k, factor in CONC_CONVERSIONS.items():
        if k.lower() == unit_lower:
            if factor is None:
                if mw and mw > 0:
                    if 'nmol' in unit_lower or 'nm' == unit_lower:
                        return value * mw / 1000.0  # nM * g/mol / 1000 = ng/mL
                    elif 'µmol' in unit_lower or 'um' == unit_lower:
                        return value * mw  # µM * g/mol = ng/mL * 1000... needs care
                return None
            return value * factor
    return None


# ─── PK-DB Fetcher ───────────────────────────────────────────────

PKDB_BASE = "https://pk-db.org/api/v1"


def fetch_pkdb_data(max_pages: int = 50) -> List[Dict]:
    """Fetch PK data from PK-DB REST API."""
    if not HAS_REQUESTS:
        print("  requests module not available, skipping PK-DB")
        return []

    all_entries = []
    page = 1

    while page <= max_pages:
        try:
            url = f"{PKDB_BASE}/pk_data/?format=json&page={page}&page_size=100"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                break
            if resp.status_code == 429:
                print(f"  PK-DB rate limited, waiting 10s...")
                time.sleep(10)
                continue
            if resp.status_code != 200:
                print(f"  PK-DB page {page} returned {resp.status_code}")
                break

            data = resp.json()
            results = data.get('results', data) if isinstance(data, dict) else data

            if not results:
                break

            for entry in results:
                all_entries.append(entry)

            # Check pagination
            if isinstance(data, dict) and data.get('next'):
                page += 1
                time.sleep(0.5)  # polite rate limiting
            else:
                break

        except requests.exceptions.RequestException as e:
            print(f"  PK-DB error on page {page}: {e}")
            break

    return all_entries


def parse_pkdb_entries(entries: List[Dict]) -> List[Dict]:
    """Parse PK-DB entries into standardized format."""
    parsed = []
    for entry in entries:
        # PK-DB schema varies, try common fields
        substance = entry.get('substance', {})
        drug_name = substance.get('name', entry.get('substance_name', ''))
        smiles = substance.get('smiles', '')

        # Get PK parameters
        pk_type = entry.get('measurement_type', entry.get('pk_type', ''))
        value = entry.get('value', entry.get('mean', None))
        unit = entry.get('unit', '')
        dose = entry.get('dose', entry.get('dose_mg', None))

        if value is None or not drug_name:
            continue

        record = {
            'drug_name': drug_name,
            'smiles': smiles,
            'source': 'PK-DB',
            'pk_type': pk_type,
            'value_raw': value,
            'unit_raw': unit,
            'dose_mg': dose,
        }

        # Convert Cmax
        if pk_type and any(k in pk_type.lower() for k in ['cmax', 'c_max', 'peak']):
            cmax = convert_concentration(float(value), unit) if value else None
            if cmax and cmax > 0:
                record['cmax_ng_ml'] = cmax
                record['confidence'] = 'high'

        parsed.append(record)

    return parsed


# ─── DailyMed Fetcher ────────────────────────────────────────────

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


def fetch_dailymed_pk(drug_name: str) -> Optional[Dict]:
    """Fetch PK section from DailyMed for a specific drug."""
    if not HAS_REQUESTS:
        return None

    try:
        # Search for drug
        search_url = f"{DAILYMED_BASE}/spls.json?drug_name={drug_name}&page=1&pagesize=1"
        resp = requests.get(search_url, timeout=15)
        if resp.status_code != 200:
            return None

        data = resp.json()
        results = data.get('data', [])
        if not results:
            return None

        setid = results[0].get('setid', '')
        if not setid:
            return None

        # Get SPL sections
        spl_url = f"{DAILYMED_BASE}/spls/{setid}.json"
        resp = requests.get(spl_url, timeout=15)
        if resp.status_code != 200:
            return None

        spl_data = resp.json()
        return {
            'drug_name': drug_name,
            'setid': setid,
            'title': spl_data.get('title', ''),
            'source': 'DailyMed',
        }

    except Exception as e:
        return None


def extract_pk_from_dailymed_text(text: str) -> Dict:
    """Extract PK parameters from DailyMed label text."""
    params = {}

    # Cmax pattern
    cmax_pat = re.compile(
        r'[Cc]max\s*(?:was|of|is|=|:)?\s*([\d.,]+)\s*'
        r'(?:±\s*[\d.,]+\s*)?'
        r'(ng/mL|µg/mL|mcg/mL|mg/L|ng/ml)',
        re.IGNORECASE
    )
    m = cmax_pat.search(text)
    if m:
        val = float(m.group(1).replace(',', ''))
        unit = m.group(2)
        cmax = convert_concentration(val, unit)
        if cmax:
            params['cmax_ng_ml'] = cmax

    # AUC pattern
    auc_pat = re.compile(
        r'AUC\s*(?:0?[-–]?∞|inf|last|0-t)?\s*(?:was|of|is|=|:)?\s*([\d.,]+)\s*'
        r'(?:±\s*[\d.,]+\s*)?'
        r'(ng[·*]h/mL|µg[·*]h/mL|ng\.h/mL|h[·*]ng/mL)',
        re.IGNORECASE
    )
    m = auc_pat.search(text)
    if m:
        params['auc_raw'] = float(m.group(1).replace(',', ''))
        params['auc_unit'] = m.group(2)

    # Tmax
    tmax_pat = re.compile(
        r'[Tt]max\s*(?:was|of|is|=|:)?\s*([\d.,]+)\s*(hours?|h|minutes?|min)',
        re.IGNORECASE
    )
    m = tmax_pat.search(text)
    if m:
        val = float(m.group(1).replace(',', ''))
        if 'min' in m.group(2).lower():
            val /= 60.0
        params['tmax_h'] = val

    # Half-life
    thalf_pat = re.compile(
        r'(?:half[- ]?life|t½|t1/2)\s*(?:was|of|is|=|:)?\s*([\d.,]+)\s*(hours?|h|days?)',
        re.IGNORECASE
    )
    m = thalf_pat.search(text)
    if m:
        val = float(m.group(1).replace(',', ''))
        if 'day' in m.group(2).lower():
            val *= 24.0
        params['t_half_h'] = val

    # Dose from context
    dose_pat = re.compile(r'(\d+(?:\.\d+)?)\s*mg\s*(?:oral|single|dose)', re.IGNORECASE)
    m = dose_pat.search(text)
    if m:
        params['dose_mg'] = float(m.group(1))

    return params


# ─── PubChem SMILES Resolver ─────────────────────────────────────

def resolve_smiles_pubchem(drug_name: str) -> Optional[str]:
    """Resolve drug name to canonical SMILES via PubChem."""
    if not HAS_REQUESTS:
        return None
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        props = data.get('PropertyTable', {}).get('Properties', [])
        if props:
            return props[0].get('CanonicalSMILES')
    except:
        pass
    return None


# ─── ChEMBL Integration (extends existing chembl_expansion.py) ──

def load_existing_chembl() -> List[Dict]:
    """Load already-expanded ChEMBL data."""
    path = Path('data/curated/chembl_pk_expansion.json')
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


# ─── Integration Pipeline ────────────────────────────────────────

def load_existing_datasets() -> Tuple[set, Dict]:
    """Load existing training + HO InChIKeys for dedup."""
    existing_iks = set()
    drug_data = {}

    # Combined dataset
    combined_path = Path('data/curated/plm_sisyphus_combined.json')
    if combined_path.exists():
        with open(combined_path) as f:
            combined = json.load(f)
        for entry in combined:
            ik = entry.get('inchikey14', '')
            if ik:
                existing_iks.add(ik)
                if ik not in drug_data:
                    drug_data[ik] = []
                drug_data[ik].append(entry)

    # Holdout
    ho_path = Path('data/validation/holdout_definition.json')
    if ho_path.exists():
        with open(ho_path) as f:
            ho_def = json.load(f)
        for ik in ho_def.get('holdout_inchikeys', []):
            existing_iks.add(ik)

    return existing_iks, drug_data


def integrate_all_sources(fetch_external: bool = True) -> Dict:
    """Main integration pipeline."""
    print("Loading existing datasets...")
    existing_iks, drug_data = load_existing_datasets()
    print(f"  Existing InChIKeys: {len(existing_iks)}")
    print(f"  Existing drugs with data: {len(drug_data)}")

    all_new = []
    all_enriched = []
    stats = {
        'sources': {},
        'total_new_drugs': 0,
        'total_new_tuples': 0,
        'total_enriched': 0,
    }

    # ── Source 1: FDA PK Table Parser (local) ──
    print("\n[1/4] Loading FDA PK table parsed data...")
    pk_parsed_path = Path('data/curated/pk_table_parsed.json')
    if pk_parsed_path.exists():
        with open(pk_parsed_path) as f:
            pk_parsed = json.load(f)

        new_from_fda = 0
        enriched_from_fda = 0
        for entry in pk_parsed:
            smiles = entry.get('smiles')
            if not smiles:
                continue
            ik = smiles_to_inchikey14(smiles)
            if not ik:
                continue

            entry['inchikey14'] = ik
            entry['source'] = 'FDA_table_parser'

            if ik not in existing_iks:
                all_new.append(entry)
                new_from_fda += 1
            else:
                all_enriched.append(entry)
                enriched_from_fda += 1

        stats['sources']['fda_table_parser'] = {
            'total': len(pk_parsed),
            'with_smiles': sum(1 for e in pk_parsed if e.get('smiles')),
            'new_drugs': new_from_fda,
            'enriched_existing': enriched_from_fda,
        }
        print(f"  Parsed: {len(pk_parsed)}, New drugs: {new_from_fda}, Enriched: {enriched_from_fda}")
    else:
        print("  No parsed data found (run pk_table_parser.py first)")

    # ── Source 2: Existing LLM extractions (local) ──
    print("\n[2/4] Loading LLM extraction data...")
    llm_path = Path('data/llm_extracted/pk_llm_merged.json')
    if llm_path.exists():
        with open(llm_path) as f:
            llm_data = json.load(f)

        new_from_llm = 0
        for entry in llm_data:
            smiles = entry.get('smiles')
            if not smiles:
                continue
            ik = smiles_to_inchikey14(smiles)
            if not ik:
                continue
            if ik not in existing_iks:
                entry['inchikey14'] = ik
                entry['source'] = 'LLM_extraction'
                all_new.append(entry)
                new_from_llm += 1

        stats['sources']['llm_extraction'] = {
            'total': len(llm_data),
            'with_smiles': sum(1 for e in llm_data if e.get('smiles')),
            'new_drugs': new_from_llm,
        }
        print(f"  Loaded: {len(llm_data)}, New drugs not in combined: {new_from_llm}")

    # ── Source 3: ChEMBL expansion (local) ──
    print("\n[3/4] Loading ChEMBL expansion data...")
    chembl_data = load_existing_chembl()
    if chembl_data:
        new_from_chembl = 0
        chembl_with_cmax = 0
        for entry in chembl_data:
            smiles = entry.get('smiles', entry.get('canonical_smiles', ''))
            cmax = entry.get('cmax_ngml', entry.get('standard_value'))
            if not smiles or not cmax:
                continue
            chembl_with_cmax += 1
            ik = smiles_to_inchikey14(smiles)
            if ik and ik not in existing_iks:
                all_new.append({
                    'drug_name': entry.get('drug_name', entry.get('molecule_name', '')),
                    'smiles': smiles,
                    'inchikey14': ik,
                    'cmax_ng_ml': float(cmax),
                    'dose_mg': entry.get('dose_mg'),
                    'source': 'ChEMBL',
                    'confidence': 'medium',
                })
                new_from_chembl += 1

        stats['sources']['chembl'] = {
            'total': len(chembl_data),
            'with_cmax': chembl_with_cmax,
            'new_drugs': new_from_chembl,
        }
        print(f"  Loaded: {len(chembl_data)}, With Cmax: {chembl_with_cmax}, New: {new_from_chembl}")
    else:
        print("  No ChEMBL data found")

    # ── Source 4: PK-DB (external API) ──
    print("\n[4/4] Fetching PK-DB data...")
    if fetch_external and HAS_REQUESTS:
        try:
            pkdb_raw = fetch_pkdb_data(max_pages=20)
            pkdb_parsed = parse_pkdb_entries(pkdb_raw)

            new_from_pkdb = 0
            for entry in pkdb_parsed:
                smiles = entry.get('smiles', '')
                if not smiles:
                    drug_name = entry.get('drug_name', '')
                    if drug_name:
                        smiles = resolve_smiles_pubchem(drug_name)
                        if smiles:
                            entry['smiles'] = smiles
                            entry['smiles_source'] = 'pubchem'
                        time.sleep(0.3)

                if not smiles:
                    continue

                ik = smiles_to_inchikey14(smiles)
                if ik and ik not in existing_iks:
                    entry['inchikey14'] = ik
                    all_new.append(entry)
                    new_from_pkdb += 1

            stats['sources']['pkdb'] = {
                'total_fetched': len(pkdb_raw),
                'parsed': len(pkdb_parsed),
                'new_drugs': new_from_pkdb,
            }
            print(f"  Fetched: {len(pkdb_raw)}, Parsed: {len(pkdb_parsed)}, New: {new_from_pkdb}")
        except Exception as e:
            print(f"  PK-DB fetch failed: {e}")
            stats['sources']['pkdb'] = {'error': str(e)}
    else:
        print("  Skipped (fetch_external=False or no requests module)")
        stats['sources']['pkdb'] = {'skipped': True}

    # ── Dedup new entries by InChIKey ──
    print("\n" + "="*60)
    print("Deduplication...")
    new_by_ik = defaultdict(list)
    for entry in all_new:
        ik = entry.get('inchikey14', '')
        if ik:
            new_by_ik[ik].append(entry)

    # Merge entries per drug (take best confidence, combine sources)
    merged_new = []
    for ik, entries in new_by_ik.items():
        # Pick best entry (highest confidence, with dose+cmax)
        best = None
        for e in entries:
            has_both = e.get('cmax_ng_ml') is not None and e.get('dose_mg') is not None
            if best is None or (has_both and not (best.get('cmax_ng_ml') and best.get('dose_mg'))):
                best = e

        if best:
            best['all_sources'] = list(set(e.get('source', '') for e in entries))
            best['n_entries'] = len(entries)
            merged_new.append(best)

    stats['total_new_drugs'] = len(merged_new)
    stats['total_new_tuples'] = len(all_new)
    stats['total_enriched'] = len(all_enriched)

    # ── Save outputs ──
    out_dir = Path('data/curated')
    out_dir.mkdir(parents=True, exist_ok=True)

    # New drugs
    out_path = out_dir / 'external_pk_integrated.json'
    with open(out_path, 'w') as f:
        json.dump(merged_new, f, indent=2, default=str)

    # Enrichment data
    enrich_path = out_dir / 'external_pk_enrichment.json'
    with open(enrich_path, 'w') as f:
        json.dump(all_enriched, f, indent=2, default=str)

    # Stats
    stats_path = out_dir / 'external_pk_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"External PK Integration Summary")
    print(f"{'='*60}")
    print(f"New unique drugs:         {stats['total_new_drugs']}")
    print(f"Total new tuples:         {stats['total_new_tuples']}")
    print(f"Enriched existing drugs:  {stats['total_enriched']}")
    print(f"\nPer source:")
    for src, src_stats in stats['sources'].items():
        print(f"  {src}: {json.dumps(src_stats)}")
    print(f"\nOutputs:")
    print(f"  New drugs: {out_path}")
    print(f"  Enrichment: {enrich_path}")
    print(f"  Stats: {stats_path}")

    return stats


# ─── CLI ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='External PK Database Integrator')
    parser.add_argument('--no-fetch', action='store_true',
                       help='Skip external API calls (use local data only)')
    args = parser.parse_args()

    stats = integrate_all_sources(fetch_external=not args.no_fetch)


if __name__ == '__main__':
    main()
