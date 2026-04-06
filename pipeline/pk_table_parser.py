"""
FDA PK Table Text Parser — Direct structured PK parameter extraction from PDF text.

Bypasses figure digitization entirely. Parses PK parameter tables from
already-extracted text files (data/llm_extracted/text/*.txt).

Strategy:
  1. Regex-based table detection (Cmax, AUC, Tmax rows with numeric values)
  2. Unit-aware value extraction (ng/mL, µg/mL, mg/L → canonical ng/mL)
  3. Dose context extraction from surrounding text
  4. Cross-validation against existing LLM extractions

Output: data/curated/pk_table_parsed.json
"""

import re
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


# ─── Unit conversion factors to ng/mL ───────────────────────────
CONC_UNIT_FACTORS = {
    'ng/ml':   1.0,
    'ng/mL':   1.0,
    'mcg/ml':  1000.0,
    'mcg/mL':  1000.0,
    'µg/ml':   1000.0,
    'µg/mL':   1000.0,
    'ug/ml':   1000.0,
    'ug/mL':   1000.0,
    'mg/l':    1000.0,
    'mg/L':    1000.0,
    'µg/l':    1.0,
    'µg/L':    1.0,
    'mcg/l':   1.0,
    'mcg/L':   1.0,
    'pg/ml':   0.001,
    'pg/mL':   0.001,
    'nm':      None,  # needs MW
    'nM':      None,
    'µm':      None,
    'µM':      None,
}

AUC_UNIT_FACTORS = {
    'ng·h/ml':     1.0,
    'ng*h/ml':     1.0,
    'ng.h/ml':     1.0,
    'ng·h/mL':     1.0,
    'ng*h/mL':     1.0,
    'ng.h/mL':     1.0,
    'h·ng/ml':     1.0,
    'h*ng/ml':     1.0,
    'h·ng/mL':     1.0,
    'h*ng/mL':     1.0,
    'mcg·h/ml':    1000.0,
    'mcg*h/ml':    1000.0,
    'µg·h/ml':     1000.0,
    'µg*h/ml':     1000.0,
    'mcg·h/mL':    1000.0,
    'mcg*h/mL':    1000.0,
    'µg·h/mL':     1000.0,
    'µg*h/mL':     1000.0,
    'mg·h/l':      1000.0,
    'mg*h/l':      1000.0,
    'mg·h/L':      1000.0,
    'mg*h/L':      1000.0,
    'h·µg/ml':     1000.0,
    'h*µg/ml':     1000.0,
    'h·mcg/ml':    1000.0,
    'h·µg/mL':     1000.0,
    'h*µg/mL':     1000.0,
    'pg·h/ml':     0.001,
    'pg*h/ml':     0.001,
    'pg·h/mL':     0.001,
}


# ─── Regex patterns ─────────────────────────────────────────────

# Match numeric values (including scientific notation, ranges with ±/SD)
NUM_PAT = re.compile(
    r'(?<![a-zA-Z])(\d+[,.]?\d*(?:\s*[×x]\s*10\s*[-–]?\s*\d+)?)\s*'
    r'(?:\([\d.,\s]+\))?'  # optional (SD) or (CV%)
)

# Cmax line patterns
CMAX_PAT = re.compile(
    r'[Cc]\s*(?:max|MAX)\s*[\s,:(]*'
    r'(\w+[/·*.]?\w*)\s*\)?\s*'  # unit
    r'[\s:]*'
    r'([\d.,]+(?:\s*[±(][\d.,\s%]+[)]?)?)',  # value(s)
    re.IGNORECASE
)

# More flexible Cmax pattern for table rows
CMAX_TABLE_PAT = re.compile(
    r'(?:C\s*max|Cmax)\s*'
    r'(?:,?\s*(?:mean|median|geometric)?\s*)?'
    r'[\s,]*\(?\s*'
    r'((?:ng|mcg|µg|ug|mg|pg)\s*[/·]\s*(?:mL|ml|L|l))\s*\)?\s*'
    r'[\s:]*'
    r'([\d.,]+(?:\s*[±(][\d.,\s%]+[)]?)?)',
    re.IGNORECASE
)

# AUC patterns
AUC_PAT = re.compile(
    r'AUC\s*(?:0?-?∞|inf|0-inf|last|0-t|0-24|0-τ|tau|0-72|0-48)?\s*'
    r'(?:,?\s*(?:mean|median|geometric)?\s*)?'
    r'[\s,]*\(?\s*'
    r'((?:ng|mcg|µg|ug|mg|pg|h)\s*[·*.]?\s*(?:h|hr)?\s*[/·*.]?\s*(?:mL|ml|L|l|ng|mcg|µg)(?:\s*[·*.]?\s*(?:h|hr|mL|ml))?)\s*\)?\s*'
    r'[\s:]*'
    r'([\d.,]+(?:\s*[±(][\d.,\s%]+[)]?)?)',
    re.IGNORECASE
)

# Tmax pattern
TMAX_PAT = re.compile(
    r'[Tt]\s*(?:max|MAX)\s*'
    r'[\s,]*\(?\s*(h(?:r|ours?)?|min)\s*\)?\s*'
    r'[\s:]*'
    r'([\d.,]+(?:\s*[±(][\d.,\s%]+[)]?)?)',
    re.IGNORECASE
)

# t1/2 pattern
THALF_PAT = re.compile(
    r'(?:t\s*½|t\s*1\s*/?\s*2|half[- ]?life)\s*'
    r'[\s,]*\(?\s*(h(?:r|ours?)?|min)\s*\)?\s*'
    r'[\s:]*'
    r'([\d.,]+(?:\s*[±(][\d.,\s%]+[)]?)?)',
    re.IGNORECASE
)

# Dose context
DOSE_PAT = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*(?:mg|MG)\s*'
    r'(?:(?:single|oral|po|once|daily|bid|tid|qd|q\.?d\.?|BID|QD|TID)\b)?',
    re.IGNORECASE
)

# Food context
FOOD_PAT = re.compile(
    r'\b(fasted?|fed|high[- ]fat|low[- ]fat|postprandial|fasting)\b',
    re.IGNORECASE
)

# Table header detection (for structured tables)
TABLE_HEADER_PAT = re.compile(
    r'(?:Parameter|PK\s*Parameter|Pharmacokinetic).*?(?:Cmax|AUC|Tmax)',
    re.IGNORECASE | re.DOTALL
)

# Page header pattern
PAGE_PAT = re.compile(r'={5}\s*PAGE\s+(\d+)\s*={5}')


def clean_number(s: str) -> Optional[float]:
    """Extract a clean float from potentially messy text."""
    if not s:
        return None
    # Remove parenthetical info (SD, CV%)
    s = re.sub(r'\(.*?\)', '', s).strip()
    # Remove ± and everything after
    s = re.sub(r'[±]\s*[\d.,]+', '', s).strip()
    # Remove commas used as thousands separator
    s = s.replace(',', '')
    # Handle scientific notation
    sci = re.match(r'([\d.]+)\s*[×x]\s*10\s*[-–]?\s*(\d+)', s)
    if sci:
        return float(sci.group(1)) * (10 ** int(sci.group(2)))
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def find_unit_factor(unit_str: str, unit_dict: dict) -> Optional[float]:
    """Match a unit string to conversion factor."""
    unit_clean = unit_str.strip()
    # Direct match
    if unit_clean in unit_dict:
        return unit_dict[unit_clean]
    # Case-insensitive match
    for k, v in unit_dict.items():
        if k.lower() == unit_clean.lower():
            return v
    # Fuzzy match (remove spaces)
    unit_nospace = re.sub(r'\s+', '', unit_clean)
    for k, v in unit_dict.items():
        if re.sub(r'\s+', '', k) == unit_nospace:
            return v
    return None


def extract_pk_from_page(text: str, page_num: int) -> List[Dict]:
    """Extract PK parameters from a single page of text."""
    results = []

    # Find dose context for this page
    doses = [clean_number(m.group(1)) for m in DOSE_PAT.finditer(text)]
    doses = [d for d in doses if d and 0.01 <= d <= 50000]

    # Food context
    food_matches = FOOD_PAT.findall(text)
    food = 'fasted' if any('fast' in f.lower() for f in food_matches) else \
           'fed' if any('fed' in f.lower() or 'fat' in f.lower() for f in food_matches) else \
           'not_specified'

    # Extract Cmax values
    cmax_entries = []
    for pat in [CMAX_TABLE_PAT, CMAX_PAT]:
        for m in pat.finditer(text):
            unit_str = m.group(1)
            val_str = m.group(2)
            factor = find_unit_factor(unit_str, CONC_UNIT_FACTORS)
            val = clean_number(val_str)
            if val is not None and factor is not None and val > 0:
                cmax_entries.append({
                    'cmax_ng_ml': val * factor,
                    'unit_raw': unit_str,
                    'val_raw': val_str,
                })

    # Extract AUC values
    auc_entries = []
    for m in AUC_PAT.finditer(text):
        unit_str = m.group(1)
        val_str = m.group(2)
        factor = find_unit_factor(unit_str, AUC_UNIT_FACTORS)
        val = clean_number(val_str)
        if val is not None and factor is not None and val > 0:
            auc_entries.append({
                'auc_ng_h_ml': val * factor,
                'unit_raw': unit_str,
                'val_raw': val_str,
            })

    # Extract Tmax
    tmax_val = None
    for m in TMAX_PAT.finditer(text):
        unit = m.group(1).lower()
        val = clean_number(m.group(2))
        if val is not None:
            if 'min' in unit:
                val /= 60.0
            tmax_val = val
            break

    # Extract t1/2
    thalf_val = None
    for m in THALF_PAT.finditer(text):
        unit = m.group(1).lower()
        val = clean_number(m.group(2))
        if val is not None:
            if 'min' in unit:
                val /= 60.0
            thalf_val = val
            break

    # Build tuples: pair Cmax with dose context
    if cmax_entries:
        # If we have equal number of doses and Cmax values, pair them
        if len(doses) == len(cmax_entries) and len(doses) > 0:
            for dose, cmax in zip(doses, cmax_entries):
                entry = {
                    'dose_mg': dose,
                    'cmax_ng_ml': cmax['cmax_ng_ml'],
                    'tmax_h': tmax_val,
                    't_half_h': thalf_val,
                    'food': food,
                    'source_page': page_num,
                    'confidence': 'medium',
                    'method': 'regex_table',
                }
                if auc_entries:
                    entry['auc_ng_h_ml'] = auc_entries[0]['auc_ng_h_ml']
                results.append(entry)
        else:
            # Can't pair doses → emit each Cmax with best-guess dose
            best_dose = doses[0] if doses else None
            for cmax in cmax_entries:
                entry = {
                    'dose_mg': best_dose,
                    'cmax_ng_ml': cmax['cmax_ng_ml'],
                    'tmax_h': tmax_val,
                    't_half_h': thalf_val,
                    'food': food,
                    'source_page': page_num,
                    'confidence': 'low' if best_dose is None else 'medium',
                    'method': 'regex_table',
                }
                if auc_entries:
                    entry['auc_ng_h_ml'] = auc_entries[0]['auc_ng_h_ml']
                results.append(entry)
    elif auc_entries:
        # AUC without Cmax
        best_dose = doses[0] if doses else None
        for auc in auc_entries:
            results.append({
                'dose_mg': best_dose,
                'cmax_ng_ml': None,
                'auc_ng_h_ml': auc['auc_ng_h_ml'],
                'tmax_h': tmax_val,
                't_half_h': thalf_val,
                'food': food,
                'source_page': page_num,
                'confidence': 'low',
                'method': 'regex_table',
            })

    return results


def parse_nda_text(text_path: Path) -> Dict:
    """Parse a full NDA text file into structured PK tuples."""
    nda = text_path.stem
    text = text_path.read_text(errors='replace')

    # Split by page markers
    pages = PAGE_PAT.split(text)
    all_tuples = []

    # pages alternates: [text_before, page_num, text, page_num, text, ...]
    if len(pages) > 1:
        for i in range(1, len(pages), 2):
            page_num = int(pages[i])
            page_text = pages[i + 1] if i + 1 < len(pages) else ''
            tuples = extract_pk_from_page(page_text, page_num)
            all_tuples.extend(tuples)
    else:
        # No page markers, parse entire text
        tuples = extract_pk_from_page(text, 0)
        all_tuples.extend(tuples)

    return {
        'nda': nda,
        'n_pages_parsed': len(pages) // 2,
        'pk_tuples': all_tuples,
    }


def validate_tuple(t: Dict) -> Tuple[bool, str]:
    """Physical plausibility checks."""
    cmax = t.get('cmax_ng_ml')
    if cmax is not None:
        if cmax <= 0 or cmax > 1e9:
            return False, f'invalid_cmax({cmax})'
    auc = t.get('auc_ng_h_ml')
    if auc is not None:
        if auc <= 0 or auc > 1e10:
            return False, f'invalid_auc({auc})'
    dose = t.get('dose_mg')
    if dose is not None and (dose <= 0 or dose > 50000):
        return False, f'invalid_dose({dose})'
    tmax = t.get('tmax_h')
    if tmax is not None and (tmax <= 0 or tmax > 72):
        return False, f'invalid_tmax({tmax})'
    thalf = t.get('t_half_h')
    if thalf is not None and (thalf <= 0 or thalf > 1000):
        return False, f'invalid_thalf({thalf})'
    if cmax is None and auc is None:
        return False, 'no_concentration'
    return True, 'ok'


def main():
    text_dir = Path('data/llm_extracted/text')
    if not text_dir.exists():
        print(f"Error: {text_dir} not found")
        return

    text_files = sorted(text_dir.glob('*.txt'))
    print(f"Found {len(text_files)} text files to parse")

    # Load existing LLM extractions for comparison
    llm_merged_path = Path('data/llm_extracted/pk_llm_merged.json')
    llm_by_nda = defaultdict(list)
    if llm_merged_path.exists():
        with open(llm_merged_path) as f:
            llm_data = json.load(f)
        for entry in llm_data:
            llm_by_nda[entry.get('nda', '')].append(entry)
        print(f"Loaded {len(llm_data)} existing LLM extractions for comparison")

    # Load SMILES mappings
    smiles_map = {}
    for map_file in ['data/llm_extracted/drug_smiles_map.json', 'data/raw/nda_drug_smiles_map.json']:
        if Path(map_file).exists():
            with open(map_file) as f:
                smiles_map.update(json.load(f))

    all_results = []
    stats = {
        'total_ndas': 0,
        'ndas_with_data': 0,
        'total_tuples': 0,
        'valid_tuples': 0,
        'with_cmax': 0,
        'with_dose': 0,
        'with_dose_and_cmax': 0,
        'invalid_reasons': defaultdict(int),
        'new_vs_llm': {'new_ndas': 0, 'overlap': 0, 'new_tuples': 0},
    }

    for tf in text_files:
        result = parse_nda_text(tf)
        stats['total_ndas'] += 1

        if not result['pk_tuples']:
            continue

        stats['ndas_with_data'] += 1
        nda = result['nda']
        has_llm = nda in llm_by_nda

        if has_llm:
            stats['new_vs_llm']['overlap'] += 1
        else:
            stats['new_vs_llm']['new_ndas'] += 1

        for t in result['pk_tuples']:
            stats['total_tuples'] += 1
            valid, reason = validate_tuple(t)
            if not valid:
                stats['invalid_reasons'][reason] += 1
                continue

            stats['valid_tuples'] += 1
            if t.get('cmax_ng_ml') is not None:
                stats['with_cmax'] += 1
            if t.get('dose_mg') is not None:
                stats['with_dose'] += 1
            if t.get('cmax_ng_ml') is not None and t.get('dose_mg') is not None:
                stats['with_dose_and_cmax'] += 1

            # Attempt SMILES lookup
            drug_name = None
            smiles = None
            llm_entries = llm_by_nda.get(nda, [])
            if llm_entries:
                drug_name = llm_entries[0].get('drug_name')
                smiles = llm_entries[0].get('smiles')
            if not smiles and nda in smiles_map:
                smiles = smiles_map[nda]

            t['nda'] = nda
            t['drug_name'] = drug_name
            t['smiles'] = smiles
            t['source'] = 'regex_table_parser'
            all_results.append(t)

            if not has_llm:
                stats['new_vs_llm']['new_tuples'] += 1

    # Save results
    out_path = Path('data/curated/pk_table_parsed.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save stats
    stats_path = Path('data/curated/pk_table_parsed_stats.json')
    stats['invalid_reasons'] = dict(stats['invalid_reasons'])
    stats['new_vs_llm'] = dict(stats['new_vs_llm'])
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"FDA PK Table Parser Results")
    print(f"{'='*60}")
    print(f"NDAs processed:       {stats['total_ndas']}")
    print(f"NDAs with PK data:    {stats['ndas_with_data']}")
    print(f"Total tuples found:   {stats['total_tuples']}")
    print(f"Valid tuples:         {stats['valid_tuples']}")
    print(f"With Cmax:            {stats['with_cmax']}")
    print(f"With dose:            {stats['with_dose']}")
    print(f"With dose + Cmax:     {stats['with_dose_and_cmax']}")
    print(f"\nNew vs LLM extraction:")
    print(f"  Overlapping NDAs:   {stats['new_vs_llm']['overlap']}")
    print(f"  New NDAs:           {stats['new_vs_llm']['new_ndas']}")
    print(f"  New tuples:         {stats['new_vs_llm']['new_tuples']}")
    if stats['invalid_reasons']:
        print(f"\nInvalid reasons:")
        for reason, count in sorted(stats['invalid_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"\nOutput: {out_path}")


if __name__ == '__main__':
    main()
