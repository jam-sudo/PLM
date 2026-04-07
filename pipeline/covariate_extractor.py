"""
Covariate Effect Extractor — Extract drug × condition → PK fold-change from FDA reviews.

Extracts quantitative PK changes for special populations:
  - Renal impairment (mild/moderate/severe/ESRD)
  - Hepatic impairment (mild/moderate/severe, Child-Pugh A/B/C)
  - Food effect (fed vs fasted)
  - Age (elderly vs young)
  - Sex (male vs female)
  - Body weight effect
  - Drug-drug interactions (CYP inhibitor/inducer)

Output format per entry:
  {drug, smiles, nda, condition_type, condition_level,
   pk_param, fold_change, direction, confidence, source_text}

Output: data/curated/covariate_effects.json
"""

import re
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


# ─── Condition Categories ────────────────────────────────────────

CONDITION_TYPES = {
    'renal': {
        'keywords': ['renal impairment', 'renal insufficiency', 'kidney',
                     'creatinine clearance', 'CrCl', 'eGFR', 'ESRD',
                     'end.stage renal', 'hemodialysis'],
        'levels': {
            'mild': ['mild renal', 'CrCl 50', 'CrCl 60', 'eGFR 60', 'eGFR 50'],
            'moderate': ['moderate renal', 'CrCl 30', 'eGFR 30'],
            'severe': ['severe renal', 'CrCl 15', 'CrCl <15', 'eGFR 15', 'eGFR <15'],
            'esrd': ['ESRD', 'end.stage', 'hemodialysis', 'dialysis'],
        },
    },
    'hepatic': {
        'keywords': ['hepatic impairment', 'hepatic insufficiency', 'liver',
                     'Child.Pugh', 'cirrhosis', 'cirrhotic'],
        'levels': {
            'mild': ['mild hepatic', 'Child.Pugh A', 'Child-Pugh A'],
            'moderate': ['moderate hepatic', 'Child.Pugh B', 'Child-Pugh B'],
            'severe': ['severe hepatic', 'Child.Pugh C', 'Child-Pugh C'],
        },
    },
    'food': {
        'keywords': ['food effect', 'fed state', 'fasted', 'high.fat',
                     'meal', 'postprandial', 'prandial'],
        'levels': {
            'fed': ['fed', 'with food', 'high.fat', 'meal', 'postprandial'],
            'high_fat': ['high.fat', 'high.calorie', 'HFHC'],
            'low_fat': ['low.fat', 'LFLC', 'light meal'],
        },
    },
    'age': {
        'keywords': ['elderly', 'geriatric', 'older adult', 'aged',
                     'pediatric', 'children', 'adolescent', 'age group'],
        'levels': {
            'elderly': ['elderly', 'geriatric', 'older', '≥65', '>65', 'aged'],
            'pediatric': ['pediatric', 'children', 'adolescent', 'child'],
        },
    },
    'sex': {
        'keywords': ['sex', 'gender', 'male', 'female', 'women', 'men'],
        'levels': {
            'female': ['female', 'women'],
            'male': ['male', 'men'],
        },
    },
}


# ─── Extraction Patterns ────────────────────────────────────────

# Pattern: "Cmax increased by 46%" or "AUC was 46% higher"
PCT_CHANGE_PAT = re.compile(
    r'(Cmax|AUC\w*|clearance|CL|Vd|half.life|t1/2|exposure)\s+'
    r'(?:was\s+|were\s+)?'
    r'(?:approximately\s+|about\s+|~\s*)?'
    r'(increased?|decreased?|reduced?|higher|lower|elevated|diminished)\s+'
    r'(?:by\s+|on average\s+by\s+)?'
    r'(?:approximately\s+|about\s+|~\s*)?'
    r'(\d+(?:\.\d+)?)\s*%',
    re.IGNORECASE
)

# Pattern: "46% increase in Cmax" or "46% higher Cmax"
PCT_BEFORE_PAT = re.compile(
    r'(?:approximately\s+|about\s+|~\s*)?'
    r'(\d+(?:\.\d+)?)\s*%\s*'
    r'(increase|decrease|reduction|higher|lower|elevation|decline)\s+'
    r'(?:in\s+)?'
    r'(Cmax|AUC\w*|clearance|CL|Vd|half.life|t1/2|exposure)',
    re.IGNORECASE
)

# Pattern: "Cmax was 1.8-fold higher" or "2.3-fold increase in AUC"
FOLD_PAT = re.compile(
    r'(?:(Cmax|AUC\w*|clearance|CL|exposure)\s+(?:was\s+)?)?'
    r'(?:approximately\s+|about\s+|~\s*)?'
    r'(\d+(?:\.\d+)?)\s*[-–]?\s*fold\s+'
    r'(higher|lower|increase|decrease|greater|reduction)'
    r'(?:\s+(?:in\s+)?(Cmax|AUC\w*|clearance|CL|exposure))?',
    re.IGNORECASE
)

# Pattern: "AUC ratio of 1.45" or "Cmax ratio was 0.72"
RATIO_PAT = re.compile(
    r'(Cmax|AUC\w*|CL|clearance|exposure)\s+'
    r'(?:geometric mean\s+)?ratio\s+'
    r'(?:of\s+|was\s+|=\s*)?'
    r'(?:approximately\s+|about\s+)?'
    r'(\d+(?:\.\d+)?)',
    re.IGNORECASE
)


def parse_direction(word: str) -> str:
    """Map direction word to increase/decrease."""
    word = word.lower()
    if word in ('increased', 'increase', 'higher', 'elevated', 'elevation', 'greater'):
        return 'increase'
    elif word in ('decreased', 'decrease', 'lower', 'reduced', 'reduction',
                  'diminished', 'decline'):
        return 'decrease'
    return 'unknown'


def pct_to_fold(pct: float, direction: str) -> float:
    """Convert percentage change to fold-change relative to reference."""
    if direction == 'increase':
        return 1.0 + pct / 100.0
    elif direction == 'decrease':
        return 1.0 - pct / 100.0
    return 1.0


def extract_effects_from_text(text: str, nda: str) -> List[Dict]:
    """Extract all covariate effects from a text block."""
    effects = []

    # Find condition context for each sentence
    sentences = re.split(r'[.;]\s+', text)

    for sent in sentences:
        # Determine which condition types are mentioned
        matched_conditions = []
        for ctype, cinfo in CONDITION_TYPES.items():
            for kw in cinfo['keywords']:
                if re.search(kw, sent, re.IGNORECASE):
                    # Determine level
                    level = 'unspecified'
                    for lname, lkws in cinfo['levels'].items():
                        for lkw in lkws:
                            if re.search(lkw, sent, re.IGNORECASE):
                                level = lname
                                break
                        if level != 'unspecified':
                            break
                    matched_conditions.append((ctype, level))
                    break

        if not matched_conditions:
            continue

        # Extract quantitative PK changes from this sentence
        pk_changes = []

        # Pattern 1: "Cmax increased by 46%"
        for m in PCT_CHANGE_PAT.finditer(sent):
            pk_param = m.group(1)
            direction = parse_direction(m.group(2))
            pct = float(m.group(3))
            fc = pct_to_fold(pct, direction)
            if 0.01 < fc < 20:  # sanity
                pk_changes.append({
                    'pk_param': pk_param,
                    'direction': direction,
                    'fold_change': round(fc, 3),
                    'raw_pct': pct,
                    'pattern': 'pct_after',
                })

        # Pattern 2: "46% increase in Cmax"
        for m in PCT_BEFORE_PAT.finditer(sent):
            pct = float(m.group(1))
            direction = parse_direction(m.group(2))
            pk_param = m.group(3)
            fc = pct_to_fold(pct, direction)
            if 0.01 < fc < 20:
                pk_changes.append({
                    'pk_param': pk_param,
                    'direction': direction,
                    'fold_change': round(fc, 3),
                    'raw_pct': pct,
                    'pattern': 'pct_before',
                })

        # Pattern 3: "1.8-fold higher"
        for m in FOLD_PAT.finditer(sent):
            pk_param = m.group(1) or m.group(4) or 'exposure'
            fold_val = float(m.group(2))
            direction = parse_direction(m.group(3))
            if direction == 'increase':
                fc = fold_val
            elif direction == 'decrease':
                fc = 1.0 / fold_val if fold_val > 0 else 1.0
            else:
                fc = fold_val
            if 0.01 < fc < 20:
                pk_changes.append({
                    'pk_param': pk_param,
                    'direction': direction,
                    'fold_change': round(fc, 3),
                    'raw_fold': fold_val,
                    'pattern': 'fold',
                })

        # Pattern 4: "Cmax ratio of 1.45"
        for m in RATIO_PAT.finditer(sent):
            pk_param = m.group(1)
            ratio = float(m.group(2))
            direction = 'increase' if ratio > 1.0 else 'decrease' if ratio < 1.0 else 'none'
            if 0.01 < ratio < 20:
                pk_changes.append({
                    'pk_param': pk_param,
                    'direction': direction,
                    'fold_change': round(ratio, 3),
                    'raw_ratio': ratio,
                    'pattern': 'ratio',
                })

        # Pair conditions with PK changes
        for ctype, level in matched_conditions:
            for pk in pk_changes:
                effects.append({
                    'nda': nda,
                    'condition_type': ctype,
                    'condition_level': level,
                    'pk_param': pk['pk_param'],
                    'fold_change': pk['fold_change'],
                    'direction': pk['direction'],
                    'pattern': pk['pattern'],
                    'confidence': 'high' if pk['pattern'] in ('ratio', 'fold') else 'medium',
                    'source_sentence': sent.strip()[:200],
                })

    return effects


# ─── Main Pipeline ───────────────────────────────────────────────

def main():
    text_dir = Path('data/llm_extracted/text')
    text_files = sorted(text_dir.glob('*.txt'))
    print(f"Scanning {len(text_files)} NDA texts for covariate effects...")

    # Load SMILES mapping
    smiles_map = {}
    for map_file in ['data/llm_extracted/drug_smiles_map.json',
                     'data/raw/nda_drug_smiles_map.json']:
        if Path(map_file).exists():
            with open(map_file) as f:
                smiles_map.update(json.load(f))

    # Load LLM extractions for drug name lookup
    llm_drug_map = {}
    llm_path = Path('data/llm_extracted/pk_llm_merged.json')
    if llm_path.exists():
        with open(llm_path) as f:
            for entry in json.load(f):
                nda = entry.get('nda', '')
                if nda and entry.get('drug_name'):
                    llm_drug_map[nda] = {
                        'drug_name': entry['drug_name'],
                        'smiles': entry.get('smiles', ''),
                    }

    all_effects = []
    ndas_with_effects = 0

    for tf in text_files:
        nda = tf.stem
        text = tf.read_text(errors='replace')
        effects = extract_effects_from_text(text, nda)

        if effects:
            ndas_with_effects += 1
            # Enrich with drug info
            drug_info = llm_drug_map.get(nda, {})
            smiles = drug_info.get('smiles', '') or smiles_map.get(nda, '')
            drug_name = drug_info.get('drug_name', '')

            for e in effects:
                e['drug_name'] = drug_name
                e['smiles'] = smiles
            all_effects.extend(effects)

    # Deduplicate exact matches
    seen = set()
    deduped = []
    for e in all_effects:
        key = (e['nda'], e['condition_type'], e['condition_level'],
               e['pk_param'], e['fold_change'])
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    # Stats
    from collections import Counter
    ctype_counts = Counter(e['condition_type'] for e in deduped)
    param_counts = Counter(e['pk_param'].lower() for e in deduped)
    with_smiles = sum(1 for e in deduped if e.get('smiles'))
    unique_drugs = len(set(e['nda'] for e in deduped))

    # Filter to Cmax and AUC only (most useful for our model)
    pk_relevant = [e for e in deduped if any(
        k in e['pk_param'].lower() for k in ['cmax', 'auc', 'exposure']
    )]

    # Save
    out_path = Path('data/curated/covariate_effects.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(deduped, f, indent=2)

    # Save Cmax/AUC-only subset
    pk_path = Path('data/curated/covariate_effects_pk.json')
    with open(pk_path, 'w') as f:
        json.dump(pk_relevant, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Covariate Effect Extraction Summary")
    print(f"{'='*60}")
    print(f"NDAs scanned:          {len(text_files)}")
    print(f"NDAs with effects:     {ndas_with_effects}")
    print(f"Total effects (raw):   {len(all_effects)}")
    print(f"After dedup:           {len(deduped)}")
    print(f"Cmax/AUC relevant:     {len(pk_relevant)}")
    print(f"Unique drugs:          {unique_drugs}")
    print(f"With SMILES:           {with_smiles}")
    print(f"\nBy condition type:")
    for ct, cnt in sorted(ctype_counts.items(), key=lambda x: -x[1]):
        print(f"  {ct:15s}: {cnt}")
    print(f"\nBy PK parameter:")
    for pp, cnt in sorted(param_counts.items(), key=lambda x: -x[1]):
        print(f"  {pp:15s}: {cnt}")
    print(f"\nOutput: {out_path}")
    print(f"PK-only: {pk_path}")

    # Show sample entries
    print(f"\nSample entries:")
    for e in pk_relevant[:10]:
        print(f"  {e['drug_name'][:20]:20s} {e['condition_type']:10s} "
              f"{e['condition_level']:12s} {e['pk_param']:8s} "
              f"FC={e['fold_change']:.2f} ({e['direction']})")


if __name__ == '__main__':
    main()
