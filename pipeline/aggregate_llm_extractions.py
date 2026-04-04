"""
Aggregate LLM extractions, validate, and compare with regex baseline.

Steps:
1. Load all per-NDA JSON files from data/llm_extracted/json/
2. Validate each PK tuple (physical plausibility, unit sanity)
3. Lookup SMILES from NDA→drug→SMILES mapping
4. Compare yield vs. regex-extracted pk_table_v2.json
5. Output merged dataset: data/llm_extracted/pk_llm_merged.json
"""

import json
from pathlib import Path
from collections import defaultdict


def validate_tuple(t):
    """Physical plausibility checks. Returns (valid, reason)."""
    dose = t.get('dose_mg')
    if not isinstance(dose, (int, float)) or dose <= 0 or dose > 50000:
        return False, f'invalid_dose({dose})'
    if not any(t.get(k) for k in ['cmax_ng_ml', 'auc_inf_ng_h_ml', 'auc_last_ng_h_ml']):
        return False, 'no_concentration'
    cmax = t.get('cmax_ng_ml')
    if cmax is not None:
        if cmax <= 0 or cmax > 1e9:
            return False, f'invalid_cmax({cmax})'
    tmax = t.get('tmax_h')
    if tmax is not None and (tmax <= 0 or tmax > 72):
        return False, f'invalid_tmax({tmax})'
    thalf = t.get('t_half_h')
    if thalf is not None and (thalf <= 0 or thalf > 1000):
        return False, f'invalid_thalf({thalf})'
    for auc_key in ['auc_inf_ng_h_ml', 'auc_last_ng_h_ml']:
        a = t.get(auc_key)
        if a is not None and (a <= 0 or a > 1e10):
            return False, f'invalid_{auc_key}({a})'
    return True, 'ok'


def main():
    # Load extraction results
    json_dir = Path('data/llm_extracted/json')
    extractions = []
    for jf in sorted(json_dir.glob('NDA*.json')):
        try:
            with open(jf) as f:
                data = json.load(f)
            extractions.append(data)
        except Exception as e:
            print(f"  Error loading {jf.name}: {e}")

    print(f"Loaded {len(extractions)} extractions")

    # Load NDA→drug→SMILES map
    with open('data/raw/nda_drug_smiles_map.json') as f:
        nda_map = json.load(f)

    # Load drug name → SMILES fallback map
    try:
        with open('data/llm_extracted/drug_smiles_map.json') as f:
            drug_smiles = json.load(f)
    except FileNotFoundError:
        drug_smiles = {}

    # Load existing regex-based PK table for comparison
    with open('data/curated/pk_table_v2.json') as f:
        regex_pk = json.load(f)
    regex_by_nda = defaultdict(list)
    for e in regex_pk:
        nda_key = e.get('nda', '').replace('NDA', '')
        regex_by_nda[nda_key].append(e)

    # Process each extraction
    all_tuples = []
    stats = {
        'total_extracted': 0,
        'total_valid': 0,
        'invalid_reasons': defaultdict(int),
        'with_smiles': 0,
        'by_drug': defaultdict(int),
    }

    for ext in extractions:
        nda = ext.get('nda', '').replace('NDA', '')
        drug_name = ext.get('drug_name', '')
        tuples = ext.get('pk_tuples', [])

        # SMILES lookup
        smiles = None
        if nda in nda_map:
            smiles = nda_map[nda].get('smiles')
        if not smiles:
            # Try by drug name in NDA map
            for k, v in nda_map.items():
                if v.get('drug', '').lower() == drug_name.lower():
                    smiles = v.get('smiles')
                    break
        if not smiles and drug_name in drug_smiles:
            smiles = drug_smiles[drug_name]

        stats['total_extracted'] += len(tuples)
        if smiles:
            stats['with_smiles'] += 1

        for t in tuples:
            valid, reason = validate_tuple(t)
            if not valid:
                stats['invalid_reasons'][reason] += 1
                continue
            stats['total_valid'] += 1
            stats['by_drug'][drug_name] += 1

            merged = {
                'nda': f'NDA{nda}',
                'drug_name': drug_name,
                'smiles': smiles,
                **t,
            }
            all_tuples.append(merged)

    # Summary
    print(f"\n=== Extraction Summary ===")
    print(f"Total tuples extracted:  {stats['total_extracted']}")
    print(f"Valid tuples:            {stats['total_valid']}")
    print(f"Extractions with SMILES: {stats['with_smiles']}/{len(extractions)}")
    print(f"Unique drugs:            {len(stats['by_drug'])}")

    if stats['invalid_reasons']:
        print(f"\nInvalid tuple reasons:")
        for reason, count in sorted(stats['invalid_reasons'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Per-drug counts
    print(f"\nTuples per drug:")
    for drug, count in sorted(stats['by_drug'].items(), key=lambda x: -x[1]):
        print(f"  {drug}: {count}")

    # Comparison vs regex
    print(f"\n=== vs. Regex-based extraction ===")
    for ext in extractions:
        nda_key = ext.get('nda', '').replace('NDA', '')
        llm_count = sum(1 for t in ext.get('pk_tuples', []) if validate_tuple(t)[0])
        regex_count = len(regex_by_nda.get(nda_key, []))
        regex_valid = sum(1 for e in regex_by_nda.get(nda_key, []) if e.get('dose_mg'))
        print(f"  NDA{nda_key} ({ext.get('drug_name', '?')}): LLM={llm_count}, regex={regex_count} (dose_ok={regex_valid})")

    # Save merged output
    out = Path('data/llm_extracted/pk_llm_merged.json')
    with open(out, 'w') as f:
        json.dump(all_tuples, f, indent=2)
    print(f"\n→ Saved {len(all_tuples)} tuples to {out}")

    # Save stats
    stats_out = Path('data/llm_extracted/extraction_stats.json')
    stats_serializable = {
        'total_extracted': stats['total_extracted'],
        'total_valid': stats['total_valid'],
        'invalid_reasons': dict(stats['invalid_reasons']),
        'with_smiles': stats['with_smiles'],
        'n_extractions': len(extractions),
        'unique_drugs': len(stats['by_drug']),
        'tuples_per_drug': dict(stats['by_drug']),
    }
    with open(stats_out, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"→ Saved stats to {stats_out}")


if __name__ == '__main__':
    main()
