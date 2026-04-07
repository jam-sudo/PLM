"""
Build PLM Dataset v12 — Integrated multi-source PK dataset.

Sources:
  1. v10 (PLM_FDA + Sisyphus_MMPK): 3,490 tuples, 1,289 drugs
  2. FDA table parser (regex): 991 tuples with dose+cmax+smiles
  3. LLM extraction (pk_llm_merged): 1,184 tuples with smiles
  4. ChEMBL expansion: ~8,000 tuples with dose+cmax
  5. External integrated (new drugs): ~8,018 entries

Dedup by InChIKey14 + dose_mg (within 10% tolerance).
Quality tiers: T1 (Sisyphus curated), T2 (FDA LLM high-conf), T3 (FDA regex), T4 (ChEMBL).

Also runs multi-source cross-validation: compares Cmax values for same
drug-dose pairs across sources.

Output:
  data/curated/plm_dataset_v12.json         — Full integrated dataset
  data/curated/v12_build_stats.json          — Build statistics
  data/curated/cross_source_validation.json  — Multi-source Cmax comparison
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import inchi
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ─── Helpers ─────────────────────────────────────────────────────

def smiles_to_ik14(smiles: str) -> Optional[str]:
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


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not HAS_RDKIT or not smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol)


def dose_match(d1: float, d2: float, tol: float = 0.10) -> bool:
    """Check if two doses match within tolerance."""
    if d1 is None or d2 is None or d1 <= 0 or d2 <= 0:
        return False
    ratio = max(d1, d2) / min(d1, d2)
    return ratio <= (1.0 + tol)


def fold_error(pred: float, obs: float) -> float:
    """Compute fold error."""
    if pred <= 0 or obs <= 0:
        return float('inf')
    return max(pred / obs, obs / pred)


def aafe(errors: list) -> float:
    """Average Absolute Fold Error (geometric mean of fold errors)."""
    if not errors:
        return float('inf')
    log_errors = [math.log10(e) for e in errors if e > 0 and math.isfinite(e)]
    if not log_errors:
        return float('inf')
    return 10 ** (sum(abs(le) for le in log_errors) / len(log_errors))


# ─── Data Loaders ────────────────────────────────────────────────

def load_v10() -> List[Dict]:
    """Load v10 dataset (PLM + Sisyphus combined)."""
    path = Path('data/curated/plm_sisyphus_combined.json')
    with open(path) as f:
        data = json.load(f)
    entries = []
    for e in data:
        entries.append({
            'smiles': e.get('smiles', ''),
            'dose_mg': e.get('dose_mg'),
            'cmax_ngml': e.get('cmax_ngml'),
            'drug': e.get('drug', ''),
            'inchikey14': e.get('inchikey14', ''),
            'source': e.get('source', 'v10'),
            'tier': 'T1' if 'Sisyphus' in e.get('source', '') else 'T2',
        })
    return entries


def load_fda_table_parsed() -> List[Dict]:
    """Load FDA PK table parser results."""
    path = Path('data/curated/pk_table_parsed.json')
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    entries = []
    for e in data:
        smiles = e.get('smiles')
        cmax = e.get('cmax_ng_ml')
        dose = e.get('dose_mg')
        if not smiles or not cmax or not dose:
            continue
        ik = e.get('inchikey14') or smiles_to_ik14(smiles)
        entries.append({
            'smiles': smiles,
            'dose_mg': dose,
            'cmax_ngml': cmax,
            'drug': e.get('drug_name', ''),
            'inchikey14': ik or '',
            'source': 'FDA_regex',
            'tier': 'T3',
            'confidence': e.get('confidence', 'medium'),
            'nda': e.get('nda', ''),
            'route': e.get('route', ''),
            'formulation': e.get('formulation', ''),
            'food': e.get('food', ''),
            'population': e.get('population', ''),
            'dose_schedule': e.get('dose_schedule', ''),
            'auc_ng_h_ml': e.get('auc_ng_h_ml'),
            'tmax_h': e.get('tmax_h'),
            't_half_h': e.get('t_half_h'),
        })
    return entries


def load_llm_extraction() -> List[Dict]:
    """Load LLM extraction data."""
    path = Path('data/llm_extracted/pk_llm_merged.json')
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    entries = []
    for e in data:
        smiles = e.get('smiles')
        cmax = e.get('cmax_ng_ml')
        dose = e.get('dose_mg')
        if not smiles or not cmax or not dose:
            continue
        ik = smiles_to_ik14(smiles)
        entries.append({
            'smiles': smiles,
            'dose_mg': dose,
            'cmax_ngml': cmax,
            'drug': e.get('drug_name', ''),
            'inchikey14': ik or '',
            'source': 'FDA_LLM',
            'tier': 'T2',
            'confidence': e.get('confidence', 'medium'),
            'nda': e.get('nda', ''),
            'route': e.get('route', ''),
            'formulation': e.get('formulation', ''),
            'food': e.get('food', ''),
            'population': e.get('population', ''),
            'dose_schedule': e.get('dose_schedule', ''),
            'auc_ng_h_ml': e.get('auc_inf_ng_h_ml') or e.get('auc_last_ng_h_ml'),
            'tmax_h': e.get('tmax_h'),
            't_half_h': e.get('t_half_h'),
        })
    return entries


def load_external_integrated() -> List[Dict]:
    """Load external integrated data (ChEMBL, PK-DB, etc.)."""
    path = Path('data/curated/external_pk_integrated.json')
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    entries = []
    for e in data:
        smiles = e.get('smiles', '')
        cmax = e.get('cmax_ng_ml') or e.get('cmax_ngml')
        dose = e.get('dose_mg')
        if not smiles or not cmax or not dose:
            continue
        ik = e.get('inchikey14') or smiles_to_ik14(smiles)
        if not ik:
            continue
        src = e.get('source', '')
        if 'ChEMBL' in src:
            tier = 'T4'
        elif 'FDA' in src:
            tier = 'T3'
        else:
            tier = 'T4'
        entries.append({
            'smiles': smiles,
            'dose_mg': dose,
            'cmax_ngml': float(cmax),
            'drug': e.get('drug_name', ''),
            'inchikey14': ik,
            'source': src,
            'tier': tier,
            'route': e.get('route', ''),
            'formulation': e.get('formulation', ''),
            'food': e.get('food', ''),
            'population': e.get('population', ''),
            'dose_schedule': e.get('dose_schedule', ''),
        })
    return entries


# ─── Deduplication ───────────────────────────────────────────────

def normalize_condition(val: str) -> str:
    """Normalize condition string for dedup comparison."""
    if not val or val in ('', 'not_specified', 'unknown', 'None', None):
        return ''
    return val.strip().lower()


def condition_key(e: Dict) -> str:
    """Build a condition key for dedup: formulation|food|route.
    Empty string if no conditions specified (matches anything)."""
    form = normalize_condition(e.get('formulation', ''))
    food = normalize_condition(e.get('food', ''))
    route = normalize_condition(e.get('route', ''))
    return f"{form}|{food}|{route}"


def conditions_match(key1: str, key2: str) -> bool:
    """Two condition keys match if they're identical,
    or if either has all-empty fields (no info → assume same)."""
    if key1 == key2:
        return True
    # If either is fully empty (no condition info), treat as match
    if key1 == '||' or key2 == '||':
        return True
    # Partial match: compare non-empty fields
    parts1 = key1.split('|')
    parts2 = key2.split('|')
    for p1, p2 in zip(parts1, parts2):
        if p1 and p2 and p1 != p2:
            return False  # Explicit conflict
    return True


def dedup_entries(all_entries: List[Dict]) -> Tuple[List[Dict], dict]:
    """Deduplicate entries by InChIKey14 + dose + conditions.

    Condition-aware: same drug+dose but different formulation (IR vs ER),
    food (fasted vs fed), or route (oral vs IV) are KEPT as separate entries.
    Only true duplicates (same drug, dose, conditions) are merged."""

    tier_priority = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}

    # Group by InChIKey14
    by_ik = defaultdict(list)
    no_ik = []
    for e in all_entries:
        ik = e.get('inchikey14', '')
        if ik:
            by_ik[ik].append(e)
        else:
            no_ik.append(e)

    deduped = []
    dup_stats = {'total_before': len(all_entries), 'duplicates_removed': 0,
                 'unique_drugs': len(by_ik), 'no_inchikey': len(no_ik),
                 'condition_preserved': 0}

    for ik, entries in by_ik.items():
        # Sort by tier priority (T1 first)
        entries.sort(key=lambda e: tier_priority.get(e.get('tier', 'T4'), 4))

        # Group by dose + condition
        dose_cond_groups = []
        for e in entries:
            dose = e.get('dose_mg')
            ckey = condition_key(e)
            matched = False
            for group in dose_cond_groups:
                ref = group[0]
                ref_ckey = condition_key(ref)
                if dose_match(dose, ref.get('dose_mg', 0)) and conditions_match(ckey, ref_ckey):
                    group.append(e)
                    matched = True
                    break
            if not matched:
                dose_cond_groups.append([e])

        # Track how many extra entries we preserved due to conditions
        if len(dose_cond_groups) > 1:
            # Check if any were saved by condition differentiation
            dose_only_groups = []
            for e in entries:
                dose = e.get('dose_mg')
                matched = False
                for group in dose_only_groups:
                    if dose_match(dose, group[0].get('dose_mg', 0)):
                        group.append(e)
                        matched = True
                        break
                if not matched:
                    dose_only_groups.append([e])
            dup_stats['condition_preserved'] += len(dose_cond_groups) - len(dose_only_groups)

        for group in dose_cond_groups:
            best = group[0]  # highest tier
            best['n_sources'] = len(group)
            best['all_sources_list'] = list(set(e.get('source', '') for e in group))
            deduped.append(best)
            dup_stats['duplicates_removed'] += len(group) - 1

    dup_stats['total_after'] = len(deduped)
    return deduped, dup_stats


# ─── Cross-Source Validation ─────────────────────────────────────

def cross_source_validation(all_entries: List[Dict]) -> Dict:
    """Compare Cmax values for same drug-dose pairs across sources."""

    # Group by InChIKey14 + dose
    by_ik = defaultdict(list)
    for e in all_entries:
        ik = e.get('inchikey14', '')
        if ik and e.get('cmax_ngml') and e.get('dose_mg'):
            by_ik[ik].append(e)

    comparisons = {
        'FDA_LLM_vs_FDA_regex': [],
        'FDA_LLM_vs_Sisyphus': [],
        'FDA_regex_vs_Sisyphus': [],
        'FDA_LLM_vs_ChEMBL': [],
        'any_pair': [],
    }

    drug_level_comparison = []

    for ik, entries in by_ik.items():
        # Group by source
        by_source = defaultdict(list)
        for e in entries:
            src = e.get('source', '')
            if 'Sisyphus' in src:
                by_source['Sisyphus'].append(e)
            elif 'LLM' in src or src == 'FDA_LLM':
                by_source['FDA_LLM'].append(e)
            elif 'regex' in src or src == 'FDA_regex':
                by_source['FDA_regex'].append(e)
            elif 'ChEMBL' in src:
                by_source['ChEMBL'].append(e)
            elif 'PLM' in src:
                by_source['PLM_FDA'].append(e)

        if len(by_source) < 2:
            continue

        # Compare each pair of sources
        source_names = list(by_source.keys())
        drug_name = entries[0].get('drug', ik)

        drug_comp = {
            'drug': drug_name,
            'inchikey14': ik,
            'sources': {s: len(v) for s, v in by_source.items()},
            'pairs': [],
        }

        for i in range(len(source_names)):
            for j in range(i + 1, len(source_names)):
                s1, s2 = source_names[i], source_names[j]
                entries1, entries2 = by_source[s1], by_source[s2]

                for e1 in entries1:
                    for e2 in entries2:
                        if dose_match(e1['dose_mg'], e2['dose_mg']):
                            c1 = e1['cmax_ngml']
                            c2 = e2['cmax_ngml']
                            fe = fold_error(c1, c2)
                            pair = {
                                'drug': drug_name,
                                'inchikey14': ik,
                                'source1': s1,
                                'source2': s2,
                                'dose1_mg': e1['dose_mg'],
                                'dose2_mg': e2['dose_mg'],
                                'cmax1_ngml': c1,
                                'cmax2_ngml': c2,
                                'fold_error': round(fe, 3),
                                'log_ratio': round(math.log10(c1 / c2) if c1 > 0 and c2 > 0 else 0, 4),
                            }

                            comparisons['any_pair'].append(pair)
                            drug_comp['pairs'].append(pair)

                            # Categorize
                            key = f"{s1}_vs_{s2}"
                            rev_key = f"{s2}_vs_{s1}"
                            for cat_key in comparisons:
                                if cat_key == 'any_pair':
                                    continue
                                norm_cat = cat_key.replace('_vs_', ' ').lower()
                                if (s1.lower() in norm_cat and s2.lower() in norm_cat) or \
                                   (s2.lower() in norm_cat and s1.lower() in norm_cat):
                                    comparisons[cat_key].append(pair)

        if drug_comp['pairs']:
            drug_level_comparison.append(drug_comp)

    # Compute summary statistics
    summary = {}
    for cat, pairs in comparisons.items():
        if not pairs:
            summary[cat] = {'n': 0}
            continue
        fes = [p['fold_error'] for p in pairs if math.isfinite(p['fold_error'])]
        log_ratios = [p['log_ratio'] for p in pairs if math.isfinite(p['log_ratio'])]
        within_2fold = sum(1 for fe in fes if fe <= 2.0)
        within_3fold = sum(1 for fe in fes if fe <= 3.0)

        summary[cat] = {
            'n': len(pairs),
            'aafe': round(aafe(fes), 3) if fes else None,
            'median_fold_error': round(sorted(fes)[len(fes)//2], 3) if fes else None,
            'mean_log_ratio': round(sum(log_ratios)/len(log_ratios), 4) if log_ratios else None,
            'within_2fold_pct': round(100 * within_2fold / len(fes), 1) if fes else None,
            'within_3fold_pct': round(100 * within_3fold / len(fes), 1) if fes else None,
            'n_unique_drugs': len(set(p['inchikey14'] for p in pairs)),
        }

    return {
        'summary': summary,
        'per_drug': drug_level_comparison,
        'all_pairs': comparisons['any_pair'],
    }


# ─── Main Build Pipeline ────────────────────────────────────────

def main():
    print("="*60)
    print("Building PLM Dataset v12")
    print("="*60)

    # Load all sources
    print("\n[1/6] Loading v10 (PLM + Sisyphus combined)...")
    v10 = load_v10()
    print(f"  {len(v10)} entries")

    print("\n[2/6] Loading FDA table parser...")
    fda_regex = load_fda_table_parsed()
    print(f"  {len(fda_regex)} entries (dose+cmax+smiles)")

    print("\n[3/6] Loading LLM extractions...")
    llm_ext = load_llm_extraction()
    print(f"  {len(llm_ext)} entries")

    print("\n[4/6] Loading external integrated (ChEMBL, etc.)...")
    external = load_external_integrated()
    print(f"  {len(external)} entries")

    # Combine all
    all_entries = v10 + fda_regex + llm_ext + external
    print(f"\n  Total raw entries: {len(all_entries)}")

    # ── Cross-source validation (BEFORE dedup) ──
    print("\n[5/6] Running cross-source validation...")
    xval = cross_source_validation(all_entries)

    print(f"\n  Cross-source comparison summary:")
    for cat, stats in xval['summary'].items():
        if stats['n'] > 0:
            print(f"    {cat}: N={stats['n']}, AAFE={stats.get('aafe','?')}, "
                  f"2-fold={stats.get('within_2fold_pct','?')}%, "
                  f"drugs={stats.get('n_unique_drugs','?')}")

    # Save cross-validation
    xval_path = Path('data/curated/cross_source_validation.json')
    with open(xval_path, 'w') as f:
        json.dump(xval, f, indent=2)
    print(f"  Saved: {xval_path}")

    # ── Deduplication ──
    print("\n[6/6] Deduplicating...")
    deduped, dup_stats = dedup_entries(all_entries)
    print(f"  Before: {dup_stats['total_before']}")
    print(f"  After:  {dup_stats['total_after']}")
    print(f"  Removed: {dup_stats['duplicates_removed']} duplicates")
    print(f"  Unique drugs: {dup_stats['unique_drugs']}")

    # Add log_cd (log10(cmax/dose)) for model compatibility
    for e in deduped:
        cmax = e.get('cmax_ngml', 0)
        dose = e.get('dose_mg', 0)
        if cmax and dose and cmax > 0 and dose > 0:
            e['log_cd'] = round(math.log10(cmax / dose), 6)
        else:
            e['log_cd'] = None

    # Sort by tier, then drug, then dose
    tier_order = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
    deduped.sort(key=lambda e: (tier_order.get(e.get('tier', 'T4'), 4),
                                 e.get('drug', ''), e.get('dose_mg', 0)))

    # ── Tier breakdown ──
    from collections import Counter
    tier_counts = Counter(e.get('tier', '?') for e in deduped)
    source_counts = Counter(e.get('source', '?') for e in deduped)
    with_log_cd = sum(1 for e in deduped if e.get('log_cd') is not None)

    print(f"\n  Tier breakdown:")
    for tier in ['T1', 'T2', 'T3', 'T4']:
        print(f"    {tier}: {tier_counts.get(tier, 0)}")

    print(f"\n  Source breakdown:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {src}: {cnt}")

    print(f"\n  With log_cd: {with_log_cd}/{len(deduped)}")

    # ── Save v12 ──
    v12_path = Path('data/curated/plm_dataset_v12.json')
    with open(v12_path, 'w') as f:
        json.dump(deduped, f, indent=2, default=str)
    print(f"\n  Dataset saved: {v12_path}")

    # ── Save build stats ──
    build_stats = {
        'version': 'v12',
        'build_date': '2026-04-06',
        'sources': {
            'v10_combined': len(v10),
            'fda_regex': len(fda_regex),
            'llm_extraction': len(llm_ext),
            'external_integrated': len(external),
        },
        'total_raw': len(all_entries),
        'total_deduped': len(deduped),
        'duplicates_removed': dup_stats['duplicates_removed'],
        'unique_drugs': dup_stats['unique_drugs'],
        'tier_breakdown': dict(tier_counts),
        'source_breakdown': dict(source_counts),
        'with_log_cd': with_log_cd,
        'cross_validation_summary': xval['summary'],
    }

    stats_path = Path('data/curated/v12_build_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(build_stats, f, indent=2)

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print(f"PLM Dataset v12 — FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries:    {len(deduped)}")
    print(f"Unique drugs:     {dup_stats['unique_drugs']}")
    print(f"Tiers:            T1={tier_counts.get('T1',0)} (Sisyphus) | "
          f"T2={tier_counts.get('T2',0)} (FDA LLM/PLM) | "
          f"T3={tier_counts.get('T3',0)} (FDA regex) | "
          f"T4={tier_counts.get('T4',0)} (ChEMBL)")
    print(f"\nCross-source (all pairs): N={xval['summary']['any_pair']['n']}, "
          f"AAFE={xval['summary']['any_pair'].get('aafe','?')}")


if __name__ == '__main__':
    main()
