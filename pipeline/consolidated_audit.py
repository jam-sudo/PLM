"""
Consolidated benchmark audit: integrate 4 independent sources.

Sources:
1. Sisyphus obs (the benchmark — questionable)
2. Sisyphus meta (their own model)
3. ChEMBL aggregated Cmax (independent database)
4. LLM extraction (FDA labels, 17 drugs)

Logic: a Sisyphus obs value is CONFIRMED SUSPECT if ≥2 independent sources
disagree with obs in the SAME direction by reasonable magnitude (3-30x).

Avoid:
- Single-source disagreements
- Extreme magnitudes (>100x suggest unit errors in source)
- Cases where meta+obs agree (could be drug-specific PK quirk)
"""

import json, math
import numpy as np
from collections import defaultdict


def main():
    # Load all sources
    with open('data/validation/holdout_definition.json') as f:
        ho = json.load(f)
    holdout_drugs = ho['holdout_drugs']

    with open('data/validation/chembl_audit.json') as f:
        chembl = {r['drug']: r for r in json.load(f)}

    with open('data/validation/holdout_audit.json') as f:
        llm_audit = json.load(f)
    llm_by_drug = {a['drug']: a for a in llm_audit['audit_details']}

    confirmed = []
    print("="*100)
    print("CONSOLIDATED AUDIT: Sisyphus obs vs meta + ChEMBL + LLM")
    print("="*100)
    print(f"{'drug':<22s} {'obs':>8s} {'meta':>8s} {'chembl':>8s} {'llm':>8s} {'verdict':>10s} {'conf':>6s}")
    print('-'*100)

    for d in holdout_drugs:
        name = d['name']
        obs = d['cmax_obs_ngml']
        meta = d.get('cmax_sisyphus_meta_mgL', 0) * 1000 if d.get('cmax_sisyphus_meta_mgL') else None

        chembl_val = chembl.get(name, {}).get('chembl_median_cmax_ngml')
        llm_val = llm_by_drug.get(name, {}).get('llm_median_cmax_at_dose')

        # Compute log deltas
        def log_delta(v, ref):
            if v is None or ref is None or ref <= 0 or v <= 0: return None
            return math.log10(v / ref)

        d_meta = log_delta(meta, obs)
        d_chembl = log_delta(chembl_val, obs)
        d_llm = log_delta(llm_val, obs)

        # Collect non-null deltas
        deltas = [(d_meta, 'meta'), (d_chembl, 'chembl'), (d_llm, 'llm')]
        deltas = [(d, s) for d, s in deltas if d is not None]

        # Filter to "reasonable" (0.48 = 3x < magnitude < 100x = 2.0)
        valid_deltas = [(d, s) for d, s in deltas if 0.48 < abs(d) < 2.0]

        verdict = 'ok'
        confidence = 0
        if len(valid_deltas) >= 2:
            # Check same sign
            signs = [np.sign(d) for d, _ in valid_deltas]
            if all(s == signs[0] for s in signs):
                verdict = 'SUSPECT'
                confidence = len(valid_deltas)
        elif len(valid_deltas) == 1 and abs(valid_deltas[0][0]) > 0.9:  # single source but >8x
            verdict = 'check'
            confidence = 1

        fmt = lambda x: f"{x:>8.1f}" if x is not None else "     n/a"
        marker = '**' if verdict == 'SUSPECT' else ('?' if verdict == 'check' else '  ')
        print(f"  {name[:20]:<22s} {obs:>8.1f} {fmt(meta)} {fmt(chembl_val)} {fmt(llm_val)} {marker}{verdict:>8s} {confidence:>6d}")

        if verdict == 'SUSPECT':
            # Use median of non-obs sources as corrected value
            corrections = [v for v in [meta, chembl_val, llm_val] if v is not None]
            corrected = float(np.median(corrections))
            confirmed.append({
                'drug': name, 'ik': d['inchikey14'],
                'original': obs, 'corrected': corrected,
                'sources': {'meta': meta, 'chembl': chembl_val, 'llm': llm_val},
                'n_sources_agree': confidence,
                'fold_change': corrected / obs,
            })

    print(f"\n{'='*70}")
    print(f"Confirmed SUSPECT drugs (≥2 independent sources): {len(confirmed)}")
    print(f"{'='*70}")
    for c in confirmed:
        print(f"  {c['drug']:<25s} obs={c['original']:>9.1f} → corrected={c['corrected']:>9.1f}  ({c['fold_change']:.2f}x)  [{c['n_sources_agree']} sources]")

    # Save
    with open('data/validation/consolidated_audit.json', 'w') as f:
        json.dump({'confirmed_suspects': confirmed, 'n_confirmed': len(confirmed)}, f, indent=2)
    print(f"\n→ Saved to data/validation/consolidated_audit.json")


if __name__ == '__main__':
    main()
