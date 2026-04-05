"""
Evaluate peer-critique predictions.
Compares:
  - Agent A (Round 2 analogical): current best 2.126
  - Agent B (peer critic): independent revision
  - Median(A, B): conservative ensemble
  - B-only: trust critic
"""

import json, math
import numpy as np


def main():
    # Load HO actual
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    ho_info = {}
    for d in ho['holdout_drugs']:
        if d.get('smiles') and d.get('dose_mg') and d.get('cmax_obs_ngml'):
            ho_info[d['name']] = {
                'dose_mg': d['dose_mg'],
                'actual_log_cd': math.log10(d['cmax_obs_ngml'] / d['dose_mg']),
            }

    # Load Agent B's revisions
    b_preds = {}
    for i in range(4):
        try:
            with open(f'/tmp/ho_peer_b{i}.json') as f:
                for item in json.load(f):
                    name = item['name']
                    b_preds[name] = {
                        'a_cmax': item.get('colleague_pred_ngml'),
                        'b_cmax': item.get('your_revised_cmax_ngml'),
                        'disagreement': item.get('disagreement', 0),
                        'critique': item.get('critique', ''),
                    }
        except FileNotFoundError:
            print(f"Missing /tmp/ho_peer_b{i}.json")

    print(f"Peer-critique predictions: {len(b_preds)}")

    # Compute metrics
    aligned = [(name, b_preds[name], ho_info[name]) for name in ho_info if name in b_preds]
    print(f"Aligned: {len(aligned)}")

    # Per-strategy
    errs = {}
    for strategy in ['A', 'B', 'median_AB', 'geomean_AB']:
        errs[strategy] = []

    disagreement_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for name, b, info in aligned:
        dose = info['dose_mg']
        actual = info['actual_log_cd']
        a_cmax = b['a_cmax']
        b_cmax = b['b_cmax']
        disagreement_counts[b['disagreement']] = disagreement_counts.get(b['disagreement'], 0) + 1
        if a_cmax is None or b_cmax is None or a_cmax <= 0 or b_cmax <= 0:
            continue
        a_log = math.log10(a_cmax / dose)
        b_log = math.log10(b_cmax / dose)
        errs['A'].append(a_log - actual)
        errs['B'].append(b_log - actual)
        errs['median_AB'].append(np.median([a_log, b_log]) - actual)
        errs['geomean_AB'].append(np.mean([a_log, b_log]) - actual)

    print(f"\n{'='*80}")
    print(f"Disagreement distribution")
    print(f"{'='*80}")
    total = sum(disagreement_counts.values())
    for lvl in sorted(disagreement_counts.keys()):
        name = ['agree', 'small', 'moderate', 'major'][lvl]
        print(f"  Level {lvl} ({name:<10s}): {disagreement_counts[lvl]:3d} ({100*disagreement_counts[lvl]/total:.0f}%)")

    print(f"\n{'='*80}")
    print(f"PEER-CRITIQUE RESULTS")
    print(f"{'='*80}")
    for strategy in ['A', 'B', 'median_AB', 'geomean_AB']:
        e = np.array(errs[strategy])
        aafe = 10 ** np.mean(np.abs(e))
        bias = float(np.mean(e))
        f2 = float(np.mean(np.abs(e) < np.log10(2))) * 100
        print(f"  {strategy:<12s}: AAFE={aafe:.3f}  bias={bias:+.3f}  2f={f2:.1f}%  n={len(e)}")

    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    print(f"  PLM baseline:          3.355")
    print(f"  Sisyphus Meta:         2.283 (SOTA)")
    print(f"  LLM single-shot:       2.228")
    print(f"  LLM self-consistency:  2.126 (prior best)")
    print(f"  Peer-critique B:       {10**np.mean(np.abs(errs['B'])):.3f}")
    print(f"  Peer-critique median:  {10**np.mean(np.abs(errs['median_AB'])):.3f}")

    # Save results
    results = {
        'n': len(aligned),
        'disagreement_dist': disagreement_counts,
        'metrics': {
            strategy: {
                'aafe': round(10 ** np.mean(np.abs(np.array(errs[strategy]))), 3),
                'bias': round(float(np.mean(np.array(errs[strategy]))), 3),
                'f2_pct': round(float(np.mean(np.abs(np.array(errs[strategy])) < np.log10(2))) * 100, 1),
            } for strategy in ['A', 'B', 'median_AB', 'geomean_AB']
        },
    }
    with open('data/validation/peer_critique_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n→ Saved data/validation/peer_critique_results.json")

    # Per-drug comparison: where does B help most?
    print(f"\n{'='*80}")
    print(f"Top 10 drugs where B improved most")
    print(f"{'='*80}")
    improvements = []
    for name, b, info in aligned:
        if b['a_cmax'] is None or b['b_cmax'] is None: continue
        dose = info['dose_mg']; actual = info['actual_log_cd']
        a_err = abs(math.log10(b['a_cmax'] / dose) - actual)
        b_err = abs(math.log10(b['b_cmax'] / dose) - actual)
        improvements.append((name, a_err - b_err, a_err, b_err, b['disagreement'], b['critique']))
    improvements.sort(key=lambda x: -x[1])
    print(f"  {'name':<25s} {'A err':>7s} {'B err':>7s} {'Δ':>7s} {'disa':>5s} {'critique':<30s}")
    for name, imp, a, b, disa, crit in improvements[:10]:
        print(f"  {name[:24]:<25s} {a:>7.2f} {b:>7.2f} {imp:>+7.2f} {disa:>5d} {crit[:30]:<30s}")

    print(f"\n{'='*80}")
    print(f"Top 10 drugs where B hurt most")
    print(f"{'='*80}")
    print(f"  {'name':<25s} {'A err':>7s} {'B err':>7s} {'Δ':>7s} {'disa':>5s} {'critique':<30s}")
    for name, imp, a, b, disa, crit in improvements[-10:]:
        print(f"  {name[:24]:<25s} {a:>7.2f} {b:>7.2f} {imp:>+7.2f} {disa:>5d} {crit[:30]:<30s}")


if __name__ == '__main__':
    main()
