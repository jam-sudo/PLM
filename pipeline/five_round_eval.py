"""
Evaluate 5-round self-consistency (rounds 1-5).
Rounds: physiological, analogical, label-recall, structure-first, reverse-from-CL.
"""

import json, math
import numpy as np


def load_round(r):
    preds = {}
    for b in range(4):
        path = f'/tmp/ho_r{r}_b{b}.json'
        try:
            with open(path) as f: batch = json.load(f)
            for item in batch:
                name = item['name']
                cmax = item.get('predicted_cmax_ngml')
                if cmax is not None and cmax > 0:
                    preds[name] = cmax
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  Error loading {path}: {e}")
    return preds


def metrics(pred_dict, ho_info):
    errs = []
    for name, cmax in pred_dict.items():
        if name not in ho_info: continue
        log_cd = math.log10(cmax / ho_info[name]['dose_mg'])
        err = log_cd - ho_info[name]['log_cd_actual']
        errs.append(err)
    if not errs: return None
    errs = np.array(errs)
    return {
        'aafe': round(10 ** np.mean(np.abs(errs)), 3),
        'bias': round(float(np.mean(errs)), 3),
        'f2': round(float(np.mean(np.abs(errs) < np.log10(2))) * 100, 1),
        'f3': round(float(np.mean(np.abs(errs) < np.log10(3))) * 100, 1),
        'n': len(errs),
    }


def main():
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    ho_info = {}
    for d in ho['holdout_drugs']:
        if d.get('smiles') and d.get('dose_mg') and d.get('cmax_obs_ngml'):
            ho_info[d['name']] = {
                'dose_mg': d['dose_mg'],
                'cmax_obs': d['cmax_obs_ngml'],
                'log_cd_actual': math.log10(d['cmax_obs_ngml'] / d['dose_mg']),
            }

    rounds = {}
    strategy_names = ['physiological', 'analogical', 'label_recall', 'structure_first', 'reverse_from_CL']
    for r in range(1, 6):
        rounds[r] = load_round(r)
        m = metrics(rounds[r], ho_info)
        if m: print(f"Round {r} ({strategy_names[r-1]:<18s}): AAFE={m['aafe']}  bias={m['bias']:+.3f}  2f={m['f2']}%  n={m['n']}")

    # Aggregations
    print(f"\n{'='*80}")
    print(f"AGGREGATION STRATEGIES")
    print(f"{'='*80}")

    # 3-round (original)
    med3 = {}; geom3 = {}
    for name in ho_info:
        cmaxs = [rounds[r].get(name) for r in [1,2,3] if name in rounds[r]]
        cmaxs = [c for c in cmaxs if c]
        if len(cmaxs) >= 2:
            med3[name] = float(np.median(cmaxs))
            geom3[name] = float(10 ** np.mean(np.log10(cmaxs)))

    # 5-round
    med5 = {}; geom5 = {}; mean5 = {}
    for name in ho_info:
        cmaxs = [rounds[r].get(name) for r in range(1,6) if name in rounds[r]]
        cmaxs = [c for c in cmaxs if c]
        if len(cmaxs) >= 3:
            med5[name] = float(np.median(cmaxs))
            geom5[name] = float(10 ** np.mean(np.log10(cmaxs)))
            mean5[name] = float(np.mean(cmaxs))

    # Trimmed mean (remove min and max, take mean of middle 3)
    trim5 = {}
    for name in ho_info:
        cmaxs = sorted([rounds[r].get(name) for r in range(1,6) if name in rounds[r] and rounds[r].get(name)])
        if len(cmaxs) == 5:
            trim5[name] = float(np.mean(cmaxs[1:4]))  # trim min and max

    for label, d in [('3-round median', med3), ('3-round geomean', geom3),
                      ('5-round median', med5), ('5-round geomean', geom5),
                      ('5-round mean', mean5), ('5-round trimmed mean', trim5)]:
        m = metrics(d, ho_info)
        if m: print(f"  {label:<25s}: AAFE={m['aafe']}  bias={m['bias']:+.3f}  2f={m['f2']}%  n={m['n']}")

    # All-round comparison
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    print(f"  PLM baseline:          3.355")
    print(f"  Sisyphus Meta:         2.283 (SOTA)")
    print(f"  LLM single-shot:       2.228")
    print(f"  3-round best (analog): 2.126 (prior best)")
    print(f"  5-round geomean:       {metrics(geom5, ho_info)['aafe']}")
    print(f"  5-round median:        {metrics(med5, ho_info)['aafe']}")
    if trim5: print(f"  5-round trimmed mean:  {metrics(trim5, ho_info)['aafe']}")

    # Save
    results = {
        'per_round': {f'round_{r}': metrics(rounds[r], ho_info) for r in range(1,6)},
        'aggregations': {
            '3round_median': metrics(med3, ho_info),
            '3round_geomean': metrics(geom3, ho_info),
            '5round_median': metrics(med5, ho_info),
            '5round_geomean': metrics(geom5, ho_info),
            '5round_mean': metrics(mean5, ho_info),
            '5round_trimmed': metrics(trim5, ho_info) if trim5 else None,
        },
        'strategy_names': strategy_names,
    }
    with open('data/validation/five_round_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved data/validation/five_round_results.json")


if __name__ == '__main__':
    main()
