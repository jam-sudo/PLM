"""
Aggregate 3-round LLM predictions via self-consistency.
Computes AAFE for each round + median/mean aggregation.
"""

import json, math
import numpy as np


def load_round(round_num):
    """Load predictions from 4 batches of a single round."""
    preds = {}
    for b in range(4):
        path = f'/tmp/ho_r{round_num}_b{b}.json'
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


def compute_aafe_from_cmax(pred_cmax_dict, ho_info):
    """Compute AAFE using dose to convert Cmax → log_cd."""
    errs = []
    signed = []
    for name, cmax_pred in pred_cmax_dict.items():
        if name not in ho_info: continue
        dose = ho_info[name]['dose_mg']
        log_cd_pred = math.log10(cmax_pred / dose)
        log_cd_actual = ho_info[name]['log_cd_actual']
        err = log_cd_pred - log_cd_actual
        errs.append(abs(err))
        signed.append(err)
    if not errs: return None, None, None, 0
    aafe = 10 ** np.mean(errs)
    bias = float(np.mean(signed))
    f2 = float(np.mean(np.array(errs) < np.log10(2))) * 100
    return aafe, bias, f2, len(errs)


def main():
    # Load HO actual data
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    ho_info = {}
    for d in ho['holdout_drugs']:
        if d.get('smiles') and d.get('dose_mg') and d.get('cmax_obs_ngml'):
            ho_info[d['name']] = {
                'dose_mg': d['dose_mg'],
                'cmax_obs': d['cmax_obs_ngml'],
                'log_cd_actual': math.log10(d['cmax_obs_ngml'] / d['dose_mg']),
            }

    print("Loading 3 rounds of predictions...")
    rounds = {}
    for r in [1, 2, 3]:
        rounds[r] = load_round(r)
        print(f"  Round {r}: {len(rounds[r])} predictions")

    # Per-round AAFE
    print(f"\n{'='*80}")
    print(f"PER-ROUND AAFE")
    print(f"{'='*80}")
    for r in [1, 2, 3]:
        aafe, bias, f2, n = compute_aafe_from_cmax(rounds[r], ho_info)
        print(f"  Round {r} ({['physiological','analogical','label_recall'][r-1]:<15s}): AAFE={aafe:.3f}  bias={bias:+.3f}  2f={f2:.1f}%  n={n}")

    # Per-drug aggregation (median of 3 rounds)
    print(f"\n{'='*80}")
    print(f"SELF-CONSISTENCY AGGREGATION")
    print(f"{'='*80}")

    median_preds = {}
    mean_preds = {}
    geomean_preds = {}
    for name in ho_info.keys():
        cmaxs = []
        for r in [1, 2, 3]:
            if name in rounds[r]:
                cmaxs.append(rounds[r][name])
        if len(cmaxs) >= 2:  # need at least 2 rounds
            median_preds[name] = float(np.median(cmaxs))
            mean_preds[name] = float(np.mean(cmaxs))
            geomean_preds[name] = float(10 ** np.mean(np.log10(cmaxs)))

    print(f"  Drugs with ≥2 rounds: {len(median_preds)}")

    for name, preds in [('Median', median_preds), ('Arithmetic mean', mean_preds),
                         ('Geometric mean', geomean_preds)]:
        aafe, bias, f2, n = compute_aafe_from_cmax(preds, ho_info)
        print(f"  {name:<18s}: AAFE={aafe:.3f}  bias={bias:+.3f}  2f={f2:.1f}%  n={n}")

    # Compare to single-shot (smoke test)
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    print(f"  PLM baseline:           HO AAFE 3.355")
    print(f"  Sisyphus Meta:          HO AAFE 2.283")
    print(f"  LLM single-shot (prior): HO AAFE 2.228")
    print(f"  LLM self-consistency:    HO AAFE {compute_aafe_from_cmax(median_preds, ho_info)[0]:.3f}")

    # Save
    median_aafe, median_bias, median_f2, median_n = compute_aafe_from_cmax(median_preds, ho_info)
    with open('data/validation/llm_cot_results.json', 'w') as f:
        json.dump({
            'rounds': {
                f'round_{r}': {'preds': rounds[r]} for r in [1,2,3]
            },
            'aggregates': {
                'median': {'aafe': round(median_aafe, 3), 'bias': round(median_bias, 3),
                           'f2': round(median_f2, 1), 'n': median_n},
            },
            'per_round_aafe': {
                r: compute_aafe_from_cmax(rounds[r], ho_info)[0] for r in [1,2,3]
            },
        }, f, indent=2, default=float)
    print(f"\n→ Saved data/validation/llm_cot_results.json")

    # Per-drug output
    per_drug = {}
    for name in sorted(ho_info.keys()):
        if name not in median_preds: continue
        dose = ho_info[name]['dose_mg']
        per_drug[name] = {
            'actual_log_cd': ho_info[name]['log_cd_actual'],
            'round1_log_cd': math.log10(rounds[1].get(name, ho_info[name]['cmax_obs']) / dose) if name in rounds[1] else None,
            'round2_log_cd': math.log10(rounds[2].get(name, ho_info[name]['cmax_obs']) / dose) if name in rounds[2] else None,
            'round3_log_cd': math.log10(rounds[3].get(name, ho_info[name]['cmax_obs']) / dose) if name in rounds[3] else None,
            'median_log_cd': math.log10(median_preds[name] / dose),
            'median_err': abs(math.log10(median_preds[name] / dose) - ho_info[name]['log_cd_actual']),
        }
    with open('data/validation/llm_cot_per_drug.json', 'w') as f:
        json.dump(per_drug, f, indent=2, default=float)

    # Top 10 worst
    print(f"\n{'='*80}")
    print(f"Top 10 worst (median-based)")
    print(f"{'='*80}")
    items = sorted(per_drug.items(), key=lambda x: -x[1]['median_err'])
    print(f"  {'name':<25s} {'actual':>7s} {'r1':>7s} {'r2':>7s} {'r3':>7s} {'med':>7s} {'|err|':>6s}")
    for name, d in items[:10]:
        r1 = f"{d['round1_log_cd']:+.2f}" if d['round1_log_cd'] is not None else "n/a"
        r2 = f"{d['round2_log_cd']:+.2f}" if d['round2_log_cd'] is not None else "n/a"
        r3 = f"{d['round3_log_cd']:+.2f}" if d['round3_log_cd'] is not None else "n/a"
        print(f"  {name[:24]:<25s} {d['actual_log_cd']:>+7.2f} {r1:>7s} {r2:>7s} {r3:>7s} {d['median_log_cd']:>+7.2f} {d['median_err']:>6.2f}")


if __name__ == '__main__':
    main()
