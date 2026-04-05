"""
Evaluate LLM direct Cmax predictions vs actual HO values.
Aggregates 4 batch prediction files and computes AAFE.
"""

import json, math
import numpy as np

def main():
    # Load actual HO
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    actual = {}
    for d in ho['holdout_drugs']:
        if d.get('smiles') and d.get('dose_mg') and d.get('cmax_obs_ngml'):
            actual[d['name']] = {
                'dose_mg': d['dose_mg'],
                'cmax_obs_ngml': d['cmax_obs_ngml'],
                'log_cd_actual': math.log10(d['cmax_obs_ngml'] / d['dose_mg']),
            }
    print(f"HO drugs: {len(actual)}")

    # Load all prediction batches
    predictions = {}
    for i in range(4):
        path = f'/tmp/ho_predictions_{i}.json'
        try:
            with open(path) as f:
                batch = json.load(f)
            for item in batch:
                predictions[item['name']] = item
        except FileNotFoundError:
            print(f"  Missing: {path}")
            continue
    print(f"LLM predictions: {len(predictions)}")

    # Align and compute
    results = []
    for name, a in actual.items():
        if name not in predictions:
            print(f"  Missing prediction: {name}")
            continue
        p = predictions[name]
        cmax_pred = p.get('predicted_cmax_ngml')
        if cmax_pred is None or cmax_pred <= 0:
            print(f"  Invalid prediction for {name}: {cmax_pred}")
            continue
        log_cd_pred = math.log10(cmax_pred / a['dose_mg'])
        err = abs(log_cd_pred - a['log_cd_actual'])
        signed = log_cd_pred - a['log_cd_actual']
        fold_err = 10 ** err
        results.append({
            'name': name,
            'dose_mg': a['dose_mg'],
            'actual_cmax': a['cmax_obs_ngml'],
            'pred_cmax': cmax_pred,
            'log_cd_actual': a['log_cd_actual'],
            'log_cd_pred': log_cd_pred,
            'abs_err': err,
            'signed_err': signed,
            'fold_err': fold_err,
            'reasoning': p.get('reasoning_brief', ''),
        })

    if not results:
        print("No valid predictions to evaluate.")
        return

    # Metrics
    errs = [r['abs_err'] for r in results]
    signed = [r['signed_err'] for r in results]
    fold_errs = [r['fold_err'] for r in results]
    aafe = 10 ** np.mean(errs)
    bias = float(np.mean(signed))
    f2 = float(np.mean(np.array(errs) < np.log10(2))) * 100
    f3 = float(np.mean(np.array(errs) < np.log10(3))) * 100

    print(f"\n{'='*80}")
    print(f"LLM DIRECT PREDICTION RESULTS")
    print(f"{'='*80}")
    print(f"N drugs evaluated: {len(results)}")
    print(f"HO AAFE: {aafe:.3f}")
    print(f"Mean signed error (bias): {bias:+.3f}")
    print(f"2-fold %: {f2:.1f}%")
    print(f"3-fold %: {f3:.1f}%")
    print(f"Max fold error: {max(fold_errs):.1f}x")
    print(f"\nComparison:")
    print(f"  PLM baseline:    HO AAFE 3.355, bias +0.269")
    print(f"  Sisyphus Engine: HO AAFE 3.416")
    print(f"  Sisyphus ML:     HO AAFE 2.336")
    print(f"  Sisyphus Meta:   HO AAFE 2.283, bias +0.037")
    print(f"  LLM direct:      HO AAFE {aafe:.3f}, bias {bias:+.3f}")

    # Top 5 best
    results.sort(key=lambda r: r['abs_err'])
    print(f"\nTop 5 best predictions:")
    for r in results[:5]:
        print(f"  {r['name'][:25]:<26s} fold={r['fold_err']:.2f}  signed={r['signed_err']:+.2f}")

    # Top 10 worst
    print(f"\nTop 10 worst predictions:")
    for r in results[-10:]:
        print(f"  {r['name'][:25]:<26s} fold={r['fold_err']:.2f}  signed={r['signed_err']:+.2f}  [{r['reasoning'][:40]}]")

    # Save
    with open('data/validation/llm_smoke_results.json', 'w') as f:
        json.dump({
            'aafe': round(aafe, 3), 'bias': round(bias, 3),
            'f2': round(f2, 1), 'f3': round(f3, 1),
            'n': len(results),
            'per_drug': results,
        }, f, indent=2, default=float)
    print(f"\n→ Saved data/validation/llm_smoke_results.json")


if __name__ == '__main__':
    main()
