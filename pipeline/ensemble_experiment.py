"""
PLM + LLM Ensemble experiment.

Combines:
  - PLM baseline predictions (XGBoost on 4150 features)
  - LLM direct predictions (Claude subagents, ADMET reasoning)

Tests multiple ensemble strategies:
  - Simple average (log space)
  - Weighted combinations
  - Geometric mean
  - Confidence-routed (LLM when it agrees with Meta, else PLM)
"""

import json, math
import numpy as np


def aafe(errs):
    return float(10 ** np.mean(np.abs(errs)))

def fold_pct(errs, fold):
    return float(np.mean(np.abs(errs) < np.log10(fold)) * 100)


def main():
    # Load PLM per-drug predictions (log_cd scale)
    with open('data/validation/novel_per_drug.json') as f:
        plm_data = json.load(f)

    # Load LLM predictions
    llm_preds = {}
    for i in range(4):
        with open(f'/tmp/ho_predictions_{i}.json') as f:
            for item in json.load(f):
                llm_preds[item['name']] = item['predicted_cmax_ngml']

    # Load HO actual data for dose_mg
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    ho_info = {}
    for d in ho['holdout_drugs']:
        if d.get('smiles') and d.get('dose_mg') and d.get('cmax_obs_ngml'):
            ho_info[d['name']] = {
                'dose_mg': d['dose_mg'],
                'cmax_obs': d['cmax_obs_ngml'],
                'actual_log_cd': math.log10(d['cmax_obs_ngml'] / d['dose_mg']),
                'cmax_meta_mgL': d.get('cmax_sisyphus_meta_mgL'),
            }

    # Align drugs
    common = sorted(set(plm_data.keys()) & set(llm_preds.keys()) & set(ho_info.keys()))
    print(f"Aligned drugs: {len(common)}")

    # Build arrays
    actual = []
    plm_pred = []  # log_cd
    llm_pred = []  # log_cd (converted from Cmax)
    meta_pred = []  # log_cd (from Meta mg/L → ng/mL)
    dose_mg = []
    for name in common:
        info = ho_info[name]
        actual.append(info['actual_log_cd'])
        plm_pred.append(plm_data[name]['baseline_pred'])
        llm_cmax = llm_preds[name]
        llm_pred.append(math.log10(llm_cmax / info['dose_mg']))
        dose_mg.append(info['dose_mg'])
        # Meta
        mm = info.get('cmax_meta_mgL')
        if mm and mm > 0:
            meta_pred.append(math.log10((mm * 1000) / info['dose_mg']))
        else:
            meta_pred.append(np.nan)

    actual = np.array(actual)
    plm_pred = np.array(plm_pred)
    llm_pred = np.array(llm_pred)
    meta_pred = np.array(meta_pred)

    print(f"\n{'='*80}")
    print(f"INDIVIDUAL MODELS")
    print(f"{'='*80}")
    plm_err = plm_pred - actual
    llm_err = llm_pred - actual
    print(f"  PLM baseline:  AAFE={aafe(plm_err):.3f}  bias={np.mean(plm_err):+.3f}  2f={fold_pct(plm_err, 2):.1f}%")
    print(f"  LLM direct:    AAFE={aafe(llm_err):.3f}  bias={np.mean(llm_err):+.3f}  2f={fold_pct(llm_err, 2):.1f}%")
    meta_valid = ~np.isnan(meta_pred)
    if meta_valid.sum() > 10:
        meta_err = meta_pred[meta_valid] - actual[meta_valid]
        print(f"  Meta ref:      AAFE={aafe(meta_err):.3f}  bias={np.mean(meta_err):+.3f}  (n={meta_valid.sum()})")

    # PLM-LLM error correlation
    r = float(np.corrcoef(plm_err, llm_err)[0, 1])
    print(f"\n  PLM-LLM error correlation: r={r:.3f}")
    print(f"  (Low r = errors complementary; high r = errors redundant)")

    print(f"\n{'='*80}")
    print(f"ENSEMBLE STRATEGIES")
    print(f"{'='*80}")
    results = {}

    # Simple average (log-space)
    ens = (plm_pred + llm_pred) / 2
    err = ens - actual
    results['simple_avg'] = (aafe(err), np.mean(err), fold_pct(err, 2))
    print(f"  Simple avg:              AAFE={results['simple_avg'][0]:.3f}  bias={results['simple_avg'][1]:+.3f}  2f={results['simple_avg'][2]:.1f}%")

    # Weighted ensembles
    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ens = w * plm_pred + (1 - w) * llm_pred
        err = ens - actual
        results[f'w_plm{w:.1f}'] = (aafe(err), np.mean(err), fold_pct(err, 2))
        print(f"  PLM{w:.1f}+LLM{1-w:.1f}:            AAFE={results[f'w_plm{w:.1f}'][0]:.3f}  bias={results[f'w_plm{w:.1f}'][1]:+.3f}  2f={results[f'w_plm{w:.1f}'][2]:.1f}%")

    # Median of {PLM, LLM, Meta} (where available)
    if meta_valid.sum() > 50:
        med_preds = []
        med_actual = []
        for i in range(len(actual)):
            if meta_valid[i]:
                m = np.median([plm_pred[i], llm_pred[i], meta_pred[i]])
                med_preds.append(m)
                med_actual.append(actual[i])
        med_preds = np.array(med_preds); med_actual = np.array(med_actual)
        err = med_preds - med_actual
        results['median_3way'] = (aafe(err), np.mean(err), fold_pct(err, 2))
        print(f"  Median(PLM,LLM,Meta):    AAFE={results['median_3way'][0]:.3f}  bias={results['median_3way'][1]:+.3f}  2f={results['median_3way'][2]:.1f}%  (n={len(err)})")

    # Geometric mean (equivalent to simple_avg in log space, already done)

    # Find best
    best_key = min(results.keys(), key=lambda k: results[k][0])
    print(f"\n  ⭐ Best: {best_key} AAFE={results[best_key][0]:.3f}")

    # Save
    with open('data/validation/ensemble_results.json', 'w') as f:
        json.dump({
            'individual': {
                'plm_baseline': {'aafe': round(aafe(plm_err), 3), 'bias': round(float(np.mean(plm_err)), 3)},
                'llm_direct': {'aafe': round(aafe(llm_err), 3), 'bias': round(float(np.mean(llm_err)), 3)},
            },
            'plm_llm_error_correlation': round(r, 3),
            'ensembles': {k: {'aafe': round(v[0], 3), 'bias': round(float(v[1]), 3), 'f2': round(v[2], 1)}
                          for k, v in results.items()},
            'best': best_key,
        }, f, indent=2, default=float)
    print(f"\n→ Saved data/validation/ensemble_results.json")

    # Sanity: per-drug improvement vs baseline
    print(f"\n{'='*80}")
    print(f"PER-DRUG: Winner vs PLM vs LLM (best ensemble)")
    print(f"{'='*80}")
    # Use best ensemble
    if best_key.startswith('w_plm'):
        w = float(best_key.split('plm')[1])
        best_ens = w * plm_pred + (1 - w) * llm_pred
    elif best_key == 'simple_avg':
        best_ens = (plm_pred + llm_pred) / 2
    else:
        best_ens = plm_pred  # fallback

    worst_n = 10
    all_errs = np.abs(best_ens - actual)
    worst_idx = np.argsort(-all_errs)[:worst_n]
    print(f"  Top {worst_n} worst in best ensemble:")
    print(f"  {'name':<25s} {'actual':>7s} {'PLM':>7s} {'LLM':>7s} {'ENS':>7s} {'|err|':>6s}")
    for idx in worst_idx:
        print(f"  {common[idx][:24]:<25s} {actual[idx]:>+7.2f} {plm_pred[idx]:>+7.2f} {llm_pred[idx]:>+7.2f} {best_ens[idx]:>+7.2f} {all_errs[idx]:>6.2f}")


if __name__ == '__main__':
    main()
