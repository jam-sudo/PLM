"""
Training 5-round std-adaptive calibration (strictly no leakage, no cherry-picking).

Pipeline:
1. Load 3 training rounds: R1 (physiological), R2 (analogical - already have), R3 (label recall)
2. Compute per-drug training stats: std(3 rounds), geomean, residual vs actual
3. Fit calibrator on TRAINING ONLY: residual = a + b * std
4. Apply to HO using HO's own 3-round std + 3-round geomean
5. Evaluate: does training-derived std-adaptive calibrator reach 1.978?

No HO labels used in calibrator fitting. No CV on HO.
"""

import json, math
import numpy as np


def load_round_batches(round_name, n_batches=8):
    preds = {}
    for b in range(n_batches):
        path = f'/tmp/train_{round_name}_b{b}.json'
        try:
            with open(path) as f:
                for item in json.load(f):
                    smi = item.get('smiles')
                    cmax = item.get('predicted_cmax_ngml')
                    if smi and cmax and cmax > 0:
                        preds[smi] = cmax
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  Error {path}: {e}")
    return preds


def main():
    # Load training drugs with actual log_cd
    with open('/tmp/train_full.json') as f: train_full = json.load(f)

    # Load 3 training rounds
    print("Loading training rounds...")
    train_r1 = load_round_batches('r1')
    print(f"  R1 (physiological): {len(train_r1)}")

    # R2 (analogical) = existing train predictions
    with open('data/llm_extracted/llm_train_predictions.json') as f:
        train_r2_data = json.load(f)
    train_r2 = {smi: d['predicted_cmax_ngml'] for smi, d in train_r2_data.items()}
    print(f"  R2 (analogical): {len(train_r2)}")

    train_r3 = load_round_batches('r3')
    print(f"  R3 (label recall): {len(train_r3)}")

    # Compute per-drug training stats
    train_stats = []
    for d in train_full:
        smi = d['smiles']
        preds = []
        for r in [train_r1, train_r2, train_r3]:
            if smi in r:
                preds.append(math.log10(r[smi] / d['dose_mg']))
        if len(preds) < 3: continue
        geomean = float(np.mean(preds))
        std = float(np.std(preds))
        actual = d['actual_log_cd']
        train_stats.append({
            'smi': smi, 'name': d['name'],
            'std': std, 'geomean': geomean,
            'actual': actual,
            'residual': geomean - actual,
        })
    print(f"\nTraining drugs with 3 rounds: {len(train_stats)}")

    stds = np.array([s['std'] for s in train_stats])
    residuals = np.array([s['residual'] for s in train_stats])
    geomeans = np.array([s['geomean'] for s in train_stats])
    actuals = np.array([s['actual'] for s in train_stats])

    print(f"Training 3-round stats:")
    print(f"  Geomean: mean={np.mean(geomeans):+.3f}, std={np.std(geomeans):.3f}")
    print(f"  Actual:  mean={np.mean(actuals):+.3f}, std={np.std(actuals):.3f}")
    print(f"  Residual (geomean - actual): mean={np.mean(residuals):+.3f}, median={np.median(residuals):+.3f}")
    print(f"  Std dist: 5%={np.percentile(stds,5):.3f}, median={np.median(stds):.3f}, 95%={np.percentile(stds,95):.3f}")

    # Fit calibrator: residual = a + b * std (on training)
    # Use linear regression
    X = np.column_stack([np.ones(len(stds)), stds])
    coefs, _, _, _ = np.linalg.lstsq(X, residuals, rcond=None)
    a, b = coefs
    print(f"\nCalibrator fit (TRAINING only): residual = {a:+.3f} + {b:+.3f} * std")

    # Quality check: training CV AAFE after calibration
    train_pred_cal = geomeans - (a + b * stds)
    train_errs_before = geomeans - actuals
    train_errs_after = train_pred_cal - actuals
    print(f"\nTraining performance:")
    print(f"  Before cal: AAFE={10**np.mean(np.abs(train_errs_before)):.3f}  bias={np.mean(train_errs_before):+.3f}")
    print(f"  After cal:  AAFE={10**np.mean(np.abs(train_errs_after)):.3f}  bias={np.mean(train_errs_after):+.3f}")

    # Also try: constant offset only (baseline)
    a_const = np.mean(residuals)
    train_pred_const = geomeans - a_const
    train_errs_const = train_pred_const - actuals
    print(f"  Const offset baseline: a={a_const:+.3f}, AAFE={10**np.mean(np.abs(train_errs_const)):.3f}")

    # Apply to HO
    print(f"\n{'='*80}\nHO EVALUATION\n{'='*80}")
    with open('data/validation/llm_cot_results.json') as f: cot = json.load(f)
    rounds_data = {f'round_{r}': cot['rounds'][f'round_{r}']['preds'] for r in [1,2,3]}
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    ho_info = {d['name']: d for d in ho['holdout_drugs'] if d.get('cmax_obs_ngml')}

    ho_data = []
    for name, info in ho_info.items():
        dose = info['dose_mg']
        actual = math.log10(info['cmax_obs_ngml']/dose)
        preds = []
        for r in [1,2,3]:
            if name in rounds_data[f'round_{r}']:
                preds.append(math.log10(rounds_data[f'round_{r}'][name]/dose))
        if len(preds) < 3: continue
        ho_data.append({
            'name': name,
            'std': float(np.std(preds)),
            'geomean': float(np.mean(preds)),
            'actual': actual,
        })
    print(f"HO drugs with 3 rounds: {len(ho_data)}")
    ho_stds = np.array([d['std'] for d in ho_data])
    ho_geomeans = np.array([d['geomean'] for d in ho_data])
    ho_actuals = np.array([d['actual'] for d in ho_data])

    # Apply calibrators trained on TRAINING
    print(f"\nApplying training-derived calibrators:")
    # Baseline
    ho_err = ho_geomeans - ho_actuals
    print(f"  Raw geomean (no cal):  AAFE={10**np.mean(np.abs(ho_err)):.3f}  bias={np.mean(ho_err):+.3f}")

    # Constant offset (mean residual from training)
    ho_err_const = ho_geomeans - a_const - ho_actuals
    print(f"  Const offset {a_const:+.3f} (train mean): AAFE={10**np.mean(np.abs(ho_err_const)):.3f}  bias={np.mean(ho_err_const):+.3f}")

    # Const offset (median residual)
    a_med = np.median(residuals)
    ho_err_med = ho_geomeans - a_med - ho_actuals
    print(f"  Const offset {a_med:+.3f} (train median): AAFE={10**np.mean(np.abs(ho_err_med)):.3f}  bias={np.mean(ho_err_med):+.3f}")

    # Linear std-adaptive (training-derived)
    ho_err_adaptive = ho_geomeans - (a + b * ho_stds) - ho_actuals
    print(f"  Std-adaptive (a={a:+.3f}, b={b:+.3f}): AAFE={10**np.mean(np.abs(ho_err_adaptive)):.3f}  bias={np.mean(ho_err_adaptive):+.3f}")

    # Robust: median-based + slope
    # Fit on (std bins, median residual per bin)
    print(f"\nBinned std-adaptive calibration (training-derived):")
    n_bins = 5
    bin_edges = np.percentile(stds, np.linspace(0, 100, n_bins+1))
    bin_medians = []
    bin_centers = []
    for i in range(n_bins):
        mask = (stds >= bin_edges[i]) & (stds <= bin_edges[i+1])
        if mask.sum() == 0:
            bin_medians.append(0); bin_centers.append((bin_edges[i] + bin_edges[i+1])/2)
            continue
        bin_medians.append(np.median(residuals[mask]))
        bin_centers.append(np.median(stds[mask]))
        print(f"  Bin[{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}]: median_resid={bin_medians[-1]:+.3f}, n={mask.sum()}")

    # Apply binned calibration via linear interpolation
    def bin_offset(s):
        return np.interp(s, bin_centers, bin_medians)
    ho_offsets = np.array([bin_offset(s) for s in ho_stds])
    ho_err_binned = ho_geomeans - ho_offsets - ho_actuals
    print(f"  Binned cal applied to HO: AAFE={10**np.mean(np.abs(ho_err_binned)):.3f}  bias={np.mean(ho_err_binned):+.3f}")

    # Summary
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(f"Prior (training-single-round cal): 2.087")
    print(f"HO CV std-adaptive (uses HO labels): 1.978")
    print(f"Training 3-round const (no leakage):      {10**np.mean(np.abs(ho_err_const)):.3f}")
    print(f"Training 3-round median const (no leak):  {10**np.mean(np.abs(ho_err_med)):.3f}")
    print(f"Training 3-round std-adaptive (no leak):  {10**np.mean(np.abs(ho_err_adaptive)):.3f}")
    print(f"Training 3-round binned (no leak):        {10**np.mean(np.abs(ho_err_binned)):.3f}")

    # Save
    results = {
        'training_stats': {
            'n': len(train_stats),
            'residual_mean': round(float(np.mean(residuals)), 3),
            'residual_median': round(float(np.median(residuals)), 3),
            'residual_std': round(float(np.std(residuals)), 3),
            'std_median': round(float(np.median(stds)), 3),
        },
        'calibrator_coefs': {
            'const_mean': round(float(a_const), 3),
            'const_median': round(float(a_med), 3),
            'linear_a': round(float(a), 3),
            'linear_b': round(float(b), 3),
        },
        'ho_results': {
            'raw_geomean': round(10**np.mean(np.abs(ho_err)), 3),
            'const_offset': round(10**np.mean(np.abs(ho_err_const)), 3),
            'median_offset': round(10**np.mean(np.abs(ho_err_med)), 3),
            'std_adaptive_linear': round(10**np.mean(np.abs(ho_err_adaptive)), 3),
            'std_adaptive_binned': round(10**np.mean(np.abs(ho_err_binned)), 3),
        },
    }
    with open('data/validation/train_std_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved data/validation/train_std_calibration_results.json")


if __name__ == '__main__':
    main()
