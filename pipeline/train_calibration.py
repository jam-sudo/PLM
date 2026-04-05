"""
Training-set LLM calibration.

1. Aggregate LLM predictions on 801 training drugs
2. Measure LLM bias pattern on training
3. Fit calibrator (constant offset, isotonic, linear)
4. Apply to HO predictions (Round 2 analogical, baseline 2.126)
5. Measure new HO AAFE
"""

import json, math
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression


def main():
    # Load training full info
    with open('/tmp/train_full.json') as f: train_full = json.load(f)
    train_map = {d['smiles']: d for d in train_full}
    print(f"Training drugs: {len(train_full)}")

    # Load all training predictions
    train_preds = {}
    for i in range(8):
        path = f'/tmp/train_pred_{i}.json'
        try:
            with open(path) as f:
                for item in json.load(f):
                    smi = item.get('smiles')
                    cmax = item.get('predicted_cmax_ngml')
                    if smi and cmax and cmax > 0:
                        train_preds[smi] = cmax
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  Error {path}: {e}")

    print(f"LLM predictions on training: {len(train_preds)}")

    # Compute LLM log_cd predictions vs actual
    llm_log_cd = []
    actual_log_cd = []
    names = []
    for d in train_full:
        smi = d['smiles']
        if smi not in train_preds: continue
        pred_log_cd = math.log10(train_preds[smi] / d['dose_mg'])
        llm_log_cd.append(pred_log_cd)
        actual_log_cd.append(d['actual_log_cd'])
        names.append(d['name'])

    llm_log_cd = np.array(llm_log_cd)
    actual_log_cd = np.array(actual_log_cd)
    residuals = llm_log_cd - actual_log_cd

    print(f"\n{'='*80}")
    print(f"Training set LLM calibration")
    print(f"{'='*80}")
    print(f"  N = {len(llm_log_cd)}")
    print(f"  Mean bias (llm - actual): {np.mean(residuals):+.3f}")
    print(f"  Std residuals: {np.std(residuals):.3f}")
    print(f"  Median residuals: {np.median(residuals):+.3f}")
    print(f"  AAFE on training: {10**np.mean(np.abs(residuals)):.3f}")
    print(f"  2-fold%: {np.mean(np.abs(residuals) < np.log10(2))*100:.1f}%")

    # Load HO predictions (Round 2 analogical) from saved results
    with open('data/validation/llm_cot_results.json') as f: cot = json.load(f)
    ho_preds = cot['rounds']['round_2']['preds']

    with open('data/validation/holdout_definition.json') as f: ho_data = json.load(f)
    ho_info = {d['name']: {'dose_mg': d['dose_mg'],
                            'actual_log_cd': math.log10(d['cmax_obs_ngml']/d['dose_mg'])}
               for d in ho_data['holdout_drugs']
               if d.get('smiles') and d.get('dose_mg') and d.get('cmax_obs_ngml')}

    # HO baseline (uncalibrated Round 2)
    ho_llm_log_cd = []
    ho_actual = []
    for name, info in ho_info.items():
        if name not in ho_preds: continue
        ho_llm_log_cd.append(math.log10(ho_preds[name] / info['dose_mg']))
        ho_actual.append(info['actual_log_cd'])
    ho_llm_log_cd = np.array(ho_llm_log_cd)
    ho_actual = np.array(ho_actual)
    ho_errs_raw = ho_llm_log_cd - ho_actual
    raw_aafe = 10**np.mean(np.abs(ho_errs_raw))
    print(f"\n  HO baseline (Round 2): AAFE={raw_aafe:.3f}, bias={np.mean(ho_errs_raw):+.3f}")

    # Calibration strategies
    print(f"\n{'='*80}")
    print(f"CALIBRATION STRATEGIES")
    print(f"{'='*80}")

    # Strategy 1: Constant offset (mean bias)
    offset_mean = np.mean(residuals)
    cal1_pred = ho_llm_log_cd - offset_mean
    cal1_errs = cal1_pred - ho_actual
    print(f"  Strategy 1 (subtract mean bias {offset_mean:+.3f}): "
          f"AAFE={10**np.mean(np.abs(cal1_errs)):.3f}, bias={np.mean(cal1_errs):+.3f}")

    # Strategy 2: Median offset
    offset_med = np.median(residuals)
    cal2_pred = ho_llm_log_cd - offset_med
    cal2_errs = cal2_pred - ho_actual
    print(f"  Strategy 2 (subtract median bias {offset_med:+.3f}): "
          f"AAFE={10**np.mean(np.abs(cal2_errs)):.3f}, bias={np.mean(cal2_errs):+.3f}")

    # Strategy 3: Isotonic regression (LLM_pred, actual)
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(llm_log_cd, actual_log_cd)
    cal3_pred = iso.predict(ho_llm_log_cd)
    cal3_errs = cal3_pred - ho_actual
    print(f"  Strategy 3 (isotonic): AAFE={10**np.mean(np.abs(cal3_errs)):.3f}, bias={np.mean(cal3_errs):+.3f}")

    # Strategy 4: Linear regression
    lin = LinearRegression()
    lin.fit(llm_log_cd.reshape(-1,1), actual_log_cd)
    cal4_pred = lin.predict(ho_llm_log_cd.reshape(-1,1))
    cal4_errs = cal4_pred - ho_actual
    print(f"  Strategy 4 (linear a={lin.coef_[0]:.3f}, b={lin.intercept_:+.3f}): "
          f"AAFE={10**np.mean(np.abs(cal4_errs)):.3f}, bias={np.mean(cal4_errs):+.3f}")

    # Strategy 5: Magnitude-dependent calibration (split by log_cd range)
    # Bin by LLM pred, compute per-bin bias
    bin_edges = np.percentile(llm_log_cd, [0, 25, 50, 75, 100])
    bin_biases = []
    for i in range(4):
        mask = (llm_log_cd >= bin_edges[i]) & (llm_log_cd <= bin_edges[i+1])
        bin_biases.append(np.mean(residuals[mask]) if mask.sum() > 0 else 0)
    print(f"  Binned biases (q0-q4): {[f'{b:+.3f}' for b in bin_biases]}")

    cal5_pred = []
    for p in ho_llm_log_cd:
        # Find bin
        idx = np.searchsorted(bin_edges[1:], p, side='right')
        idx = min(idx, 3)
        cal5_pred.append(p - bin_biases[idx])
    cal5_pred = np.array(cal5_pred)
    cal5_errs = cal5_pred - ho_actual
    print(f"  Strategy 5 (binned calibration): AAFE={10**np.mean(np.abs(cal5_errs)):.3f}, bias={np.mean(cal5_errs):+.3f}")

    # Summary
    strategies = {
        'raw (no calibration)': 10**np.mean(np.abs(ho_errs_raw)),
        'constant mean offset': 10**np.mean(np.abs(cal1_errs)),
        'constant median offset': 10**np.mean(np.abs(cal2_errs)),
        'isotonic': 10**np.mean(np.abs(cal3_errs)),
        'linear': 10**np.mean(np.abs(cal4_errs)),
        'binned (4 bins)': 10**np.mean(np.abs(cal5_errs)),
    }

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    for k, v in sorted(strategies.items(), key=lambda x: x[1]):
        print(f"  {k:<30s}: AAFE={v:.3f}")

    best = min(strategies.items(), key=lambda x: x[1])
    print(f"\n  ⭐ Best: {best[0]} → AAFE={best[1]:.3f}")
    print(f"  vs uncalibrated Round 2: {raw_aafe:.3f}")
    print(f"  vs Sisyphus Meta (SOTA): 2.283")

    # Save
    results = {
        'n_training': len(llm_log_cd),
        'training_bias_mean': round(float(np.mean(residuals)), 3),
        'training_bias_median': round(float(np.median(residuals)), 3),
        'training_aafe': round(10**np.mean(np.abs(residuals)), 3),
        'calibration_strategies': {k: round(v, 3) for k, v in strategies.items()},
        'best_strategy': best[0],
        'best_aafe': round(best[1], 3),
        'uncalibrated_aafe': round(raw_aafe, 3),
    }
    with open('data/validation/train_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved data/validation/train_calibration_results.json")


if __name__ == '__main__':
    main()
