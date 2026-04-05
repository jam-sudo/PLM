"""
Drug similarity-based local calibration.

For each HO drug:
1. Find top-k most Tanimoto-similar training drugs
2. Retrieve their LLM residuals (llm_pred_log_cd - actual_log_cd)
3. Compute similarity-weighted average residual = local bias estimate
4. Apply: HO_calibrated = HO_pred - local_bias

Expected: targets drug-class-specific biases that global offset can't capture.
"""

import json, math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def morgan_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def main():
    # Load training full info + predictions
    with open('/tmp/train_full.json') as f: train_full = json.load(f)
    train_preds = {}
    for i in range(8):
        with open(f'/tmp/train_pred_{i}.json') as f:
            for item in json.load(f):
                smi = item.get('smiles')
                cmax = item.get('predicted_cmax_ngml')
                if smi and cmax and cmax > 0:
                    train_preds[smi] = cmax

    # Compute training residuals + FPs
    train_data = []
    for d in train_full:
        smi = d['smiles']
        if smi not in train_preds: continue
        pred_lcd = math.log10(train_preds[smi] / d['dose_mg'])
        residual = pred_lcd - d['actual_log_cd']
        fp = morgan_fp(smi)
        if fp is None: continue
        train_data.append({
            'smi': smi, 'fp': fp,
            'residual': residual,
            'actual_lcd': d['actual_log_cd'],
            'pred_lcd': pred_lcd,
            'name': d['name'],
        })
    print(f"Training data: {len(train_data)}")

    # HO data
    with open('data/validation/llm_cot_results.json') as f: cot = json.load(f)
    ho_preds = cot['rounds']['round_2']['preds']
    with open('data/validation/holdout_definition.json') as f: ho_data = json.load(f)
    ho_info = {d['name']: {'dose_mg': d['dose_mg'], 'smiles': d['smiles'],
                            'cmax_obs': d['cmax_obs_ngml'],
                            'actual_lcd': math.log10(d['cmax_obs_ngml']/d['dose_mg'])}
               for d in ho_data['holdout_drugs']
               if d.get('smiles') and d.get('dose_mg') and d.get('cmax_obs_ngml')}

    # Precompute training FPs array
    train_fps = [d['fp'] for d in train_data]

    # For each HO drug, find top-k neighbors and compute local bias
    results = {}
    for k in [5, 10, 20, 50]:
        errs = []
        local_biases = []
        for name, info in ho_info.items():
            if name not in ho_preds: continue
            q_fp = morgan_fp(info['smiles'])
            if q_fp is None: continue
            pred_cmax = ho_preds[name]
            pred_lcd = math.log10(pred_cmax / info['dose_mg'])

            # Find top-k similar training drugs
            sims = DataStructs.BulkTanimotoSimilarity(q_fp, train_fps)
            sims_array = np.array(sims)
            top_idx = np.argsort(-sims_array)[:k]
            top_sims = sims_array[top_idx]
            top_residuals = np.array([train_data[i]['residual'] for i in top_idx])

            # Similarity-weighted average residual
            if top_sims.sum() < 1e-6:
                local_bias = 0.0
            else:
                local_bias = float(np.sum(top_sims * top_residuals) / top_sims.sum())
            local_biases.append(local_bias)

            calibrated_lcd = pred_lcd - local_bias
            err = calibrated_lcd - info['actual_lcd']
            errs.append(err)

        errs = np.array(errs)
        aafe = 10 ** np.mean(np.abs(errs))
        bias = np.mean(errs)
        f2 = np.mean(np.abs(errs) < np.log10(2)) * 100
        avg_lb = float(np.mean(np.abs(local_biases)))
        max_lb = float(np.max(np.abs(local_biases)))
        print(f"  k={k:2d}: AAFE={aafe:.3f}  bias={bias:+.3f}  2f={f2:.1f}%  mean|local_bias|={avg_lb:.3f}  max={max_lb:.3f}")
        results[f'k{k}'] = {'aafe': round(aafe,3), 'bias': round(bias,3), 'f2': round(f2,1)}

    # Also test: top-k with Tanimoto threshold (only use highly similar)
    print(f"\nWith Tanimoto threshold filter (only similar neighbors):")
    for k in [10, 20]:
        for thresh in [0.3, 0.5]:
            errs = []
            fallback_count = 0
            for name, info in ho_info.items():
                if name not in ho_preds: continue
                q_fp = morgan_fp(info['smiles'])
                if q_fp is None: continue
                pred_cmax = ho_preds[name]
                pred_lcd = math.log10(pred_cmax / info['dose_mg'])

                sims = DataStructs.BulkTanimotoSimilarity(q_fp, train_fps)
                sims_array = np.array(sims)
                top_idx = np.argsort(-sims_array)[:k]
                top_sims = sims_array[top_idx]
                # Filter by threshold
                mask = top_sims >= thresh
                if mask.sum() == 0:
                    # Fallback to global median
                    local_bias = 0.019
                    fallback_count += 1
                else:
                    residuals = np.array([train_data[i]['residual'] for i in top_idx])[mask]
                    weights = top_sims[mask]
                    local_bias = float(np.sum(weights * residuals) / weights.sum())

                calibrated_lcd = pred_lcd - local_bias
                err = calibrated_lcd - info['actual_lcd']
                errs.append(err)

            errs = np.array(errs)
            aafe = 10 ** np.mean(np.abs(errs))
            f2 = np.mean(np.abs(errs) < np.log10(2)) * 100
            print(f"  k={k} thresh={thresh}: AAFE={aafe:.3f}  2f={f2:.1f}%  fallback={fallback_count}")

    # Save
    with open('data/validation/similarity_calibration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved data/validation/similarity_calibration_results.json")


if __name__ == '__main__':
    main()
