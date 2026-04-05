"""
MoLFormer-augmented XGBoost experiment.

Compares:
  E0: Baseline (Morgan FP + PhysChem + TDC + microPBPK + conditions + dose)
  E1: + MoLFormer 768-dim (concatenated)
  E2: MoLFormer ONLY + conditions + dose (no Morgan FP)
  E3: MoLFormer + PhysChem + TDC + conditions + dose (drop Morgan FP)
"""

import json, math, warnings, sys
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import xgboost as xgb
warnings.filterwarnings('ignore')

sys.path.insert(0, 'pipeline')
from llm_enriched_experiment import (
    smi_to_ik, build_sample, CANONICAL_COND, normalize_condition, XGB_PARAMS,
    smiles_to_fp, smiles_to_physchem,
)
from ho_diagnostic import build_training


def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return Chem.MolToSmiles(mol)


def aafe(pred, actual):
    return float(10 ** np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))


def fold_pct(pred, actual, fold=2.0):
    err = np.abs(np.asarray(pred) - np.asarray(actual))
    return float(np.mean(err < np.log10(fold)) * 100)


def safe_arr(X):
    return np.where(np.isinf(X), np.nan, X).astype(np.float32)


def fit_eval(X_tr, Y_tr, W_tr, X_ho, Y_ho, label=""):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(safe_arr(X_tr), Y_tr, sample_weight=W_tr)
    pred = m.predict(safe_arr(X_ho))
    a = aafe(pred, Y_ho)
    bias = float(np.mean(pred - Y_ho))
    f2 = fold_pct(pred, Y_ho, 2.0)
    f3 = fold_pct(pred, Y_ho, 3.0)
    print(f"  {label:<40s}  HO AAFE={a:.3f}  bias={bias:+.3f}  2f={f2:.1f}%  3f={f3:.1f}%")
    return {'aafe': round(a, 3), 'bias': round(bias, 3), 'f2': round(f2, 1), 'f3': round(f3, 1), 'pred': pred.tolist()}


def augment_molformer(X, smiles_list, molformer_emb, dim=768):
    """Append MoLFormer 768-dim embedding to each row."""
    extra = []
    for smi in smiles_list:
        can = canonicalize(smi)
        if can and can in molformer_emb:
            extra.append(np.array(molformer_emb[can], dtype=np.float32))
        else:
            extra.append(np.zeros(dim, dtype=np.float32))
    extra = np.array(extra, dtype=np.float32)
    return np.hstack([X, extra])


def build_molformer_only(smiles_list, doses, iks, conditions_list, tdc, molformer_emb):
    """Build feature vector = MoLFormer(768) + TDC(9) + microPBPK(6) + conditions(18) + log(dose)."""
    from llm_enriched_experiment import get_tdc_features, compute_micropbpk, build_condition_features
    X = []
    for smi, dose, ik, cond in zip(smiles_list, doses, iks, conditions_list):
        can = canonicalize(smi)
        if can and can in molformer_emb:
            mf = np.array(molformer_emb[can], dtype=np.float32)
        else:
            mf = np.zeros(768, dtype=np.float32)
        adme = get_tdc_features(ik, tdc)
        mpbpk = compute_micropbpk(ik, tdc)
        ld = np.float32(math.log10(dose))
        cond_feat = build_condition_features(cond)
        X.append(np.concatenate([mf, adme, mpbpk, [ld], cond_feat]))
    return np.array(X, dtype=np.float32)


def build_holdout_basic(holdout_drugs, tdc):
    """Standard HO features (4150-dim) + SMILES list + doses + iks."""
    X_ho, Y_ho, smi_ho, names, doses, iks = [], [], [], [], [], []
    for d in holdout_drugs:
        smi, dose, cmax = d.get('smiles'), d.get('dose_mg'), d.get('cmax_obs_ngml')
        if not smi or not dose or dose <= 0 or not cmax or cmax <= 0: continue
        ik = d.get('inchikey14', '')
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None: continue
        X_ho.append(s); Y_ho.append(math.log10(cmax / dose))
        smi_ho.append(smi); names.append(d['name']); doses.append(dose); iks.append(ik)
    return (np.array(X_ho, dtype=np.float32), np.array(Y_ho, dtype=np.float32),
            smi_ho, names, doses, iks)


def main():
    print("=" * 80)
    print("MoLFormer-Augmented XGBoost Experiment")
    print("=" * 80)

    # Load data
    with open('data/curated/tdc_adme_data.json') as f: tdc = json.load(f)
    with open('data/validation/holdout_definition.json') as f: ho_data = json.load(f)
    holdout_drugs = ho_data['holdout_drugs']
    ho_iks_14 = set(d['inchikey14'] for d in holdout_drugs)
    with open('data/curated/plm_dataset_v10_labels.json') as f: v10 = json.load(f)
    with open('data/llm_extracted/pk_llm_merged.json') as f: llm = json.load(f)
    with open('data/curated/molformer_embeddings.json') as f: mf_emb = json.load(f)
    print(f"Data loaded. MoLFormer embeddings: {len(mf_emb)}")

    # Build baseline training data (v10 + LLM median + agreement + conf weights)
    print("\nBuilding training data...")
    X_tr, Y_tr, g_tr, W_tr, smi_tr = build_training(v10, llm, ho_iks_14, tdc)
    X_ho, Y_ho, smi_ho, names_ho, doses_ho, iks_ho = build_holdout_basic(holdout_drugs, tdc)
    print(f"  Training: {len(Y_tr)} profiles, HO: {len(Y_ho)} drugs")

    # Coverage check
    tr_coverage = sum(1 for s in smi_tr if canonicalize(s) and canonicalize(s) in mf_emb) / len(smi_tr)
    ho_coverage = sum(1 for s in smi_ho if canonicalize(s) and canonicalize(s) in mf_emb) / len(smi_ho)
    print(f"  MoLFormer coverage: train={tr_coverage:.1%}  HO={ho_coverage:.1%}")

    results = {}
    print("\n" + "=" * 80)
    print("Experiments")
    print("=" * 80)

    # E0: Baseline
    results['E0_baseline'] = fit_eval(X_tr, Y_tr, W_tr, X_ho, Y_ho, 'E0: Baseline (Morgan+PhysChem+...)')

    # E1: Baseline + MoLFormer 768
    X_tr_e1 = augment_molformer(X_tr, smi_tr, mf_emb)
    X_ho_e1 = augment_molformer(X_ho, smi_ho, mf_emb)
    print(f"  E1 feature dim: {X_tr_e1.shape[1]}")
    results['E1_baseline_plus_molformer'] = fit_eval(X_tr_e1, Y_tr, W_tr, X_ho_e1, Y_ho,
                                                      'E1: Baseline + MoLFormer 768')

    # E2: MoLFormer only (+ TDC + microPBPK + conditions + dose, no Morgan FP, no PhysChem)
    # Need to reconstruct training samples for this
    print("\nBuilding E2/E3 alt feature sets...")
    train_smis, train_doses, train_iks, train_conds = [], [], [], []
    for p in v10:
        smi, dose, ik, lcd = p.get('smiles'), p.get('dose_mg'), p.get('ik'), p.get('log_cd')
        if not smi or not dose or dose <= 0 or lcd is None: continue
        s_check = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s_check is None: continue
        train_smis.append(smi); train_doses.append(dose); train_iks.append(ik)
        train_conds.append(CANONICAL_COND)

    # Add LLM median groups (replicate build_training logic)
    from collections import defaultdict
    v10_mean_lcd = defaultdict(list)
    for p in v10:
        ik, lcd = p.get('ik'), p.get('log_cd')
        if ik and lcd is not None: v10_mean_lcd[ik].append(lcd)
    v10_mean = {ik: float(np.mean(l)) for ik, l in v10_mean_lcd.items()}

    grouped = defaultdict(list)
    for t in llm:
        if not t.get('smiles'): continue
        smi, dose, cmax = t['smiles'], t.get('dose_mg', 0), t.get('cmax_ng_ml', 0)
        if not dose or dose <= 0 or not cmax or cmax <= 0: continue
        ik = smi_to_ik(smi)
        if not ik or ik in ho_iks_14: continue
        log_cd = math.log10(cmax / dose)
        if log_cd < -3 or log_cd > 3: continue
        key = (ik, normalize_condition(t.get('route', 'oral')),
               normalize_condition(t.get('dose_schedule', 'single_dose')),
               normalize_condition(t.get('food', 'not_specified')),
               normalize_condition(t.get('population', 'healthy_adult')))
        grouped[key].append({'smi': smi, 'dose': dose, 'log_cd': log_cd,
                             'conf': t.get('confidence', 'medium')})
    for key, entries in grouped.items():
        log_cds = [e['log_cd'] for e in entries]
        med = float(np.median(log_cds))
        if key[0] in v10_mean and abs(med - v10_mean[key[0]]) > 1.0: continue
        doses = [e['dose'] for e in entries]
        smi = entries[0]['smi']
        cond = {'route': key[1], 'schedule': key[2], 'food': key[3],
                'formulation': 'tablet', 'population': key[4]}
        train_smis.append(smi); train_doses.append(float(np.median(doses)))
        train_iks.append(key[0]); train_conds.append(cond)

    # Sanity: must match Y_tr length
    assert len(train_smis) == len(Y_tr), f"Mismatch: {len(train_smis)} vs {len(Y_tr)}"

    X_tr_mfonly = build_molformer_only(train_smis, train_doses, train_iks, train_conds, tdc, mf_emb)
    ho_conds = [CANONICAL_COND] * len(smi_ho)
    X_ho_mfonly = build_molformer_only(smi_ho, doses_ho, iks_ho, ho_conds, tdc, mf_emb)
    print(f"  MoLFormer-only feature dim: {X_tr_mfonly.shape[1]}")
    results['E2_molformer_only'] = fit_eval(X_tr_mfonly, Y_tr, W_tr, X_ho_mfonly, Y_ho,
                                             'E2: MoLFormer + TDC + cond + dose (no FP)')

    # E3: MoLFormer + PhysChem + TDC + conditions + dose (no Morgan FP)
    def build_no_fp(smis, doses, iks, conds, tdc, mf_emb):
        from llm_enriched_experiment import get_tdc_features, compute_micropbpk, build_condition_features
        X = []
        for smi, dose, ik, cond in zip(smis, doses, iks, conds):
            can = canonicalize(smi)
            mf = np.array(mf_emb[can], dtype=np.float32) if can and can in mf_emb else np.zeros(768, dtype=np.float32)
            pc = smiles_to_physchem(smi)
            adme = get_tdc_features(ik, tdc)
            mpbpk = compute_micropbpk(ik, tdc)
            ld = np.float32(math.log10(dose))
            cond_feat = build_condition_features(cond)
            X.append(np.concatenate([mf, pc, adme, mpbpk, [ld], cond_feat]))
        return np.array(X, dtype=np.float32)

    X_tr_e3 = build_no_fp(train_smis, train_doses, train_iks, train_conds, tdc, mf_emb)
    X_ho_e3 = build_no_fp(smi_ho, doses_ho, iks_ho, ho_conds, tdc, mf_emb)
    print(f"  E3 feature dim: {X_tr_e3.shape[1]}")
    results['E3_molformer_physchem_no_fp'] = fit_eval(X_tr_e3, Y_tr, W_tr, X_ho_e3, Y_ho,
                                                       'E3: MoLFormer + PhysChem + TDC (no FP)')

    # Save
    for k, v in results.items():
        v.pop('pred', None)  # don't save preds to summary
    with open('data/validation/molformer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n→ Saved data/validation/molformer_results.json")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<45s}  {'HO AAFE':>8s}  {'Δ':>7s}  {'bias':>7s}  {'2-fold':>7s}")
    print("-" * 85)
    for k, r in results.items():
        delta = r['aafe'] - results['E0_baseline']['aafe']
        print(f"{k:<45s}  {r['aafe']:>8.3f}  {delta:>+7.3f}  {r['bias']:>+7.3f}  {r['f2']:>6.1f}%")


if __name__ == '__main__':
    main()
