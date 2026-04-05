"""
HO Diagnostic: decompose PLM holdout error to diagnose next-step direction.

Reproduces the best-model training (drug-median LLM + agreement filter + conf
weights + condition features), predicts 97 holdout drugs, then performs:
  - Per-drug error table with descriptors + ionization + NN Tanimoto
  - Stratified error analysis (Tanimoto, MW, LogP, ionization)
  - Shared-vs-PLM-specific failure categorization (vs Sisyphus Meta)
  - PLM-Meta error correlation (noise-floor test)
  - Top-20 worst predictions
  - Cumulative error
  - Decision thresholds → Phase A (features) / B (data) / C (wrap) / fallback

Output: data/validation/ho_diagnostic.json + console report.
"""

import json, math, warnings
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import sys; sys.path.insert(0, 'pipeline')
from llm_enriched_experiment import (smi_to_ik, build_sample, CANONICAL_COND,
    normalize_condition, XGB_PARAMS)
import xgboost as xgb
warnings.filterwarnings('ignore')


def morgan_fp_2048(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def compute_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return {
        'mw': float(Descriptors.ExactMolWt(mol)),
        'logp': float(Descriptors.MolLogP(mol)),
        'tpsa': float(Descriptors.TPSA(mol)),
        'hbd': int(Descriptors.NumHDonors(mol)),
        'hba': int(Descriptors.NumHAcceptors(mol)),
        'rotb': int(Descriptors.NumRotatableBonds(mol)),
        'arom_rings': int(Descriptors.NumAromaticRings(mol)),
        'formal_charge': int(Chem.GetFormalCharge(mol)),
        'heavy_atoms': int(Descriptors.HeavyAtomCount(mol)),
    }


def classify_ionization(formal_charge):
    """SMILES-level ionization proxy. Limitation: physiological pKa not used."""
    if formal_charge > 0: return 'cation'
    if formal_charge < 0: return 'anion'
    return 'neutral_smi'


def build_training(v10, llm, ho_iks_14, tdc):
    """Reproduce best-model training: v10 + LLM drug-median + agreement filter + conf weights."""
    X_tr, Y_tr, g_tr, W_tr, smi_tr = [], [], [], [], []
    for p in v10:
        smi, dose, ik, lcd = p.get('smiles'), p.get('dose_mg'), p.get('ik'), p.get('log_cd')
        if not smi or not dose or dose <= 0 or lcd is None: continue
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None: continue
        X_tr.append(s); Y_tr.append(lcd); g_tr.append(ik); W_tr.append(1.0); smi_tr.append(smi)

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
        conf = t.get('confidence', 'medium')
        key = (ik, normalize_condition(t.get('route', 'oral')),
               normalize_condition(t.get('dose_schedule', 'single_dose')),
               normalize_condition(t.get('food', 'not_specified')),
               normalize_condition(t.get('population', 'healthy_adult')))
        grouped[key].append({'smi': smi, 'dose': dose, 'log_cd': log_cd, 'conf': conf})

    conf_w = {'high': 1.0, 'medium': 0.7, 'low': 0.3}
    for key, entries in grouped.items():
        log_cds = [e['log_cd'] for e in entries]
        med_lcd = float(np.median(log_cds))
        if key[0] in v10_mean and abs(med_lcd - v10_mean[key[0]]) > 1.0: continue
        doses = [e['dose'] for e in entries]
        w = float(np.mean([conf_w.get(e['conf'], 0.5) for e in entries]))
        conditions = {'route': key[1], 'schedule': key[2], 'food': key[3],
                      'formulation': 'tablet', 'population': key[4]}
        s = build_sample(entries[0]['smi'], float(np.median(doses)), key[0], conditions, tdc, True)
        if s is None: continue
        X_tr.append(s); Y_tr.append(med_lcd); g_tr.append(key[0])
        W_tr.append(w); smi_tr.append(entries[0]['smi'])

    return (np.array(X_tr, dtype=np.float32), np.array(Y_tr, dtype=np.float32),
            np.array(g_tr), np.array(W_tr, dtype=np.float32), smi_tr)


def bootstrap_mean_err(errs, n_boot=1000, seed=42):
    if len(errs) < 2: return None, None
    rng = np.random.default_rng(seed)
    means = [np.mean(rng.choice(errs, size=len(errs), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    print("=" * 80)
    print("HO DIAGNOSTIC — Phase 0")
    print("=" * 80)

    # Load
    with open('data/curated/tdc_adme_data.json') as f: tdc = json.load(f)
    with open('data/validation/holdout_definition.json') as f: ho_data = json.load(f)
    holdout_drugs = ho_data['holdout_drugs']
    ho_iks_14 = set(d['inchikey14'] for d in holdout_drugs)
    with open('data/curated/plm_dataset_v10_labels.json') as f: v10 = json.load(f)
    with open('data/llm_extracted/pk_llm_merged.json') as f: llm = json.load(f)

    # Train
    X_tr, Y_tr, g_tr, W_tr, smi_tr = build_training(v10, llm, ho_iks_14, tdc)
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    print(f"Training: {len(Y_tr)} profiles, {len(set(g_tr))} unique drugs")

    # Unique train FPs for Tanimoto
    seen = set(); train_fps = []
    for ik, smi in zip(g_tr, smi_tr):
        if ik in seen: continue
        seen.add(ik)
        fp = morgan_fp_2048(smi)
        if fp is not None: train_fps.append(fp)
    print(f"Training unique FPs: {len(train_fps)}")

    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, Y_tr, sample_weight=W_tr)

    # Predict HO
    per_drug = []
    for d in holdout_drugs:
        smi, dose, cmax = d.get('smiles'), d.get('dose_mg'), d.get('cmax_obs_ngml')
        if not smi or not dose or dose <= 0 or not cmax or cmax <= 0: continue
        ik = d.get('inchikey14', '')
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None: continue
        X_h = np.array([s], dtype=np.float32)
        X_h = np.where(np.isinf(X_h), np.nan, X_h)
        pred = float(m.predict(X_h)[0])
        actual = math.log10(cmax / dose)
        err = abs(pred - actual); signed = pred - actual

        desc = compute_descriptors(smi)
        if desc is None: continue
        ion = classify_ionization(desc['formal_charge'])

        q_fp = morgan_fp_2048(smi)
        nn_t = max(DataStructs.BulkTanimotoSimilarity(q_fp, train_fps)) if q_fp else None

        meta_mgL = d.get('cmax_sisyphus_meta_mgL')
        meta_err = None; meta_signed = None
        if meta_mgL and meta_mgL > 0:
            meta_log_cd = math.log10((meta_mgL * 1000) / dose)
            meta_err = abs(meta_log_cd - actual)
            meta_signed = meta_log_cd - actual

        per_drug.append({
            'name': d['name'], 'ik': ik, 'smiles': smi, 'dose_mg': dose,
            'cmax_obs_ngml': cmax, 'actual_log_cd': actual, 'pred_log_cd': pred,
            'abs_err': err, 'signed_err': signed, 'fold_err': 10 ** err,
            'meta_abs_err': meta_err, 'meta_signed_err': meta_signed,
            'nn_tanimoto': nn_t, 'ionization': ion, **desc,
        })

    aafe = 10 ** np.mean([r['abs_err'] for r in per_drug])
    print(f"\nHO AAFE: {aafe:.3f}  (N={len(per_drug)})")

    # ── Stratification ──
    def strat(label, bins, key):
        print(f"\n{label}:")
        print(f"  {'bucket':<20s} {'n':>4s} {'AAFE':>7s} {'bias':>7s} {'95% CI':>18s}")
        for lo, hi in bins:
            subset = [r for r in per_drug if r[key] is not None and lo <= r[key] < hi]
            if not subset: continue
            errs = [r['abs_err'] for r in subset]
            mean_err = np.mean(errs); mean_bias = np.mean([r['signed_err'] for r in subset])
            lo_ci, hi_ci = bootstrap_mean_err(errs)
            ci_str = f"[{10**lo_ci:.2f},{10**hi_ci:.2f}]" if lo_ci else "n/a"
            print(f"  [{lo:>6.2f}-{hi:>6.2f}]  {len(subset):>4d} {10**mean_err:>7.3f} {mean_bias:>+7.3f} {ci_str:>18s}")

    strat("NN Tanimoto to training set", [(0,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.01)], 'nn_tanimoto')
    strat("MW", [(0,300),(300,500),(500,700),(700,2000)], 'mw')
    strat("LogP", [(-5,0),(0,2),(2,4),(4,6),(6,15)], 'logp')

    # Ionization
    print(f"\nIonization (SMILES formal charge):")
    for ion in ['anion','cation','neutral_smi']:
        subset = [r for r in per_drug if r['ionization'] == ion]
        if not subset: continue
        errs = [r['abs_err'] for r in subset]
        mean_err = np.mean(errs); bias = np.mean([r['signed_err'] for r in subset])
        print(f"  {ion:<14s} n={len(subset):>3d}  AAFE={10**mean_err:.3f}  bias={bias:+.3f}")

    # ── Shared vs specific ──
    print(f"\n{'='*80}")
    print(f"Shared vs PLM-specific failures (vs Sisyphus Meta)")
    print(f"{'='*80}")
    categories = {'SHARED_HARD':[], 'PLM_SPECIFIC':[], 'META_SPECIFIC':[], 'BOTH_OK':[], 'MIDDLE':[], 'NO_META':[]}
    for r in per_drug:
        if r['meta_abs_err'] is None: categories['NO_META'].append(r); continue
        p, mm = r['abs_err'], r['meta_abs_err']
        if p > 0.5 and mm > 0.5: categories['SHARED_HARD'].append(r)
        elif p > 0.5 and mm < 0.3: categories['PLM_SPECIFIC'].append(r)
        elif mm > 0.5 and p < 0.3: categories['META_SPECIFIC'].append(r)
        elif p < 0.3 and mm < 0.3: categories['BOTH_OK'].append(r)
        else: categories['MIDDLE'].append(r)
    for k, v in categories.items():
        print(f"  {k:<16s}: {len(v):>3d}")

    # ── Correlation (noise floor test) ──
    pairs = [(r['abs_err'], r['meta_abs_err']) for r in per_drug if r['meta_abs_err'] is not None]
    corr = None
    if len(pairs) > 10:
        p_err, m_err = zip(*pairs)
        corr = float(np.corrcoef(p_err, m_err)[0, 1])
        print(f"\nPLM-Meta error correlation (Pearson r): {corr:.3f}  (N={len(pairs)})")

    # ── Top 20 worst ──
    print(f"\n{'='*100}")
    print(f"Top 20 worst PLM predictions")
    print(f"{'='*100}")
    print(f"{'drug':<25s} {'fold':>6s} {'signed':>7s} {'meta_f':>7s} {'NNT':>5s} {'MW':>5s} {'LogP':>5s} {'TPSA':>5s} {'ion':<10s}")
    sorted_drugs = sorted(per_drug, key=lambda r: -r['abs_err'])
    for r in sorted_drugs[:20]:
        meta_f = f"{10**r['meta_abs_err']:.2f}" if r['meta_abs_err'] is not None else "n/a"
        nnt = f"{r['nn_tanimoto']:.2f}" if r['nn_tanimoto'] is not None else "n/a"
        print(f"{r['name'][:24]:<25s} {r['fold_err']:>6.2f} {r['signed_err']:>+7.2f} {meta_f:>7s} "
              f"{nnt:>5s} {r['mw']:>5.0f} {r['logp']:>5.1f} {r['tpsa']:>5.0f} {r['ionization']:<10s}")

    # ── Cumulative error ──
    sorted_errs = sorted([r['abs_err'] for r in per_drug], reverse=True)
    total = sum(sorted_errs)
    for n in [5, 10, 20]:
        print(f"  Top {n} drugs account for {sum(sorted_errs[:n])/total*100:.1f}% of total |error|")

    # ── Decision analysis ──
    print(f"\n{'='*80}")
    print(f"DECISION ANALYSIS")
    print(f"{'='*80}")

    # (1) Noise floor
    noise_floor = corr is not None and corr > 0.6
    if noise_floor:
        print(f"[PRIMARY: NOISE FLOOR] r={corr:.2f} > 0.6 → shared difficulty dominant")
        print(f"  → Phase C (wrap up). Current ~3.258 is near irreducible error.")
    else:
        print(f"[OK] r={corr:.2f} → not shared-noise-dominated")

    # (2) Tanimoto gap
    low_t = [r['abs_err'] for r in per_drug if r['nn_tanimoto'] is not None and r['nn_tanimoto'] < 0.3]
    high_t = [r['abs_err'] for r in per_drug if r['nn_tanimoto'] is not None and r['nn_tanimoto'] >= 0.5]
    if low_t and high_t:
        ratio = np.mean(low_t) / max(np.mean(high_t), 0.01)
        print(f"\n[Tanimoto] low(<0.3) n={len(low_t)} AAFE={10**np.mean(low_t):.3f}  |  "
              f"high(>=0.5) n={len(high_t)} AAFE={10**np.mean(high_t):.3f}  |  ratio={ratio:.2f}")
        if ratio > 1.7:
            print(f"  → Phase B (DATA GAP): expand training chemistry — DailyMed/EMA/PMDA")
        elif ratio > 1.3:
            print(f"  → Mild Tanimoto gap (consider data expansion as secondary)")

    # (3) Ionization bias
    charged_errs = [r['signed_err'] for r in per_drug if r['ionization'] in ('anion','cation')]
    neutral_errs = [r['signed_err'] for r in per_drug if r['ionization'] == 'neutral_smi']
    if charged_errs and neutral_errs:
        charged_bias = np.mean(charged_errs); neutral_bias = np.mean(neutral_errs)
        print(f"\n[Ionization] charged bias={charged_bias:+.3f}  neutral bias={neutral_bias:+.3f}")
        if abs(charged_bias - neutral_bias) > 0.15:
            print(f"  → Phase A (FEATURE GAP): ionization-aware features (pKa, logD)")

    # (4) PLM_SPECIFIC count
    n_specific = len(categories['PLM_SPECIFIC'])
    n_hard = len(categories['SHARED_HARD'])
    print(f"\n[Failure categorization] PLM_SPECIFIC={n_specific}  SHARED_HARD={n_hard}")
    if n_specific >= 8:
        print(f"  → PLM-specific failures are fixable. Analyze their common features.")
        for r in categories['PLM_SPECIFIC'][:10]:
            print(f"     {r['name'][:25]:<26s} fold={r['fold_err']:.2f}  NNT={r['nn_tanimoto']:.2f}  MW={r['mw']:.0f}  LogP={r['logp']:.1f}  ion={r['ionization']}")

    # ── Save ──
    out = {
        'ho_aafe': round(float(aafe), 3),
        'n_drugs': len(per_drug),
        'plm_meta_correlation': round(corr, 3) if corr is not None else None,
        'categories': {k: len(v) for k, v in categories.items()},
        'noise_floor_triggered': bool(noise_floor),
        'per_drug': per_drug,
    }
    with open('data/validation/ho_diagnostic.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n→ Saved to data/validation/ho_diagnostic.json")


if __name__ == '__main__':
    main()
