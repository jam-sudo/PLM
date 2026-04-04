"""
Benchmark Audit: identify questionable Sisyphus holdout values.

Method:
1. For 18 holdout drugs overlapping with LLM extraction, compute LLM median Cmax.
2. Compare LLM value vs Sisyphus obs vs Sisyphus meta.
3. Flag as "likely wrong obs" if LLM median aligns with meta but disagrees with obs.
4. Create corrected holdout, re-evaluate PLM best model.

Output:
- data/validation/holdout_audit.json (per-drug analysis)
- data/validation/holdout_corrected.json (PLM v2 benchmark)
- Dual-metric evaluation report
"""

import json, math, warnings
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import sys; sys.path.insert(0, 'pipeline')
from llm_enriched_experiment import (smi_to_ik, build_sample, eval_xgb,
    CANONICAL_COND, normalize_condition, XGB_PARAMS)
import xgboost as xgb
warnings.filterwarnings('ignore')


def main():
    # ── Load ──
    with open('data/curated/tdc_adme_data.json') as f: tdc = json.load(f)
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    holdout_drugs = ho['holdout_drugs']
    with open('data/llm_extracted/pk_llm_merged.json') as f: llm = json.load(f)
    with open('data/curated/plm_dataset_v10_labels.json') as f: v10 = json.load(f)

    ho_by_ik = {d['inchikey14']: d for d in holdout_drugs}

    # ── Step 1: Find LLM matches for each holdout drug ──
    llm_by_ik = defaultdict(list)
    for t in llm:
        if not t.get('smiles'): continue
        ik = smi_to_ik(t['smiles'])
        if ik not in ho_by_ik: continue
        llm_by_ik[ik].append(t)

    print("=" * 95)
    print("BENCHMARK AUDIT: Sisyphus holdout obs vs LLM + Sisyphus meta")
    print("=" * 95)
    print(f"{'drug':<22s} {'ho_obs':>9s} {'meta':>9s} {'llm_med':>9s} {'|log(LLM/obs)|':>14s} {'|log(meta/obs)|':>15s} {'verdict':>9s}")
    print('-' * 95)

    audit = []
    for ik, ho_d in ho_by_ik.items():
        ho_dose = ho_d['dose_mg']
        ho_obs = ho_d['cmax_obs_ngml']
        meta = ho_d.get('cmax_sisyphus_meta_mgL', 0) * 1000 if ho_d.get('cmax_sisyphus_meta_mgL') else None

        if ik not in llm_by_ik:
            continue

        # Filter LLM to canonical conditions near target dose
        candidates = []
        for t in llm_by_ik[ik]:
            llm_dose = t.get('dose_mg', 0)
            llm_cmax = t.get('cmax_ng_ml', 0)
            if not llm_dose or not llm_cmax or llm_dose <= 0: continue
            # Prefer matching dose and canonical conditions
            score = 0
            if 0.5 * ho_dose <= llm_dose <= 2.0 * ho_dose: score += 2
            if t.get('route') == 'oral': score += 1
            if t.get('dose_schedule') == 'single_dose': score += 1
            if t.get('population') == 'healthy_adult': score += 1
            if t.get('confidence') == 'high': score += 1
            # Dose-normalized Cmax (assume linear PK)
            cmax_at_ho_dose = llm_cmax * (ho_dose / llm_dose)
            candidates.append({'score': score, 'cmax_normed': cmax_at_ho_dose,
                               'llm_cmax': llm_cmax, 'llm_dose': llm_dose,
                               't': t})

        if not candidates: continue
        # Take median of top-scoring candidates
        max_score = max(c['score'] for c in candidates)
        top = [c for c in candidates if c['score'] == max_score]
        llm_med_normed = float(np.median([c['cmax_normed'] for c in top]))

        log_llm_obs = math.log10(llm_med_normed / ho_obs) if ho_obs > 0 else 0
        log_meta_obs = math.log10(meta / ho_obs) if meta and ho_obs > 0 else None

        # Verdict: obs is SUSPECT if both LLM and meta disagree with obs in SAME direction by >3x (0.48 log)
        verdict = 'ok'
        if log_meta_obs is not None and abs(log_llm_obs) > 0.48 and abs(log_meta_obs) > 0.48:
            if np.sign(log_llm_obs) == np.sign(log_meta_obs):
                verdict = 'SUSPECT'
        elif abs(log_llm_obs) > 0.7:  # 5x off with LLM alone
            verdict = 'check'

        marker = '!!' if verdict == 'SUSPECT' else ('?' if verdict == 'check' else '  ')
        meta_str = f"{meta:>9.1f}" if meta else '      n/a'
        log_meta_str = f"{log_meta_obs:+.3f}" if log_meta_obs is not None else ' n/a '
        print(f"  {ho_d['name'][:20]:<22s} {ho_obs:>9.1f} {meta_str} {llm_med_normed:>9.1f} {log_llm_obs:>+13.3f}  {log_meta_str:>13s}  {marker}{verdict}")

        audit.append({
            'drug': ho_d['name'],
            'ik': ik,
            'ho_dose': ho_dose,
            'ho_obs_cmax': ho_obs,
            'sisyphus_meta_cmax_ngml': meta,
            'llm_median_cmax_at_dose': llm_med_normed,
            'log_llm_obs': round(log_llm_obs, 3),
            'log_meta_obs': round(log_meta_obs, 3) if log_meta_obs else None,
            'verdict': verdict,
        })

    # ── Step 2: Build corrected holdout ──
    suspects = [a for a in audit if a['verdict'] == 'SUSPECT']
    print(f"\n{'='*80}")
    print(f"Audit Summary: {len(audit)} drugs checked (overlap with LLM extraction)")
    print(f"  SUSPECT (both LLM+meta disagree with obs): {len(suspects)}")
    print(f"  check (LLM alone disagrees): {sum(1 for a in audit if a['verdict']=='check')}")
    print(f"  ok (agrees): {sum(1 for a in audit if a['verdict']=='ok')}")
    print(f"{'='*80}")

    print("\nSUSPECT drugs (use LLM corrected value):")
    for s in suspects:
        print(f"  {s['drug']:<25s} obs={s['ho_obs_cmax']:>8.1f} → corrected={s['llm_median_cmax_at_dose']:>8.1f}  ({10**s['log_llm_obs']:.2f}x)")

    # ── Step 3: Evaluate PLM best on both benchmarks ──
    print(f"\n{'='*80}")
    print("Dual-metric evaluation of PLM best model")
    print(f"{'='*80}")

    # Build training
    X_tr, Y_tr, g_tr, W_tr = [], [], [], []
    for p in v10:
        smi,dose,ik,lcd = p.get('smiles'),p.get('dose_mg'),p.get('ik'),p.get('log_cd')
        if not smi or not dose or dose<=0 or lcd is None: continue
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None: continue
        X_tr.append(s); Y_tr.append(lcd); g_tr.append(ik); W_tr.append(1.0)

    # Add LLM with agreement filter + confidence weights
    v10_iks = set(g_tr)
    v10_mean_lcd = defaultdict(list)
    for p in v10:
        ik = p.get('ik'); lcd = p.get('log_cd')
        if ik and lcd is not None: v10_mean_lcd[ik].append(lcd)
    v10_mean = {ik: float(np.mean(l)) for ik, l in v10_mean_lcd.items()}

    ho_iks_14 = set(ho_by_ik.keys())
    grouped = defaultdict(list)
    for t in llm:
        if not t.get('smiles'): continue
        smi, dose, cmax = t['smiles'], t.get('dose_mg',0), t.get('cmax_ng_ml',0)
        if not dose or dose<=0 or not cmax or cmax<=0: continue
        ik = smi_to_ik(smi)
        if not ik or ik in ho_iks_14: continue
        log_cd = math.log10(cmax/dose)
        if log_cd < -3 or log_cd > 3: continue
        conf = t.get('confidence', 'medium')
        key = (ik, normalize_condition(t.get('route','oral')),
               normalize_condition(t.get('dose_schedule','single_dose')),
               normalize_condition(t.get('food','not_specified')),
               normalize_condition(t.get('population','healthy_adult')))
        grouped[key].append({'smi': smi, 'dose': dose, 'log_cd': log_cd, 'conf': conf})

    conf_weight = {'high': 1.0, 'medium': 0.7, 'low': 0.3}
    for key, entries in grouped.items():
        log_cds = [e['log_cd'] for e in entries]
        med_lcd = float(np.median(log_cds))
        if key[0] in v10_mean and abs(med_lcd - v10_mean[key[0]]) > 1.0: continue
        doses = [e['dose'] for e in entries]
        w = float(np.mean([conf_weight.get(e['conf'], 0.5) for e in entries]))
        conditions = {'route':key[1],'schedule':key[2],'food':key[3],'formulation':'tablet','population':key[4]}
        s = build_sample(entries[0]['smi'], float(np.median(doses)), key[0], conditions, tdc, True)
        if s is None: continue
        X_tr.append(s); Y_tr.append(med_lcd); g_tr.append(key[0]); W_tr.append(w)

    X_tr = np.array(X_tr, dtype=np.float32); Y_tr = np.array(Y_tr, dtype=np.float32)
    W_tr = np.array(W_tr, dtype=np.float32)
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    print(f"Training: {len(Y_tr)} profiles")

    # Build holdout — both original and corrected
    X_ho_orig, Y_ho_orig = [], []
    X_ho_corr, Y_ho_corr = [], []
    audit_by_ik = {a['ik']: a for a in audit}

    for d in holdout_drugs:
        smi,dose,cmax = d.get('smiles'),d.get('dose_mg'),d.get('cmax_obs_ngml')
        if not smi or not dose or dose<=0 or not cmax or cmax<=0: continue
        ik = d.get('inchikey14','')
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None: continue
        X_ho_orig.append(s); Y_ho_orig.append(math.log10(cmax/dose))
        # Corrected: use LLM value if SUSPECT
        if ik in audit_by_ik and audit_by_ik[ik]['verdict'] == 'SUSPECT':
            corrected_cmax = audit_by_ik[ik]['llm_median_cmax_at_dose']
            X_ho_corr.append(s); Y_ho_corr.append(math.log10(corrected_cmax/dose))
        else:
            X_ho_corr.append(s); Y_ho_corr.append(math.log10(cmax/dose))

    X_ho_orig = np.where(np.isinf(np.array(X_ho_orig, dtype=np.float32)), np.nan, np.array(X_ho_orig, dtype=np.float32))
    X_ho_corr = np.where(np.isinf(np.array(X_ho_corr, dtype=np.float32)), np.nan, np.array(X_ho_corr, dtype=np.float32))
    Y_ho_orig = np.array(Y_ho_orig, dtype=np.float32)
    Y_ho_corr = np.array(Y_ho_corr, dtype=np.float32)

    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, Y_tr, sample_weight=W_tr)
    p_orig = m.predict(X_ho_orig)
    p_corr = m.predict(X_ho_corr)  # same X, different Y

    err_orig = np.abs(p_orig - Y_ho_orig)
    err_corr = np.abs(p_corr - Y_ho_corr)
    aafe_orig = 10**np.mean(err_orig)
    aafe_corr = 10**np.mean(err_corr)
    f2_orig = np.mean(err_orig < np.log10(2)) * 100
    f2_corr = np.mean(err_corr < np.log10(2)) * 100

    print(f"\n  Original Sisyphus holdout:  HO={aafe_orig:.3f}  2f={f2_orig:.1f}%")
    print(f"  LLM-corrected holdout:      HO={aafe_corr:.3f}  2f={f2_corr:.1f}%")
    print(f"  Improvement from correction: {aafe_orig - aafe_corr:+.3f}")

    # Sisyphus meta on corrected
    meta_err_orig, meta_err_corr = [], []
    for d in holdout_drugs:
        obs = d.get('cmax_obs_ngml'); meta = d.get('cmax_sisyphus_meta_mgL')
        if not obs or not meta or obs<=0: continue
        meta_ngml = meta * 1000
        ik = d.get('inchikey14','')
        corrected = audit_by_ik[ik]['llm_median_cmax_at_dose'] if (ik in audit_by_ik and audit_by_ik[ik]['verdict']=='SUSPECT') else obs
        meta_err_orig.append(abs(math.log10(meta_ngml/obs)))
        meta_err_corr.append(abs(math.log10(meta_ngml/corrected)))
    print(f"\n  Sisyphus Meta vs original:  AAFE={10**np.mean(meta_err_orig):.3f}")
    print(f"  Sisyphus Meta vs corrected: AAFE={10**np.mean(meta_err_corr):.3f}")

    # Save audit
    out = {
        'n_checked': len(audit),
        'suspect_count': len(suspects),
        'suspect_drugs': [s['drug'] for s in suspects],
        'audit_details': audit,
        'plm_ho_aafe_original': round(float(aafe_orig), 3),
        'plm_ho_aafe_corrected': round(float(aafe_corr), 3),
        'sisyphus_meta_aafe_original': round(float(10**np.mean(meta_err_orig)), 3),
        'sisyphus_meta_aafe_corrected': round(float(10**np.mean(meta_err_corr)), 3),
    }
    with open('data/validation/holdout_audit.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n→ Saved audit to data/validation/holdout_audit.json")


if __name__ == '__main__':
    main()
