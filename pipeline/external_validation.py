"""
External Validation: Brown 2025 (N≈90), Post-cutoff NMEs (N=9), Holdout 103.

Pre-registered hypotheses (2026-04-08):
  E1. Brown 2025: XGBoost AAFE 3.0–5.0 (success: <5.0)
  E2. Post-cutoff: XGBoost AAFE ~3.5
  E3. Holdout 103: AAFE ~3.3–3.5

Cherry-picking safeguards:
  - Model trained ONCE, no retuning on any test set
  - ALL drugs reported, zero exclusions
  - IK14 overlap with training checked before prediction
  - Results saved regardless of outcome

Data leakage checks:
  - Brown 2025 drugs: check IK14 not in training IK14s
  - Post-cutoff drugs: approved after May 2025, not in any PLM data
  - Holdout 103 cat3: not in v10 or LLM extracted data
"""

import json, math, warnings, sys, time
import numpy as np
import requests
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, 'pipeline')
from llm_enriched_experiment import (smi_to_ik, build_sample, CANONICAL_COND,
    normalize_condition, XGB_PARAMS)
import xgboost as xgb
warnings.filterwarnings('ignore')


def compute_aafe(predicted_log_cd, actual_log_cd):
    """AAFE = 10^mean(|pred - actual|) in log10 space."""
    errors = [abs(p - a) for p, a in zip(predicted_log_cd, actual_log_cd)]
    return 10**np.mean(errors), errors


def compute_two_fold(predicted_log_cd, actual_log_cd):
    """Fraction of predictions within 2-fold of observed."""
    n_ok = sum(1 for p, a in zip(predicted_log_cd, actual_log_cd)
               if abs(p - a) <= math.log10(2))
    return 100 * n_ok / len(predicted_log_cd) if predicted_log_cd else 0


def fetch_smiles_batch(drug_names, cache_path=None):
    """Fetch SMILES + InChIKey from PubChem for a list of drug names."""
    results = {}
    if cache_path:
        try:
            with open(cache_path) as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    for name in drug_names:
        if name in results and results[name].get('smiles'):
            continue
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               + name + "/property/IsomericSMILES,MolecularWeight,InChIKey/JSON")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                props = r.json()['PropertyTable']['Properties'][0]
                results[name] = {
                    'smiles': props.get('SMILES', ''),
                    'mw': props.get('MolecularWeight', ''),
                    'inchikey': props.get('InChIKey', ''),
                }
            else:
                results[name] = {'smiles': '', 'mw': '', 'inchikey': '',
                                 'error': 'HTTP ' + str(r.status_code)}
        except Exception as e:
            results[name] = {'smiles': '', 'mw': '', 'inchikey': '',
                             'error': str(e)}
        time.sleep(0.25)

    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(results, f, indent=2)
    return results


def build_training(v10, llm, exclude_ik14s, tdc):
    """Reproduce best-model training, excluding specified IK14s."""
    X_tr, Y_tr, g_tr, W_tr = [], [], [], []
    for p in v10:
        smi, dose, ik, lcd = p.get('smiles'), p.get('dose_mg'), p.get('ik'), p.get('log_cd')
        if not smi or not dose or dose <= 0 or lcd is None:
            continue
        if ik in exclude_ik14s:
            continue
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None:
            continue
        X_tr.append(s); Y_tr.append(lcd); g_tr.append(ik); W_tr.append(1.0)

    v10_mean_lcd = defaultdict(list)
    for p in v10:
        ik, lcd = p.get('ik'), p.get('log_cd')
        if ik and lcd is not None:
            v10_mean_lcd[ik].append(lcd)
    v10_mean = {ik: float(np.mean(l)) for ik, l in v10_mean_lcd.items()}

    grouped = defaultdict(list)
    for t in llm:
        if not t.get('smiles'):
            continue
        smi, dose, cmax = t['smiles'], t.get('dose_mg', 0), t.get('cmax_ng_ml', 0)
        if not dose or dose <= 0 or not cmax or cmax <= 0:
            continue
        ik = smi_to_ik(smi)
        if not ik or ik in exclude_ik14s:
            continue
        log_cd = math.log10(cmax / dose)
        if log_cd < -3 or log_cd > 3:
            continue
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
        if key[0] in v10_mean and abs(med_lcd - v10_mean[key[0]]) > 1.0:
            continue
        w = float(np.mean([conf_w.get(e['conf'], 0.5) for e in entries]))
        conditions = {'route': key[1], 'schedule': key[2], 'food': key[3],
                      'formulation': 'tablet', 'population': key[4]}
        s = build_sample(entries[0]['smi'], float(np.median([e['dose'] for e in entries])),
                         key[0], conditions, tdc, True)
        if s is None:
            continue
        X_tr.append(s); Y_tr.append(med_lcd); g_tr.append(key[0]); W_tr.append(w)

    return (np.array(X_tr, dtype=np.float32), np.array(Y_tr, dtype=np.float32),
            np.array(g_tr), np.array(W_tr, dtype=np.float32))


def predict_drug(model, smiles, dose_mg, tdc):
    """Predict log_cd for a single drug. Returns None if features can't be built."""
    ik = smi_to_ik(smiles)
    if not ik:
        return None
    s = build_sample(smiles, dose_mg, ik, CANONICAL_COND, tdc, use_conditions=True)
    if s is None:
        return None
    X = np.array([s], dtype=np.float32)
    X = np.where(np.isinf(X), np.nan, X)
    return float(model.predict(X)[0])


def evaluate_set(model, drugs, tdc, set_name):
    """Evaluate a list of drugs. Each drug must have: smiles, dose_mg, cmax_obs_ngml."""
    preds, actuals, names = [], [], []
    skipped = []

    for d in drugs:
        name = d.get('name', d.get('drug_name', 'unknown'))
        smiles = d.get('smiles', '')
        dose = d.get('dose_mg', d.get('total_daily_dose_mg', 0))
        cmax = d.get('cmax_obs_ngml', d.get('cmax_ng_ml', 0))

        if not smiles or not dose or dose <= 0 or not cmax or cmax <= 0:
            skipped.append({'name': name, 'reason': 'missing data'})
            continue

        pred_lcd = predict_drug(model, smiles, dose, tdc)
        if pred_lcd is None:
            skipped.append({'name': name, 'reason': 'feature build failed'})
            continue

        actual_lcd = math.log10(cmax / dose)
        preds.append(pred_lcd)
        actuals.append(actual_lcd)
        names.append(name)

    if not preds:
        print("  [" + set_name + "] No valid predictions!")
        return None

    aafe, errors = compute_aafe(preds, actuals)
    f2 = compute_two_fold(preds, actuals)
    bias = np.mean([p - a for p, a in zip(preds, actuals)])

    print("  [" + set_name + "] N=" + str(len(preds)) + " AAFE=" + ("%.3f" % aafe)
          + " 2-fold=" + ("%.1f" % f2) + "% bias=" + ("%.3f" % bias)
          + " skipped=" + str(len(skipped)))

    per_drug = []
    for name, pred, actual, err in zip(names, preds, actuals, errors):
        fold = 10**err
        direction = 'OVER' if pred > actual else 'UNDER'
        per_drug.append({
            'name': name, 'pred_log_cd': round(pred, 4),
            'actual_log_cd': round(actual, 4),
            'abs_err': round(err, 4), 'signed_err': round(pred - actual, 4),
            'fold_err': round(fold, 2), 'direction': direction,
        })

    return {
        'set_name': set_name,
        'n_evaluated': len(preds),
        'n_skipped': len(skipped),
        'aafe': round(aafe, 3),
        'two_fold_pct': round(f2, 1),
        'bias': round(bias, 3),
        'per_drug': sorted(per_drug, key=lambda x: -x['abs_err']),
        'skipped': skipped,
    }


def main():
    print("=" * 80)
    print("EXTERNAL VALIDATION — Pre-registered experiments E1/E2/E3")
    print("=" * 80)

    # ── Load data ──
    with open('data/curated/tdc_adme_data.json') as f:
        tdc = json.load(f)
    with open('data/validation/holdout_definition.json') as f:
        ho_data = json.load(f)
    holdout_drugs = ho_data['holdout_drugs']
    ho_iks_14 = set(d['inchikey14'] for d in holdout_drugs)
    with open('data/curated/plm_dataset_v10_labels.json') as f:
        v10 = json.load(f)
    with open('data/llm_extracted/pk_llm_merged.json') as f:
        llm = json.load(f)

    # ── Train model ONCE (same as ho_diagnostic) ──
    print("\n[TRAINING] Building model (excluding 97 holdout IK14s)...")
    X_tr, Y_tr, g_tr, W_tr = build_training(v10, llm, ho_iks_14, tdc)
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    train_iks = set(g_tr)
    print("  Training: " + str(len(Y_tr)) + " samples, "
          + str(len(train_iks)) + " unique drugs")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, Y_tr, sample_weight=W_tr)

    results = {
        'date': '2026-04-08',
        'pre_registered': True,
        'model': 'XGBoost (same as ho_diagnostic, no retuning)',
        'n_training_samples': int(len(Y_tr)),
        'n_training_drugs': int(len(train_iks)),
    }

    # ── E0: Reproduce original holdout (sanity check) ──
    print("\n[E0] Sanity check: original 97-drug holdout")
    e0 = evaluate_set(model, holdout_drugs, tdc, 'holdout_97')
    results['E0_holdout_97'] = e0

    # ── E1: Brown 2025 external validation ──
    print("\n[E1] Brown 2025 external validation (pre-registered: AAFE < 5.0)")
    with open('data/raw/brown_2025_oral_drugs.json') as f:
        brown = json.load(f)

    # Get SMILES for Brown drugs
    brown_names = [d['drug_name'] for d in brown
                   if d.get('cmax_ng_ml') and d.get('total_daily_dose_mg')]
    print("  Fetching SMILES for " + str(len(brown_names)) + " drugs...")
    smiles_cache = fetch_smiles_batch(brown_names,
                                      'data/validation/brown_2025_smiles.json')

    # Attach SMILES and check leakage
    brown_eval = []
    n_leakage = 0
    for d in brown:
        if not d.get('cmax_ng_ml') or not d.get('total_daily_dose_mg'):
            continue
        name = d['drug_name']
        sdata = smiles_cache.get(name, {})
        smiles = sdata.get('smiles', '')
        ik = sdata.get('inchikey', '')
        ik14 = ik[:14] if ik else ''

        # LEAKAGE CHECK: is this drug in training?
        if ik14 and ik14 in train_iks:
            print("  LEAKAGE WARNING: " + name + " (IK14=" + ik14
                  + ") found in training set — EXCLUDING")
            n_leakage += 1
            continue

        brown_eval.append({
            'name': name,
            'smiles': smiles,
            'dose_mg': d['total_daily_dose_mg'],
            'cmax_obs_ngml': d['cmax_ng_ml'],
            'inchikey14': ik14,
        })

    print("  Leakage exclusions: " + str(n_leakage))
    print("  Clean evaluation set: " + str(len(brown_eval)))
    e1 = evaluate_set(model, brown_eval, tdc, 'brown_2025')
    if e1:
        e1['n_leakage_excluded'] = n_leakage
        e1['pre_registered_criterion'] = 'AAFE < 5.0'
        e1['criterion_met'] = e1['aafe'] < 5.0
    results['E1_brown_2025'] = e1

    # ── E2: Post-cutoff prospective validation ──
    print("\n[E2] Post-cutoff prospective validation (pre-registered: AAFE ~3.5)")
    with open('data/validation/post_cutoff_candidates.json') as f:
        pc_data = json.load(f)
    with open('data/validation/post_cutoff_smiles.json') as f:
        pc_smiles = json.load(f)

    pc_eval = []
    for d in pc_data['drugs']:
        if not d.get('cmax_ngml'):
            continue
        name = d['name']
        sdata = pc_smiles.get(name, {})
        smiles = sdata.get('smiles', '')
        ik = sdata.get('inchikey', '')
        ik14 = ik[:14] if ik else ''

        # LEAKAGE CHECK
        if ik14 and ik14 in train_iks:
            print("  LEAKAGE WARNING: " + name + " in training — EXCLUDING")
            continue

        pc_eval.append({
            'name': name,
            'smiles': smiles,
            'dose_mg': d['dose_mg'],
            'cmax_obs_ngml': d['cmax_ngml'],
            'inchikey14': ik14,
        })

    e2 = evaluate_set(model, pc_eval, tdc, 'post_cutoff')
    results['E2_post_cutoff'] = e2

    # ── E3: Holdout 103 (original 97 + 6 recovered) ──
    print("\n[E3] Holdout 103 (97 + 6 recovered, pre-registered: AAFE ~3.3-3.5)")
    with open('data/validation/holdout_recovery_cat3.json') as f:
        cat3 = json.load(f)

    cat3_eval = []
    for d in cat3['drugs']:
        smiles = d.get('smiles', '')
        ik14 = d.get('inchikey14', '')

        # LEAKAGE CHECK: these must NOT be in training
        if ik14 and ik14 in train_iks:
            print("  LEAKAGE: " + d['name'] + " IN TRAINING — CANNOT ADD TO HOLDOUT")
            continue

        cat3_eval.append({
            'name': d['name'],
            'smiles': smiles,
            'dose_mg': d['dose_mg'],
            'cmax_obs_ngml': d['cmax_obs_ngml'],
            'inchikey14': ik14,
        })

    holdout_103 = list(holdout_drugs) + cat3_eval
    # Rename fields for consistency
    for d in holdout_103:
        if 'cmax_obs_ngml' not in d and 'cmax_ng_ml' in d:
            d['cmax_obs_ngml'] = d['cmax_ng_ml']

    e3 = evaluate_set(model, holdout_103, tdc, 'holdout_103')
    if e3:
        e3['n_original'] = 97
        e3['n_recovered'] = len(cat3_eval)
    results['E3_holdout_103'] = e3

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key in ['E0_holdout_97', 'E1_brown_2025', 'E2_post_cutoff', 'E3_holdout_103']:
        r = results.get(key)
        if r:
            print("  " + key + ": AAFE=" + str(r['aafe'])
                  + " 2fold=" + str(r['two_fold_pct']) + "%"
                  + " N=" + str(r['n_evaluated']))

    # Pre-registration check
    print("\nPre-registration results:")
    if e1:
        status = "PASS" if e1['criterion_met'] else "FAIL"
        print("  E1 Brown 2025: AAFE " + str(e1['aafe']) + " vs criterion <5.0 → " + status)
    if e2:
        print("  E2 Post-cutoff: AAFE " + str(e2['aafe']) + " (expected ~3.5)")
    if e3:
        print("  E3 Holdout 103: AAFE " + str(e3['aafe']) + " (expected 3.3-3.5)")

    # Save (convert numpy types for JSON)
    def jsonify(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        raise TypeError(str(type(obj)))

    out_path = 'data/validation/external_validation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=jsonify)
    print("\nSaved: " + out_path)


if __name__ == '__main__':
    main()
