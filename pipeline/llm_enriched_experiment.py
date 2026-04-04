"""
PLM: LLM-enriched training with condition features.

Two key innovations:
1. Drug-level median estimation from LLM tuples (robust to dose-escalation outliers)
2. Condition features added to model (route, schedule, food, formulation)

This lets the model learn condition-dependent PK variations, so diverse LLM data
can be used WITHOUT violating linear-PK assumptions.

Comparison experiments:
  E1: v10 baseline (no conditions, no LLM)
  E2: v10 + conditions (canonical assumed for v10)
  E3: v10 + clean-filtered LLM + conditions
  E4: v10 + drug-median LLM + conditions (best-of-both)
"""

import json, math, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import GroupKFold
import xgboost as xgb
warnings.filterwarnings('ignore')

FP_BITS = 4096
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.3, reg_alpha=1.0, reg_lambda=5.0,
    min_child_weight=5, random_state=42, n_jobs=1, verbosity=0)

# Condition encoding
ROUTES = ['oral', 'IV', 'IM', 'SC', 'other']  # 5
SCHEDULES = ['single_dose', 'multiple_dose', 'steady_state']  # 3
FOODS = ['fasted', 'fed', 'not_specified']  # 3
FORMS = ['tablet', 'capsule', 'solution', 'other']  # 4
POPS = ['healthy_adult', 'patient', 'impaired']  # 3

CANONICAL_COND = {
    'route': 'oral', 'schedule': 'single_dose', 'food': 'fasted',
    'formulation': 'tablet', 'population': 'healthy_adult',
}


def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS), dtype=np.float32)

def smiles_to_physchem(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return np.full(20, np.nan)
    def safe(fn):
        try: v=fn(mol); return v if np.isfinite(v) else np.nan
        except: return np.nan
    return np.array([safe(Descriptors.ExactMolWt),safe(Descriptors.MolLogP),safe(Descriptors.TPSA),
        Descriptors.NumHDonors(mol),Descriptors.NumHAcceptors(mol),Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),Descriptors.NumAromaticRings(mol),safe(Descriptors.FractionCSP3),
        Descriptors.HeavyAtomCount(mol),Descriptors.NumHeteroatoms(mol),safe(Descriptors.LabuteASA),
        safe(Descriptors.BertzCT),safe(Descriptors.Chi0v),safe(Descriptors.Chi1v),
        safe(Descriptors.HallKierAlpha),safe(Descriptors.Kappa1),safe(Descriptors.Kappa2),
        safe(Descriptors.MaxPartialCharge),safe(Descriptors.MinPartialCharge)], dtype=np.float32)

def get_tdc_features(ik, tdc):
    e = tdc.get(ik, {})
    keys = ['logS','caco2_logPapp','ppb_pct','vd_L_kg','half_life_h',
            'clearance_ul_min_mg','clearance_ul_min_million_cells','logD','bioavailability_binary']
    return np.array([e.get(k, np.nan) for k in keys], dtype=np.float32)

def compute_micropbpk(ik, tdc):
    e = tdc.get(ik, {})
    feats = np.full(6, np.nan)
    caco2,ppb,cl_hep,cl_mic,vd,thalf = e.get('caco2_logPapp'),e.get('ppb_pct'),e.get('clearance_ul_min_million_cells'),e.get('clearance_ul_min_mg'),e.get('vd_L_kg'),e.get('half_life_h')
    if caco2 is not None:
        papp = 10**caco2
        feats[0] = min(max(papp/(papp+1e-6),0.01),1.0)
    if ppb is not None:
        feats[1] = max((100-ppb)/100, 0.001)
    Q_h = 1500
    if cl_hep is not None:
        cl_int = cl_hep * 120 * 20 / 1000
        feats[2] = min(max(cl_int/(Q_h+cl_int),0.001),0.999)
    elif cl_mic is not None:
        cl_int = cl_mic*45*20/1000
        feats[2] = min(max(cl_int/(Q_h+cl_int),0.001),0.999)
    if not np.isnan(feats[0]) and not np.isnan(feats[2]):
        feats[3] = feats[0]*(1-feats[2])
    if vd is not None: feats[4] = vd
    if thalf is not None and thalf > 0: feats[5] = 0.693/thalf
    return feats


def onehot(val, categories):
    vec = np.zeros(len(categories), dtype=np.float32)
    if val in categories:
        vec[categories.index(val)] = 1.0
    else:
        vec[-1] = 1.0  # 'other' fallback
    return vec

def normalize_condition(c):
    """Normalize condition string to canonical set."""
    c = c.lower() if isinstance(c, str) else ''
    # Route
    if c in ('subcutaneous', 'sc'): return 'SC'
    if c in ('intramuscular', 'im'): return 'IM'
    if c in ('intravenous', 'iv'): return 'IV'
    if c == 'oral': return 'oral'
    # Schedule
    if c in ('single_dose', 'multiple_dose', 'steady_state'): return c
    # Food
    if c == 'high_fat_meal': return 'fed'
    if c in ('fasted', 'fed', 'not_specified'): return c
    # Formulation
    if c in ('IR', 'ER', 'DR', 'tablet'): return 'tablet'
    if c == 'capsule': return 'capsule'
    if c in ('solution', 'suspension', 'injection'): return 'solution' if c != 'injection' else 'other'
    # Population
    if c in ('hepatic_impaired', 'renal_impaired', 'elderly', 'pediatric'): return 'impaired'
    if c in ('healthy_adult', 'patient'): return c
    return c or 'other'

def build_condition_features(conditions):
    """5+3+3+4+3 = 18 onehot features."""
    route = normalize_condition(conditions.get('route', 'oral'))
    sched = normalize_condition(conditions.get('schedule', 'single_dose'))
    food = normalize_condition(conditions.get('food', 'fasted'))
    form = normalize_condition(conditions.get('formulation', 'tablet'))
    pop = normalize_condition(conditions.get('population', 'healthy_adult'))
    return np.concatenate([
        onehot(route, ROUTES),
        onehot(sched, SCHEDULES),
        onehot(food, FOODS),
        onehot(form, FORMS),
        onehot(pop, POPS),
    ])

def build_sample(smi, dose, ik, conditions, tdc, use_conditions=True):
    fp = smiles_to_fp(smi)
    if fp is None: return None
    pc = smiles_to_physchem(smi)
    adme = get_tdc_features(ik, tdc)
    mpbpk = compute_micropbpk(ik, tdc)
    ld = np.float32(math.log10(dose))
    base = np.concatenate([fp, pc, adme, mpbpk, [ld]])
    if use_conditions:
        cond = build_condition_features(conditions)
        return np.concatenate([base, cond])
    return base


def eval_xgb(X_tr, Y_tr, g_tr, X_ho, Y_ho, label=""):
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    X_ho = np.where(np.isinf(X_ho), np.nan, X_ho)
    n_splits = min(5, len(set(g_tr)))
    gkf = GroupKFold(n_splits=n_splits)
    cv = np.full_like(Y_tr, np.nan)
    for ti, vi in gkf.split(X_tr, Y_tr, g_tr):
        m = xgb.XGBRegressor(**XGB_PARAMS); m.fit(X_tr[ti], Y_tr[ti])
        cv[vi] = m.predict(X_tr[vi])
    cv_aafe = 10**np.nanmean(np.abs(cv-Y_tr))
    m = xgb.XGBRegressor(**XGB_PARAMS); m.fit(X_tr, Y_tr)
    p_ho = m.predict(X_ho)
    err = np.abs(p_ho - Y_ho)
    ho_aafe = 10**np.mean(err)
    f2 = np.mean(err<np.log10(2))*100; f3 = np.mean(err<np.log10(3))*100
    print(f"  {label:<35s} N={len(Y_tr):<5d} D={len(set(g_tr)):<4d} CV={cv_aafe:.3f}  HO={ho_aafe:.3f}  2f={f2:.1f}%  3f={f3:.1f}%")
    return dict(cv=cv_aafe, ho=ho_aafe, f2=f2, f3=f3, n=len(Y_tr))


def smi_to_ik(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try: return Chem.InchiToInchiKey(Chem.MolToInchi(mol))[:14]
    except: return None


def main():
    # ─── Load data ──
    print("Loading data...")
    with open('data/curated/tdc_adme_data.json') as f: tdc = json.load(f)
    with open('data/validation/holdout_definition.json') as f: ho = json.load(f)
    holdout_drugs = ho['holdout_drugs']
    ho_iks_14 = set(d['inchikey14'] for d in holdout_drugs)
    with open('data/curated/plm_dataset_v10_labels.json') as f: v10 = json.load(f)
    if isinstance(v10, dict): v10 = v10.get('profiles', v10)
    with open('data/llm_extracted/pk_llm_merged.json') as f: llm = json.load(f)

    print(f"  v10: {len(v10)} profiles | LLM: {len(llm)} tuples")

    # ─── Build holdout features (canonical conditions) ──
    X_ho_nocond, X_ho_cond, Y_ho = [], [], []
    for d in holdout_drugs:
        smi, dose, cmax = d.get('smiles'), d.get('dose_mg'), d.get('cmax_obs_ngml')
        if not smi or not dose or dose<=0 or not cmax or cmax<=0: continue
        ik = d.get('inchikey14', '')
        sample_nc = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=False)
        sample_c = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if sample_nc is None: continue
        X_ho_nocond.append(sample_nc); X_ho_cond.append(sample_c)
        Y_ho.append(math.log10(cmax/dose))
    X_ho_nocond = np.array(X_ho_nocond, dtype=np.float32)
    X_ho_cond = np.array(X_ho_cond, dtype=np.float32)
    Y_ho = np.array(Y_ho, dtype=np.float32)
    print(f"  Holdout: {len(Y_ho)} drugs")

    # ─── E1: v10 baseline (no conditions) ──
    print(f"\n{'='*80}\n[Experiments]\n{'='*80}")
    X_tr, Y_tr, g_tr = [], [], []
    for p in v10:
        smi,dose,ik,lcd = p.get('smiles'),p.get('dose_mg'),p.get('ik'),p.get('log_cd')
        if not smi or not dose or dose<=0 or lcd is None: continue
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=False)
        if s is None: continue
        X_tr.append(s); Y_tr.append(lcd); g_tr.append(ik)
    X_tr = np.array(X_tr, dtype=np.float32); Y_tr = np.array(Y_tr, dtype=np.float32); g_tr = np.array(g_tr)
    r1 = eval_xgb(X_tr, Y_tr, g_tr, X_ho_nocond, Y_ho, 'E1: v10 baseline (no cond)')

    # ─── E2: v10 with condition features (all canonical) ──
    X_tr, Y_tr, g_tr = [], [], []
    for p in v10:
        smi,dose,ik,lcd = p.get('smiles'),p.get('dose_mg'),p.get('ik'),p.get('log_cd')
        if not smi or not dose or dose<=0 or lcd is None: continue
        s = build_sample(smi, dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None: continue
        X_tr.append(s); Y_tr.append(lcd); g_tr.append(ik)
    X_tr = np.array(X_tr, dtype=np.float32); Y_tr = np.array(Y_tr, dtype=np.float32); g_tr = np.array(g_tr)
    r2 = eval_xgb(X_tr, Y_tr, g_tr, X_ho_cond, Y_ho, 'E2: v10 + cond (canonical)')

    # ─── E3: v10 + filtered LLM + conditions ──
    # Filter LLM tuples: exclude DDI/population modifiers, exclude extreme doses
    X_extra, Y_extra, g_extra = [], [], []
    for t in llm:
        if not t.get('smiles'): continue
        smi, dose, cmax = t['smiles'], t.get('dose_mg',0), t.get('cmax_ng_ml',0)
        if not dose or dose<=0 or not cmax or cmax<=0: continue
        ik = smi_to_ik(smi)
        if not ik or ik in ho_iks_14: continue  # no holdout leak
        log_cd = math.log10(cmax/dose)
        if log_cd < -3 or log_cd > 3: continue
        conditions = {
            'route': t.get('route','oral'),
            'schedule': t.get('dose_schedule','single_dose'),
            'food': t.get('food','not_specified'),
            'formulation': t.get('formulation','tablet'),
            'population': t.get('population','healthy_adult'),
        }
        s = build_sample(smi, dose, ik, conditions, tdc, use_conditions=True)
        if s is None: continue
        X_extra.append(s); Y_extra.append(log_cd); g_extra.append(ik)
    print(f"\nLLM filtered tuples: {len(X_extra)}")
    X_tr_e3 = np.vstack([X_tr, X_extra])
    Y_tr_e3 = np.concatenate([Y_tr, Y_extra])
    g_tr_e3 = np.concatenate([g_tr, g_extra])
    r3 = eval_xgb(X_tr_e3, Y_tr_e3, g_tr_e3, X_ho_cond, Y_ho, 'E3: v10 + all LLM + cond')

    # ─── E4: v10 + drug-level median LLM + conditions ──
    # Group LLM by (ik, route, schedule, food, population) and take median log_cd
    grouped = defaultdict(list)
    for i, t in enumerate(llm):
        if not t.get('smiles'): continue
        smi, dose, cmax = t['smiles'], t.get('dose_mg',0), t.get('cmax_ng_ml',0)
        if not dose or dose<=0 or not cmax or cmax<=0: continue
        ik = smi_to_ik(smi)
        if not ik or ik in ho_iks_14: continue
        log_cd = math.log10(cmax/dose)
        if log_cd < -3 or log_cd > 3: continue
        route = normalize_condition(t.get('route','oral'))
        sched = normalize_condition(t.get('dose_schedule','single_dose'))
        food = normalize_condition(t.get('food','not_specified'))
        pop = normalize_condition(t.get('population','healthy_adult'))
        key = (ik, route, sched, food, pop)
        grouped[key].append({'smi': smi, 'dose': dose, 'log_cd': log_cd, 't': t})

    X_med, Y_med, g_med = [], [], []
    for key, entries in grouped.items():
        ik, route, sched, food, pop = key
        if len(entries) >= 1:
            # Use median log_cd and median dose
            log_cds = [e['log_cd'] for e in entries]
            doses = [e['dose'] for e in entries]
            med_log_cd = float(np.median(log_cds))
            med_dose = float(np.median(doses))
            smi = entries[0]['smi']
            conditions = {
                'route': route, 'schedule': sched, 'food': food,
                'formulation': 'tablet', 'population': pop,
            }
            s = build_sample(smi, med_dose, ik, conditions, tdc, use_conditions=True)
            if s is None: continue
            X_med.append(s); Y_med.append(med_log_cd); g_med.append(ik)
    print(f"\nLLM drug-median tuples: {len(X_med)} (from {len(grouped)} groups)")
    X_tr_e4 = np.vstack([X_tr, X_med])
    Y_tr_e4 = np.concatenate([Y_tr, Y_med])
    g_tr_e4 = np.concatenate([g_tr, g_med])
    r4 = eval_xgb(X_tr_e4, Y_tr_e4, g_tr_e4, X_ho_cond, Y_ho, 'E4: v10 + LLM-median + cond')

    # ─── E5: v10 + STRICT filtered LLM (canonical only) + cond ──
    X_str, Y_str, g_str = [], [], []
    for t in llm:
        if not t.get('smiles'): continue
        smi, dose, cmax = t['smiles'], t.get('dose_mg',0), t.get('cmax_ng_ml',0)
        if not dose or dose<=0 or not cmax or cmax<=0: continue
        # Strict: oral + single_dose + fasted + healthy only + high conf
        if t.get('route') != 'oral': continue
        if t.get('dose_schedule') != 'single_dose': continue
        if t.get('food') not in ('fasted', 'not_specified'): continue
        if t.get('population') != 'healthy_adult': continue
        if t.get('confidence') != 'high': continue
        ik = smi_to_ik(smi)
        if not ik or ik in ho_iks_14: continue
        log_cd = math.log10(cmax/dose)
        if log_cd < -3 or log_cd > 3: continue
        conditions = CANONICAL_COND
        s = build_sample(smi, dose, ik, conditions, tdc, use_conditions=True)
        if s is None: continue
        X_str.append(s); Y_str.append(log_cd); g_str.append(ik)
    print(f"\nLLM strict filter: {len(X_str)}")
    X_tr_e5 = np.vstack([X_tr, X_str])
    Y_tr_e5 = np.concatenate([Y_tr, Y_str])
    g_tr_e5 = np.concatenate([g_tr, g_str])
    r5 = eval_xgb(X_tr_e5, Y_tr_e5, g_tr_e5, X_ho_cond, Y_ho, 'E5: v10 + LLM-strict + cond')

    # ─── E6: Strict filter + drug-level median ──
    grouped2 = defaultdict(list)
    for t in llm:
        if not t.get('smiles'): continue
        smi, dose, cmax = t['smiles'], t.get('dose_mg',0), t.get('cmax_ng_ml',0)
        if not dose or dose<=0 or not cmax or cmax<=0: continue
        if t.get('route') != 'oral': continue
        if t.get('dose_schedule') != 'single_dose': continue
        if t.get('food') not in ('fasted', 'not_specified'): continue
        if t.get('population') != 'healthy_adult': continue
        if t.get('confidence') != 'high': continue
        ik = smi_to_ik(smi)
        if not ik or ik in ho_iks_14: continue
        log_cd = math.log10(cmax/dose)
        if log_cd < -3 or log_cd > 3: continue
        grouped2[ik].append({'smi': smi, 'dose': dose, 'log_cd': log_cd})

    X_med2, Y_med2, g_med2 = [], [], []
    for ik, entries in grouped2.items():
        log_cds = [e['log_cd'] for e in entries]
        doses = [e['dose'] for e in entries]
        med_log_cd = float(np.median(log_cds))
        med_dose = float(np.median(doses))
        smi = entries[0]['smi']
        s = build_sample(smi, med_dose, ik, CANONICAL_COND, tdc, use_conditions=True)
        if s is None: continue
        X_med2.append(s); Y_med2.append(med_log_cd); g_med2.append(ik)
    print(f"\nLLM strict+median: {len(X_med2)} drugs")
    X_tr_e6 = np.vstack([X_tr, X_med2])
    Y_tr_e6 = np.concatenate([Y_tr, Y_med2])
    g_tr_e6 = np.concatenate([g_tr, g_med2])
    r6 = eval_xgb(X_tr_e6, Y_tr_e6, g_tr_e6, X_ho_cond, Y_ho, 'E6: v10 + LLM-strict-median + cond')

    # ─── Summary ──
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    print(f"{'Experiment':<38s} {'N':>6s} {'CV':>7s} {'HO':>7s} {'2-fold':>8s}")
    print('-'*70)
    for label, r in [('E1: v10 baseline (no cond)', r1),
                      ('E2: v10 + cond (canonical)', r2),
                      ('E3: v10 + all LLM + cond', r3),
                      ('E4: v10 + LLM-median + cond', r4),
                      ('E5: v10 + LLM-strict + cond', r5),
                      ('E6: v10 + LLM-strict-median+cond', r6)]:
        print(f"{label:<38s} {r['n']:>6d} {r['cv']:>7.3f} {r['ho']:>7.3f} {r['f2']:>7.1f}%")
    print(f"\nSisyphus Engine: HO=3.416")
    print(f"Previous best PLM: HO=3.217")

    results = {f'E{i+1}': r for i, r in enumerate([r1,r2,r3,r4,r5,r6])}
    with open('models/llm_enriched_results.json', 'w') as f:
        json.dump({k: {kk:round(vv,3) for kk,vv in v.items()} for k,v in results.items()}, f, indent=2)

if __name__ == '__main__':
    main()
