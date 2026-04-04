"""
PLM: Multi-Task ADME Pre-training → PK-aware Embedding → XGBoost Cmax

Architecture:
  Pre-train:  SMILES → Morgan FP 2048 → MLP(512→256→128) → 11 ADME task heads
  Predict:    128-dim PK-aware embedding + features → XGBoost → log(Cmax/dose)

Hypothesis: encoder learns PK-relevant molecular representation that generalizes
to OOD drugs better than raw Morgan fingerprints.

Usage:
    python models/pretrain_adme_xgb.py
"""

import json
import math
import warnings
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.model_selection import GroupKFold
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────

TDC_TASKS = [
    ('Solubility_AqSolDB',       'continuous'),
    ('Caco2_Wang',               'continuous'),
    ('PPBR_AZ',                  'continuous'),
    ('VDss_Lombardo',            'continuous'),
    ('Half_Life_Obach',          'continuous'),
    ('Clearance_Hepatocyte_AZ',  'continuous'),
    ('Clearance_Microsome_AZ',   'continuous'),
    ('HIA_Hou',                  'binary'),
    ('Bioavailability_Ma',       'binary'),
    ('BBB_Martins',              'binary'),
    ('PAMPA_NCATS',              'continuous'),
]
N_TASKS = len(TDC_TASKS)
FP_BITS = 4096
EMB_DIM = 128
N_PHYSCHEM = 20

XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.3,
    reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5,
    random_state=42, n_jobs=1, verbosity=0,
)


# ─── Molecular Features ──────────────────────────────────────

def smiles_to_fp(smiles, n_bits=FP_BITS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def smiles_to_physchem(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(N_PHYSCHEM, np.nan)
    def safe(fn):
        try:
            v = fn(mol)
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan
    return np.array([
        safe(Descriptors.ExactMolWt),
        safe(Descriptors.MolLogP),
        safe(Descriptors.TPSA),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol),
        safe(Descriptors.FractionCSP3),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumHeteroatoms(mol),
        safe(Descriptors.LabuteASA),
        safe(Descriptors.BertzCT),
        safe(Descriptors.Chi0v),
        safe(Descriptors.Chi1v),
        safe(Descriptors.HallKierAlpha),
        safe(Descriptors.Kappa1),
        safe(Descriptors.Kappa2),
        safe(Descriptors.MaxPartialCharge),
        safe(Descriptors.MinPartialCharge),
    ], dtype=np.float32)


def get_tdc_features(ik, tdc_adme):
    """9 ADME features from TDC lookup."""
    entry = tdc_adme.get(ik, {})
    keys = ['logS', 'caco2_logPapp', 'ppb_pct', 'vd_L_kg', 'half_life_h',
            'clearance_ul_min_mg', 'clearance_ul_min_million_cells',
            'logD', 'bioavailability_binary']
    return np.array([entry.get(k, np.nan) for k in keys], dtype=np.float32)


def compute_micropbpk(ik, tdc_adme, dose_mg):
    """Micro-PBPK features from ADME data: fa, fu, Eh, F_oral, Vd_pred, ke."""
    entry = tdc_adme.get(ik, {})
    feats = np.full(6, np.nan)

    caco2 = entry.get('caco2_logPapp')
    ppb = entry.get('ppb_pct')
    cl_hep = entry.get('clearance_ul_min_million_cells')
    cl_mic = entry.get('clearance_ul_min_mg')
    vd = entry.get('vd_L_kg')
    thalf = entry.get('half_life_h')

    # fa (fraction absorbed) from Caco2 permeability
    if caco2 is not None:
        papp = 10 ** caco2  # cm/s
        fa = papp / (papp + 1e-6)  # simplified: high perm → fa≈1
        fa = min(max(fa, 0.01), 1.0)
        feats[0] = fa

    # fu (fraction unbound) from PPB
    if ppb is not None:
        fu = max((100 - ppb) / 100, 0.001)
        feats[1] = fu

    # Eh (hepatic extraction ratio)
    Q_h = 1500  # mL/min hepatic blood flow
    if cl_hep is not None:
        # Scale: uL/min/million cells → mL/min (120M cells/g × 20g liver)
        cl_int = cl_hep * 120 * 20 / 1000  # mL/min
        Eh = cl_int / (Q_h + cl_int)
        Eh = min(max(Eh, 0.001), 0.999)
        feats[2] = Eh
    elif cl_mic is not None:
        cl_int = cl_mic * 45 * 20  # uL/min/mg × 45 mg/g × 20g → uL/min → /1000
        cl_int_ml = cl_int / 1000
        Eh = cl_int_ml / (Q_h + cl_int_ml)
        Eh = min(max(Eh, 0.001), 0.999)
        feats[2] = Eh

    # F_oral = fa × (1 - Eh)
    if feats[0] is not np.nan and feats[2] is not np.nan:
        if not np.isnan(feats[0]) and not np.isnan(feats[2]):
            feats[3] = feats[0] * (1 - feats[2])

    # Vd
    if vd is not None:
        feats[4] = vd

    # ke from t½
    if thalf is not None and thalf > 0:
        feats[5] = 0.693 / thalf

    return feats


# ─── PyTorch Model ────────────────────────────────────────────

class ADMEEncoder(nn.Module):
    def __init__(self, input_dim=FP_BITS, hidden=(512, 256, EMB_DIM),
                 n_tasks=N_TASKS, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden[-1], 1) for _ in range(n_tasks)])

    def forward(self, x):
        emb = self.encoder(x)
        return emb, [h(emb).squeeze(-1) for h in self.heads]


class SimpleDS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ─── Pre-training ─────────────────────────────────────────────

def make_dataloaders(X, Y, seed=42, val_frac=0.15, batch_size=256):
    """Create train/val DataLoaders with fixed split."""
    n = len(X)
    idx = np.random.RandomState(seed).permutation(n)
    nv = int(val_frac * n)
    tr_dl = DataLoader(SimpleDS(X[idx[nv:]], Y[idx[nv:]]), batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(SimpleDS(X[idx[:nv]], Y[idx[:nv]]), batch_size=batch_size)
    return tr_dl, va_dl


def pretrain_encoder(tr_dl, va_dl, task_types, device, epochs=150, patience=20):
    model = ADMEEncoder(input_dim=FP_BITS, hidden=(768, 384, EMB_DIM), dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Encoder params: {params:,}")

    best_vl, best_st, wait = float('inf'), None, 0

    for ep in range(epochs):
        # Train
        model.train()
        tl, nb = 0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            _, outs = model(xb)
            loss, nt = 0, 0
            for t in range(N_TASKS):
                m = ~torch.isnan(yb[:, t])
                if m.sum() < 2:
                    continue
                if task_types[t] == 'binary':
                    loss += F.binary_cross_entropy_with_logits(outs[t][m], yb[:, t][m])
                else:
                    loss += F.mse_loss(outs[t][m], yb[:, t][m])
                nt += 1
            if nt > 0:
                loss /= nt
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tl += loss.item()
                nb += 1
        tl /= max(nb, 1)

        # Val
        model.eval()
        vl, nvb = 0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                _, outs = model(xb)
                loss, nt = 0, 0
                for t in range(N_TASKS):
                    m = ~torch.isnan(yb[:, t])
                    if m.sum() < 2:
                        continue
                    if task_types[t] == 'binary':
                        loss += F.binary_cross_entropy_with_logits(outs[t][m], yb[:, t][m])
                    else:
                        loss += F.mse_loss(outs[t][m], yb[:, t][m])
                    nt += 1
                if nt > 0:
                    loss /= nt
                    vl += loss.item()
                    nvb += 1
        vl /= max(nvb, 1)
        sched.step(ep)

        if ep % 20 == 0:
            lr_now = opt.param_groups[0]['lr']
            print(f"    Epoch {ep:3d}: train={tl:.4f}  val={vl:.4f}  lr={lr_now:.1e}")

        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop @ epoch {ep}")
                break

    model.load_state_dict(best_st)
    print(f"    Best val loss: {best_vl:.4f}")
    return model


# ─── XGBoost Eval ─────────────────────────────────────────────

def eval_xgb(X_tr, Y_tr, g_tr, X_ho, Y_ho, label=""):
    """GroupKFold CV + holdout AAFE. Returns results + holdout predictions."""
    n_splits = min(5, len(set(g_tr)))
    gkf = GroupKFold(n_splits=n_splits)
    cv_preds = np.full_like(Y_tr, np.nan)

    for _, (ti, vi) in enumerate(gkf.split(X_tr, Y_tr, g_tr)):
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(X_tr[ti], Y_tr[ti])
        cv_preds[vi] = m.predict(X_tr[vi])

    cv_err = np.abs(cv_preds - Y_tr)
    cv_aafe = 10 ** np.nanmean(cv_err[np.isfinite(cv_err)])

    # Holdout
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr, Y_tr)
    ho_preds = m.predict(X_ho)

    ho_err = np.abs(ho_preds - Y_ho)
    ho_aafe = 10 ** np.mean(ho_err)
    ho_2f = np.mean(ho_err < np.log10(2)) * 100
    ho_3f = np.mean(ho_err < np.log10(3)) * 100

    print(f"  {label:<22s} CV={cv_aafe:.3f}  HO={ho_aafe:.3f}  2f={ho_2f:.1f}%  3f={ho_3f:.1f}%")
    return dict(cv_aafe=round(cv_aafe, 3), ho_aafe=round(ho_aafe, 3),
                ho_2fold=round(ho_2f, 1), ho_3fold=round(ho_3f, 1),
                ho_preds=ho_preds)


# ─── Main ─────────────────────────────────────────────────────

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── 1. Load TDC ADME ──
    print("\n[1] Loading TDC ADME datasets...")
    from tdc.single_pred import ADME

    compounds = {}  # ik14 → {smiles, labels: {task_idx: value}}
    task_types = [t[1] for t in TDC_TASKS]

    for ti, (name, _) in enumerate(TDC_TASKS):
        data = ADME(name=name)
        df = data.get_data()
        n = 0
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['Drug'])
            if mol is None:
                continue
            try:
                ik = Chem.InchiToInchiKey(Chem.MolToInchi(mol))[:14]
            except Exception:
                continue
            if ik not in compounds:
                compounds[ik] = {'smiles': Chem.MolToSmiles(mol), 'labels': {}}
            compounds[ik]['labels'][ti] = float(row['Y'])
            n += 1
        print(f"  {name}: {n}")

    print(f"  → {len(compounds)} unique compounds")

    # ── 2. Build pre-training matrix ──
    print("\n[2] Computing Morgan FPs for pre-training...")
    ik_order = []
    fp_list, label_mat = [], []

    for ik, info in compounds.items():
        fp = smiles_to_fp(info['smiles'])
        if fp is None:
            continue
        labels = np.full(N_TASKS, np.nan)
        for t, v in info['labels'].items():
            labels[t] = v
        ik_order.append(ik)
        fp_list.append(fp)
        label_mat.append(labels)

    X_pre = np.array(fp_list)
    Y_pre = np.array(label_mat)
    print(f"  Matrix: {X_pre.shape[0]} × {X_pre.shape[1]}")

    # Normalize continuous tasks (z-score)
    task_stats = {}
    for t in range(N_TASKS):
        vals = Y_pre[:, t]
        valid = vals[~np.isnan(vals)]
        task_stats[t] = (np.mean(valid), max(np.std(valid), 1e-6))
        if task_types[t] == 'continuous':
            mask = ~np.isnan(Y_pre[:, t])
            Y_pre[mask, t] = (Y_pre[mask, t] - task_stats[t][0]) / task_stats[t][1]
        n_valid = np.sum(~np.isnan(Y_pre[:, t]))
        print(f"  {TDC_TASKS[t][0]}: {n_valid} labels")

    # ── 3. Pre-train encoder ──
    print("\n[3] Pre-training ADME encoder...")
    tr_dl, va_dl = make_dataloaders(X_pre, Y_pre)
    encoder = pretrain_encoder(tr_dl, va_dl, task_types, device)
    torch.save(encoder.state_dict(), 'models/adme_encoder.pt')
    print("  → Saved models/adme_encoder.pt")

    # ── 4. Build PLM features ──
    print("\n[4] Building PLM feature matrix...")

    with open('data/curated/plm_dataset_v10_labels.json') as f:
        profiles = json.load(f)
    if isinstance(profiles, dict):
        profiles = profiles.get('profiles', [])

    with open('data/validation/holdout_definition.json') as f:
        ho = json.load(f)
    holdout_drugs = ho.get('holdout_drugs', [])

    with open('data/curated/tdc_adme_data.json') as f:
        tdc_adme = json.load(f)

    encoder.eval()
    emb_cache = {}

    def get_embedding(smi, fp):
        if smi not in emb_cache:
            with torch.no_grad():
                t = torch.tensor(fp).unsqueeze(0).to(device)
                emb_cache[smi] = encoder.encoder(t).cpu().numpy().flatten()
        return emb_cache[smi]

    def build_features(smi, dose, ik, fp):
        """Build 4 feature configs for one compound."""
        emb = get_embedding(smi, fp)
        pc = smiles_to_physchem(smi)
        adme = get_tdc_features(ik, tdc_adme)
        mpbpk = compute_micropbpk(ik, tdc_adme, dose)
        ld = np.float32(math.log10(dose))
        base = np.concatenate([pc, adme, mpbpk, [ld]])  # PhysChem + ADME + microPBPK + dose
        return {
            'fp_base':        np.concatenate([fp, base]),
            'enc_only':       np.concatenate([emb, [ld]]),
            'enc_base':       np.concatenate([emb, base]),
            'fp_enc_base':    np.concatenate([fp, emb, base]),
        }

    # ── Training data (all v10 profiles) ──
    tr_feats = {k: [] for k in ['fp_base', 'enc_only', 'enc_base', 'fp_enc_base']}
    Y_train, grp_train = [], []

    for p in profiles:
        smi = p.get('smiles')
        dose = p.get('dose_mg', 0)
        ik = p.get('ik', '')
        log_cd = p.get('log_cd')

        if not smi or not dose or dose <= 0 or log_cd is None:
            continue
        fp = smiles_to_fp(smi)
        if fp is None:
            continue

        f = build_features(smi, dose, ik, fp)
        for k in tr_feats:
            tr_feats[k].append(f[k])
        Y_train.append(log_cd)
        grp_train.append(ik)

    Y_train = np.array(Y_train, dtype=np.float32)
    grp_train = np.array(grp_train)
    print(f"  Train: {len(Y_train)} profiles ({len(set(grp_train))} drugs)")

    # ── Holdout data (from holdout_drugs ground truth) ──
    ho_feats = {k: [] for k in ['fp_base', 'enc_only', 'enc_base', 'fp_enc_base']}
    Y_ho = []
    ho_names = []

    for d in holdout_drugs:
        smi = d.get('smiles')
        dose = d.get('dose_mg', 0)
        cmax = d.get('cmax_obs_ngml', 0)
        ik = d.get('inchikey14', '')

        if not smi or not dose or dose <= 0 or not cmax or cmax <= 0:
            continue
        fp = smiles_to_fp(smi)
        if fp is None:
            continue

        f = build_features(smi, dose, ik, fp)
        for k in ho_feats:
            ho_feats[k].append(f[k])
        Y_ho.append(math.log10(cmax / dose))
        ho_names.append(d.get('name', ''))

    Y_ho = np.array(Y_ho, dtype=np.float32)
    print(f"  Holdout: {len(Y_ho)} drugs")

    # ── 5. Compare XGBoost configs ──
    print(f"\n{'='*70}")
    print("[5] XGBoost comparison")
    print(f"{'='*70}")

    config_labels = {
        'fp_base':     'FP4096+PC+ADME+uPBPK',
        'enc_only':    'Encoder(128)+dose',
        'enc_base':    'Enc+PC+ADME+uPBPK',
        'fp_enc_base': 'FP+Enc+PC+ADME+uPBPK',
    }

    results = {}
    for key in config_labels:
        X_tr = np.array(tr_feats[key], dtype=np.float32)
        X_ho = np.array(ho_feats[key], dtype=np.float32)
        X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
        X_ho = np.where(np.isinf(X_ho), np.nan, X_ho)
        results[key] = eval_xgb(
            X_tr, Y_train, grp_train,
            X_ho, Y_ho,
            label=config_labels[key]
        )

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'CV AAFE':>8s} {'HO AAFE':>8s} {'2-fold':>8s} {'3-fold':>8s}")
    print('-' * 60)
    for key, r in results.items():
        print(f"{config_labels[key]:<25s} {r['cv_aafe']:>8.3f} {r['ho_aafe']:>8.3f} {r['ho_2fold']:>7.1f}% {r['ho_3fold']:>7.1f}%")
    print('-' * 60)
    print(f"{'Sisyphus Engine':<25s} {'':>8s} {'3.416':>8s}")
    print(f"{'Previous best PLM':<25s} {'':>8s} {'3.217':>8s}")

    # ── 6. Ensemble evaluation ──
    print(f"\n{'='*70}")
    print("[6] Ensemble predictions")
    print(f"{'='*70}")

    def aafe_from_preds(preds, Y_true):
        err = np.abs(preds - Y_true)
        return (10 ** np.mean(err),
                np.mean(err < np.log10(2)) * 100,
                np.mean(err < np.log10(3)) * 100)

    # Try weighted averages of best configs
    best_configs = ['enc_base', 'fp_base', 'fp_enc_base']
    for i, k1 in enumerate(best_configs):
        for k2 in best_configs[i+1:]:
            p1 = results[k1]['ho_preds']
            p2 = results[k2]['ho_preds']
            for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
                avg = w * p1 + (1 - w) * p2
                aafe, f2, f3 = aafe_from_preds(avg, Y_ho)
                if aafe < 3.39:
                    print(f"  {w:.1f}×{config_labels[k1]} + {1-w:.1f}×{config_labels[k2]}")
                    print(f"    → HO={aafe:.3f}  2f={f2:.1f}%  3f={f3:.1f}%")

    # 3-way ensemble
    for w1 in [0.4, 0.5, 0.6]:
        for w2 in [0.2, 0.3]:
            w3 = 1 - w1 - w2
            if w3 < 0.05:
                continue
            avg = (w1 * results['enc_base']['ho_preds'] +
                   w2 * results['fp_base']['ho_preds'] +
                   w3 * results['fp_enc_base']['ho_preds'])
            aafe, f2, f3 = aafe_from_preds(avg, Y_ho)
            if aafe < 3.39:
                print(f"  3-way({w1:.1f}/{w2:.1f}/{w3:.1f}): HO={aafe:.3f}  2f={f2:.1f}%  3f={f3:.1f}%")

    # ── 7. EMB_DIM sweep ──
    print(f"\n{'='*70}")
    print("[7] Embedding dimension sweep")
    print(f"{'='*70}")

    for edim in [64, 256, 384]:
        print(f"\n  --- EMB_DIM={edim} ---")
        # Re-train encoder with different embedding dim
        enc2 = ADMEEncoder(input_dim=FP_BITS, hidden=(768, 384, edim), dropout=0.2).to(device)
        opt2 = torch.optim.AdamW(enc2.parameters(), lr=5e-4, weight_decay=1e-3)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=30, T_mult=2)

        best_vl2, best_st2, wait2 = float('inf'), None, 0
        for ep in range(150):
            enc2.train()
            tl, nb = 0, 0
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                _, outs = enc2(xb)
                loss, nt = 0, 0
                for t in range(N_TASKS):
                    m = ~torch.isnan(yb[:, t])
                    if m.sum() < 2: continue
                    if task_types[t] == 'binary':
                        loss += F.binary_cross_entropy_with_logits(outs[t][m], yb[:, t][m])
                    else:
                        loss += F.mse_loss(outs[t][m], yb[:, t][m])
                    nt += 1
                if nt > 0:
                    loss /= nt; opt2.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(enc2.parameters(), 1.0)
                    opt2.step(); tl += loss.item(); nb += 1
            tl /= max(nb, 1)
            enc2.eval()
            vl, nvb = 0, 0
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    _, outs = enc2(xb)
                    loss, nt = 0, 0
                    for t in range(N_TASKS):
                        m2 = ~torch.isnan(yb[:, t])
                        if m2.sum() < 2: continue
                        if task_types[t] == 'binary':
                            loss += F.binary_cross_entropy_with_logits(outs[t][m2], yb[:, t][m2])
                        else:
                            loss += F.mse_loss(outs[t][m2], yb[:, t][m2])
                        nt += 1
                    if nt > 0:
                        loss /= nt; vl += loss.item(); nvb += 1
            vl /= max(nvb, 1)
            sched2.step(ep)
            if vl < best_vl2:
                best_vl2 = vl
                best_st2 = {k: v.clone() for k, v in enc2.state_dict().items()}
                wait2 = 0
            else:
                wait2 += 1
                if wait2 >= 20: break

        enc2.load_state_dict(best_st2)
        enc2.eval()
        print(f"  val_loss={best_vl2:.4f}")

        # Extract embeddings and evaluate Enc+base config
        emb_cache2 = {}
        def get_emb2(smi, fp):
            if smi not in emb_cache2:
                with torch.no_grad():
                    t2 = torch.tensor(fp).unsqueeze(0).to(device)
                    emb_cache2[smi] = enc2.encoder(t2).cpu().numpy().flatten()
            return emb_cache2[smi]

        # Training features
        X_tr2 = []
        for p in profiles:
            smi = p.get('smiles'); dose = p.get('dose_mg', 0)
            ik = p.get('ik', ''); log_cd = p.get('log_cd')
            if not smi or not dose or dose <= 0 or log_cd is None: continue
            fp = smiles_to_fp(smi)
            if fp is None: continue
            emb2 = get_emb2(smi, fp)
            pc = smiles_to_physchem(smi)
            adme = get_tdc_features(ik, tdc_adme)
            mpbpk = compute_micropbpk(ik, tdc_adme, dose)
            ld = np.float32(math.log10(dose))
            X_tr2.append(np.concatenate([emb2, pc, adme, mpbpk, [ld]]))

        # Holdout features
        X_ho2 = []
        for d in holdout_drugs:
            smi = d.get('smiles'); dose = d.get('dose_mg', 0)
            ik = d.get('inchikey14', '')
            cmax = d.get('cmax_obs_ngml', 0)
            if not smi or not dose or dose <= 0 or not cmax or cmax <= 0: continue
            fp = smiles_to_fp(smi)
            if fp is None: continue
            emb2 = get_emb2(smi, fp)
            pc = smiles_to_physchem(smi)
            adme = get_tdc_features(ik, tdc_adme)
            mpbpk = compute_micropbpk(ik, tdc_adme, dose)
            ld = np.float32(math.log10(dose))
            X_ho2.append(np.concatenate([emb2, pc, adme, mpbpk, [ld]]))

        X_tr2 = np.where(np.isinf(np.array(X_tr2, dtype=np.float32)), np.nan, np.array(X_tr2, dtype=np.float32))
        X_ho2 = np.where(np.isinf(np.array(X_ho2, dtype=np.float32)), np.nan, np.array(X_ho2, dtype=np.float32))

        eval_xgb(X_tr2, Y_train, grp_train, X_ho2, Y_ho, label=f"Enc({edim})+base")

    # ── Save final results ──
    # Remove non-serializable preds before saving
    save_results = {}
    for k, v in results.items():
        save_results[k] = {kk: vv for kk, vv in v.items() if kk != 'ho_preds'}
    with open('models/pretrain_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    print("\n→ Results saved to models/pretrain_results.json")


if __name__ == '__main__':
    main()
