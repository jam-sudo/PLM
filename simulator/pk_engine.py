"""PK engines: analytical 1-compartment with lag time, and PLM adapter stub.

Performance notes:
    multi_dose_concentration uses 2D broadcasting (doses × timepoints) to
    eliminate the Python for-loop over doses.  When Numba is available, a
    JIT-compiled inner loop provides an additional ~5-10x speedup.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from simulator.patient import VirtualPatient


# ──────────────────────────────────────────────────────────────────────
# Optional Numba JIT (graceful fallback if not installed)
# ──────────────────────────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _njit(f=None, **kw):
        """No-op decorator when Numba is unavailable."""
        return f if f is not None else lambda fn: fn


# ──────────────────────────────────────────────────────────────────────
# Core analytical functions
# ──────────────────────────────────────────────────────────────────────

def pk_concentration(
    t: float | np.ndarray,
    dose_mg: float,
    ka: float,
    ke: float,
    vd_f: float,
    tlag: float = 0.0,
) -> float | np.ndarray:
    """1-compartment oral model with lag time: C(t) after a single dose.

    C(t) = (Dose * ka) / (Vd_F * (ka - ke)) * (exp(-ke*t') - exp(-ka*t'))
    where t' = max(0, t - tlag)
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    t_eff = np.maximum(0.0, t - tlag)

    if abs(ka - ke) < 1e-10:
        # Degenerate case: ka ~ ke -> L'Hopital limit
        c = (dose_mg / vd_f) * ka * t_eff * np.exp(-ke * t_eff)
    else:
        coeff = (dose_mg * ka) / (vd_f * (ka - ke))
        c = coeff * (np.exp(-ke * t_eff) - np.exp(-ka * t_eff))

    # Zero out concentrations before lag time
    c[t < tlag] = 0.0
    return c if c.size > 1 else float(c[0])


@_njit(cache=True)
def _multi_dose_numba(
    t: np.ndarray,
    dose_times: np.ndarray,
    dose_amounts: np.ndarray,
    ka: float,
    ke: float,
    vd_f: float,
    tlag: float,
) -> np.ndarray:
    """Numba-JIT'd multi-dose superposition (tight loop, no Python overhead)."""
    n_t = t.shape[0]
    n_d = dose_times.shape[0]
    c = np.zeros(n_t)
    degenerate = abs(ka - ke) < 1e-10

    for j in range(n_d):
        t_dose = dose_times[j]
        dose_mg = dose_amounts[j]
        if degenerate:
            coeff_a = (dose_mg / vd_f) * ka
        else:
            coeff_b = (dose_mg * ka) / (vd_f * (ka - ke))
        for i in range(n_t):
            dt = t[i] - t_dose
            if dt <= 0.0:
                continue
            t_eff = max(0.0, dt - tlag)
            if t_eff <= 0.0:
                continue
            if degenerate:
                c[i] += coeff_a * t_eff * np.exp(-ke * t_eff)
            else:
                c[i] += coeff_b * (np.exp(-ke * t_eff) - np.exp(-ka * t_eff))
    return c


def _multi_dose_vectorized(
    t: np.ndarray,
    dose_times: np.ndarray,
    dose_amounts: np.ndarray,
    ka: float,
    ke: float,
    vd_f: float,
    tlag: float,
) -> np.ndarray:
    """Vectorized multi-dose via 2D broadcasting (no Python loop over doses)."""
    # dt[d, t] = t[t] - dose_times[d]  (shape: n_doses × n_timepoints)
    dt = t[np.newaxis, :] - dose_times[:, np.newaxis]
    t_eff = np.maximum(0.0, dt - tlag)
    mask = dt > 0  # only contribute after dosing

    if abs(ka - ke) < 1e-10:
        coeffs = (dose_amounts / vd_f) * ka  # shape (n_doses,)
        contrib = coeffs[:, np.newaxis] * t_eff * np.exp(-ke * t_eff)
    else:
        coeffs = (dose_amounts * ka) / (vd_f * (ka - ke))
        contrib = coeffs[:, np.newaxis] * (
            np.exp(-ke * t_eff) - np.exp(-ka * t_eff)
        )

    contrib = np.where(mask, contrib, 0.0)
    return contrib.sum(axis=0)


def multi_dose_concentration(
    t: float | np.ndarray,
    doses: list[tuple[float, float]],  # [(time_h, dose_mg), ...]
    ka: float,
    ke: float,
    vd_f: float,
    tlag: float = 0.0,
) -> np.ndarray:
    """Superposition of multiple oral doses with lag time.

    Uses Numba JIT when available, otherwise 2D broadcasting.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if not doses:
        return np.zeros_like(t)

    dose_times = np.array([d[0] for d in doses], dtype=np.float64)
    dose_amounts = np.array([d[1] for d in doses], dtype=np.float64)

    if _HAS_NUMBA:
        return _multi_dose_numba(t, dose_times, dose_amounts, ka, ke, vd_f, tlag)
    return _multi_dose_vectorized(t, dose_times, dose_amounts, ka, ke, vd_f, tlag)


# ──────────────────────────────────────────────────────────────────────
# PKEngine protocol (for swappable engines)
# ──────────────────────────────────────────────────────────────────────

@runtime_checkable
class PKEngine(Protocol):
    """Interface for PK concentration prediction."""

    def concentration(
        self,
        t: np.ndarray,
        doses: list[tuple[float, float]],
        patient: VirtualPatient,
    ) -> np.ndarray:
        """Predict concentrations at times t given dose history and patient."""
        ...


class AnalyticalPKEngine:
    """1-compartment oral model using patient-level PK parameters."""

    def concentration(
        self,
        t: np.ndarray,
        doses: list[tuple[float, float]],
        patient: VirtualPatient,
    ) -> np.ndarray:
        return multi_dose_concentration(
            t, doses, patient.ka, patient.ke, patient.vd_f, patient.tlag
        )


class PLMPKEngine:
    """PLM-driven PK engine: predicts C(t) from SMILES + dose alone.

    Workflow:
    1. Predict Cmax from SMILES via trained XGBoost (fp_enc_base pipeline)
    2. Derive 1-compartment PK parameters from predicted Cmax
    3. Generate C(t) via analytical model with superposition

    This enables clinical trial simulation for novel compounds where
    no PK parameters are known — only SMILES and dose are required.

    Usage:
        engine = PLMPKEngine(smiles="CC(=O)Oc1ccccc1C(=O)O")
        protocol = TrialProtocol(drug_name="aspirin", ...)
        result = simulate_trial(protocol, pk_engine=engine)
    """

    # Population defaults for profile shape (from PLM training data median)
    DEFAULT_KA = 1.2        # 1/h, typical oral absorption rate
    DEFAULT_TLAG = 0.3      # h, typical absorption lag
    DEFAULT_THALF_H = 8.0   # h, population median elimination half-life

    def __init__(
        self,
        smiles: str,
        model_dir: str | None = None,
        ka: float | None = None,
        thalf_h: float | None = None,
    ):
        """Initialize PLMPKEngine.

        Args:
            smiles: SMILES string of the drug
            model_dir: path to PLM model directory (default: ROOT/models/)
            ka: override absorption rate constant (1/h)
            thalf_h: override elimination half-life (h)
        """
        self.smiles = smiles
        self._ka_override = ka
        self._thalf_override = thalf_h
        self._model = None
        self._encoder = None
        self._features = None
        self._cmax_cache: dict[float, float] = {}  # dose → predicted Cmax

        self._load_pipeline(model_dir)

    def _load_pipeline(self, model_dir: str | None) -> None:
        """Load XGBoost model and ADME encoder for Cmax prediction."""
        import json
        import math
        from pathlib import Path

        import torch
        import xgboost as xgb
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")

        root = Path(model_dir) if model_dir else Path(__file__).parent.parent
        models_dir = root if model_dir else root / "models"

        # 1. Compute Morgan fingerprint
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self.smiles}")

        FP_BITS = 4096
        fp = np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS),
            dtype=np.float32,
        )

        # 2. ADME encoder embedding
        encoder_path = models_dir / "adme_encoder.pt"
        if encoder_path.exists():
            # Inline encoder definition (same as s12_v12_retrain.py)
            import torch.nn as nn
            EMB_DIM = 128

            class _ADMEEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    layers = []
                    prev = FP_BITS
                    for h in (768, 384, EMB_DIM):
                        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                                   nn.ReLU(), nn.Dropout(0.2)]
                        prev = h
                    self.encoder = nn.Sequential(*layers)
                    self.heads = nn.ModuleList(
                        [nn.Linear(EMB_DIM, 1) for _ in range(11)]
                    )

                def forward(self, x):
                    emb = self.encoder(x)
                    return emb, [h(emb).squeeze(-1) for h in self.heads]

            device = "cpu"
            encoder = _ADMEEncoder().to(device)
            state = torch.load(str(encoder_path), map_location=device,
                               weights_only=True)
            encoder.load_state_dict(state)
            encoder.eval()

            with torch.no_grad():
                t_fp = torch.tensor(fp).unsqueeze(0)
                emb, _ = encoder(t_fp)
                emb = emb.cpu().numpy().flatten()
        else:
            emb = np.zeros(128, dtype=np.float32)

        # 3. Physicochemical descriptors (20)
        def _safe(fn):
            try:
                v = fn(mol)
                return v if (v is not None and np.isfinite(v)) else np.nan
            except Exception:
                return np.nan

        pc = np.array([
            _safe(Descriptors.ExactMolWt), _safe(Descriptors.MolLogP),
            _safe(Descriptors.TPSA), Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol), Descriptors.NumAromaticRings(mol),
            _safe(Descriptors.FractionCSP3), Descriptors.HeavyAtomCount(mol),
            Descriptors.NumHeteroatoms(mol), _safe(Descriptors.LabuteASA),
            _safe(Descriptors.BertzCT), _safe(Descriptors.Chi0v),
            _safe(Descriptors.Chi1v), _safe(Descriptors.HallKierAlpha),
            _safe(Descriptors.Kappa1), _safe(Descriptors.Kappa2),
            _safe(Descriptors.MaxPartialCharge),
            _safe(Descriptors.MinPartialCharge),
        ], dtype=np.float32)

        # 4. TDC + µPBPK features (placeholder NaN — filled from TDC if available)
        tdc_feats = np.full(9, np.nan, dtype=np.float32)
        upbpk_feats = np.full(6, np.nan, dtype=np.float32)

        # Try loading TDC data
        tdc_path = root / "data/curated/tdc_adme_data.json"
        if tdc_path.exists():
            from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
            try:
                inchi = MolToInchi(mol)
                ik = InchiToInchiKey(inchi)[:14] if inchi else None
                if ik:
                    tdc = json.loads(tdc_path.read_text())
                    tdc_ik = {k[:14]: v for k, v in tdc.items()}
                    if ik in tdc_ik:
                        e = tdc_ik[ik]
                        keys = [
                            "logS", "caco2_logPapp", "ppb_pct", "vd_L_kg",
                            "half_life_h", "clearance_ul_min_mg",
                            "clearance_ul_min_million_cells", "logD",
                            "bioavailability_binary",
                        ]
                        tdc_feats = np.array(
                            [e.get(k, np.nan) for k in keys], dtype=np.float32
                        )
                        # µPBPK derivation
                        caco2 = e.get("caco2_logPapp")
                        ppb = e.get("ppb_pct")
                        cl_hep = e.get("clearance_ul_min_million_cells")
                        cl_mic = e.get("clearance_ul_min_mg")
                        vd = e.get("vd_L_kg")
                        th = e.get("half_life_h")
                        if caco2 is not None:
                            papp = 10**caco2
                            upbpk_feats[0] = min(max(papp / (papp + 1e-6), 0.01), 1.0)
                        if ppb is not None:
                            upbpk_feats[1] = max((100 - ppb) / 100, 0.001)
                        Q = 1500
                        if cl_hep is not None:
                            ci = cl_hep * 120 * 20 / 1000
                            upbpk_feats[2] = min(max(ci / (Q + ci), 1e-3), 0.999)
                        elif cl_mic is not None:
                            ci = cl_mic * 45 * 20 / 1000
                            upbpk_feats[2] = min(max(ci / (Q + ci), 1e-3), 0.999)
                        if (not np.isnan(upbpk_feats[0])
                                and not np.isnan(upbpk_feats[2])):
                            upbpk_feats[3] = upbpk_feats[0] * (1 - upbpk_feats[2])
                        if vd is not None:
                            upbpk_feats[4] = vd
                        if th is not None and th > 0:
                            upbpk_feats[5] = 0.693 / th
                            if self._thalf_override is None:
                                self._thalf_override = th
            except Exception:
                pass

        # 5. Assemble base feature vector (dose appended at prediction time)
        self._features_base = np.concatenate([
            fp, emb, pc, tdc_feats, upbpk_feats,
        ]).astype(np.float32)
        self._features_base[~np.isfinite(self._features_base)] = np.nan

        # 6. Load XGBoost model
        xgb_path = models_dir / "b1" / "plm_cmax_model.pkl"
        if not xgb_path.exists():
            # Train a fresh model on v12 for this session
            self._model = self._train_model(root)
        else:
            import pickle
            with open(xgb_path, "rb") as f:
                self._model = pickle.load(f)

    def _train_model(self, root) -> "xgb.XGBRegressor":
        """Train XGBoost on v12 and save for reuse."""
        import json
        import math
        from pathlib import Path

        import xgboost as xgb
        from sklearn.model_selection import GroupKFold

        v12_path = root / "data/curated/plm_dataset_v12_chembl.json"
        if not v12_path.exists():
            v12_path = root / "data/curated/plm_dataset_v11_llm.json"

        v12 = json.loads(v12_path.read_text())
        ho = json.loads(
            (root / "data/validation/holdout_definition.json").read_text()
        )
        ho_iks = set((k or "")[:14] for k in ho["holdout_inchikeys"])
        tdc = {}
        tdc_path = root / "data/curated/tdc_adme_data.json"
        if tdc_path.exists():
            tdc = {
                k[:14]: v
                for k, v in json.loads(tdc_path.read_text()).items()
            }

        # Build features using the same pipeline as s12_v12_retrain
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        import torch
        import torch.nn as nn

        FP_BITS = 4096
        encoder_path = root / "models/adme_encoder.pt"

        class _E(nn.Module):
            def __init__(self):
                super().__init__()
                layers, prev = [], FP_BITS
                for h in (768, 384, 128):
                    layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                               nn.ReLU(), nn.Dropout(0.2)]
                    prev = h
                self.encoder = nn.Sequential(*layers)
                self.heads = nn.ModuleList(
                    [nn.Linear(128, 1) for _ in range(11)]
                )
            def forward(self, x):
                emb = self.encoder(x)
                return emb, [h(emb).squeeze(-1) for h in self.heads]

        enc = _E()
        if encoder_path.exists():
            enc.load_state_dict(
                torch.load(str(encoder_path), map_location="cpu",
                           weights_only=True)
            )
        enc.eval()

        fp_cache, emb_cache = {}, {}
        X, y = [], []
        for row in v12:
            ik = (row.get("ik") or "")[:14]
            if ik in ho_iks:
                continue
            log_cd = row.get("log_cd")
            if log_cd is None:
                cmax = row.get("cmax_ngml")
                dose = row.get("dose_mg")
                if not cmax or not dose or dose <= 0:
                    continue
                log_cd = math.log10(cmax / dose)
            smi = row["smiles"]
            if smi not in fp_cache:
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                fp_cache[smi] = np.array(
                    AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=FP_BITS),
                    dtype=np.float32,
                )
            fp = fp_cache[smi]
            if smi not in emb_cache:
                with torch.no_grad():
                    e, _ = enc(torch.tensor(fp).unsqueeze(0))
                    emb_cache[smi] = e.cpu().numpy().flatten()
            emb = emb_cache[smi]

            m = Chem.MolFromSmiles(smi)
            def _s(fn):
                try:
                    v = fn(m)
                    return v if (v is not None and np.isfinite(v)) else np.nan
                except Exception:
                    return np.nan
            pc = np.array([
                _s(Descriptors.ExactMolWt), _s(Descriptors.MolLogP),
                _s(Descriptors.TPSA), Descriptors.NumHDonors(m),
                Descriptors.NumHAcceptors(m), Descriptors.NumRotatableBonds(m),
                Descriptors.RingCount(m), Descriptors.NumAromaticRings(m),
                _s(Descriptors.FractionCSP3), Descriptors.HeavyAtomCount(m),
                Descriptors.NumHeteroatoms(m), _s(Descriptors.LabuteASA),
                _s(Descriptors.BertzCT), _s(Descriptors.Chi0v),
                _s(Descriptors.Chi1v), _s(Descriptors.HallKierAlpha),
                _s(Descriptors.Kappa1), _s(Descriptors.Kappa2),
                _s(Descriptors.MaxPartialCharge),
                _s(Descriptors.MinPartialCharge),
            ], dtype=np.float32)

            e_tdc = tdc.get(ik, {})
            tdc_k = [
                "logS", "caco2_logPapp", "ppb_pct", "vd_L_kg", "half_life_h",
                "clearance_ul_min_mg", "clearance_ul_min_million_cells",
                "logD", "bioavailability_binary",
            ]
            tdc_f = np.array(
                [e_tdc.get(k, np.nan) for k in tdc_k], dtype=np.float32
            )
            upk = np.full(6, np.nan, dtype=np.float32)
            ld = np.float32(math.log10(max(row["dose_mg"], 1e-6)))
            feat = np.concatenate([fp, emb, pc, tdc_f, upk, [ld]])
            feat[~np.isfinite(feat)] = np.nan
            X.append(feat)
            y.append(float(log_cd))

        X = np.stack(X)
        y_arr = np.array(y, dtype=np.float32)

        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.01,
            subsample=0.8, colsample_bytree=0.3,
            reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5,
            n_jobs=8, verbosity=0, tree_method="hist", random_state=42,
        )
        model.fit(X, y_arr)

        # Save for reuse
        import pickle
        out_path = root / "models/b1/plm_cmax_model.pkl"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(model, f)

        return model

    def predict_cmax(self, dose_mg: float) -> float:
        """Predict Cmax (ng/mL) for a given dose.

        Returns the PLM-predicted Cmax based on SMILES + dose features.
        This is the core PLM prediction — structure → exposure.
        """
        import math

        if dose_mg in self._cmax_cache:
            return self._cmax_cache[dose_mg]

        log_dose = np.float32(math.log10(max(dose_mg, 1e-6)))
        features = np.concatenate([self._features_base, [log_dose]])
        features = features.reshape(1, -1).astype(np.float32)
        features[~np.isfinite(features)] = np.nan

        log_cd = float(self._model.predict(features)[0])
        cmax_ngml = 10**log_cd * dose_mg
        self._cmax_cache[dose_mg] = cmax_ngml
        return cmax_ngml

    def _derive_pk_params(self, dose_mg: float) -> tuple[float, float, float, float]:
        """Derive 1-compartment PK parameters from predicted Cmax.

        Returns (ka, ke, vd_f, tlag) in units compatible with AnalyticalPKEngine.
        Concentration units: mg/L (simulator standard).
        """
        cmax_ngml = self.predict_cmax(dose_mg)
        cmax_mgL = cmax_ngml / 1000.0  # ng/mL → mg/L

        ka = self._ka_override if self._ka_override else self.DEFAULT_KA
        thalf = (self._thalf_override if self._thalf_override
                 else self.DEFAULT_THALF_H)
        ke = 0.693 / thalf
        tlag = self.DEFAULT_TLAG

        # Derive Vd/F from Cmax = (dose * ka) / (Vd_F * (ka - ke)) * peak_factor
        # For 1-compartment: tmax = ln(ka/ke) / (ka - ke)
        # peak_factor = exp(-ke*tmax) - exp(-ka*tmax)
        if abs(ka - ke) < 1e-10:
            tmax = 1.0 / ke
            peak_factor = tmax * ke * np.exp(-ke * tmax)
            vd_f = dose_mg * peak_factor / (cmax_mgL + 1e-10)
        else:
            tmax = np.log(ka / ke) / (ka - ke)
            peak_factor = np.exp(-ke * tmax) - np.exp(-ka * tmax)
            vd_f = (dose_mg * ka) / (cmax_mgL * (ka - ke) + 1e-10) * peak_factor

        # Sanity bounds
        vd_f = max(1.0, min(vd_f, 50000.0))

        return ka, ke, vd_f, tlag

    def concentration(
        self,
        t: np.ndarray,
        doses: list[tuple[float, float]],
        patient: VirtualPatient,
    ) -> np.ndarray:
        """Predict C(t) from SMILES + dose using PLM-derived PK parameters.

        The PLM model predicts Cmax, which is used to derive Vd/F.
        Patient IIV is applied as multiplicative scaling on ka, ke, Vd/F.
        """
        if not doses:
            return np.zeros_like(t)

        # Use the first dose amount to derive base PK parameters
        base_dose = doses[0][1]
        ka_base, ke_base, vd_base, tlag = self._derive_pk_params(base_dose)

        # Apply patient IIV as ratios relative to population
        # Patient's ka, ke, vd_f encode individual variability
        ka_ratio = patient.ka / self.DEFAULT_KA if self.DEFAULT_KA > 0 else 1.0
        ke_ratio = patient.ke / (0.693 / self.DEFAULT_THALF_H) if self.DEFAULT_THALF_H > 0 else 1.0
        vd_ratio = patient.vd_f / 84.0  # 84 L is the default Vd_ref in patient.py

        ka = ka_base * ka_ratio
        ke = ke_base * ke_ratio
        vd_f = vd_base * vd_ratio
        tlag_p = max(0.0, patient.tlag)

        return multi_dose_concentration(t, doses, ka, ke, vd_f, tlag_p)
