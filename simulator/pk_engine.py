"""PK engines: analytical 1-compartment with lag time, and PLM adapter stub."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from simulator.patient import VirtualPatient


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


def multi_dose_concentration(
    t: float | np.ndarray,
    doses: list[tuple[float, float]],  # [(time_h, dose_mg), ...]
    ka: float,
    ke: float,
    vd_f: float,
    tlag: float = 0.0,
) -> np.ndarray:
    """Superposition of multiple oral doses with lag time."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    c = np.zeros_like(t)
    for t_dose, dose_mg in doses:
        dt = t - t_dose
        mask = dt > 0
        if np.any(mask):
            contrib = pk_concentration(dt[mask], dose_mg, ka, ke, vd_f, tlag)
            c[mask] += np.atleast_1d(contrib)
    return c


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
    """Stub: plug in a trained PLM model to predict C-t profiles.

    Production usage:
        engine = PLMPKEngine(model_path="models/novel_phase1.pkl")
        protocol.pk_engine = engine

    The engine would:
    1. Take SMILES + dose from the TrialProtocol
    2. Predict log10(C/dose) on the standard 13-point grid
    3. Interpolate to arbitrary query times t
    """

    STANDARD_GRID_H = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]

    def __init__(self, model_path: str | None = None, smiles: str | None = None):
        self.model_path = model_path
        self.smiles = smiles
        self._model = None
        if model_path is not None:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load a trained PLM XGBoost model."""
        try:
            import pickle
            with open(path, "rb") as f:
                self._model = pickle.load(f)
        except Exception as e:
            raise RuntimeError(
                f"PLMPKEngine: could not load model from {path}: {e}"
            )

    def concentration(
        self,
        t: np.ndarray,
        doses: list[tuple[float, float]],
        patient: VirtualPatient,
    ) -> np.ndarray:
        """Predict C-t using PLM model, with superposition for multiple doses.

        For now, falls back to analytical engine if no model is loaded.
        """
        if self._model is None:
            # Fallback to analytical
            return AnalyticalPKEngine().concentration(t, doses, patient)

        # TODO: implement full PLM prediction pipeline:
        # 1. Compute Morgan FP from self.smiles
        # 2. Build feature vector [fp + log10(dose) + route + form + food]
        # 3. Predict log10(C/dose) at 13 standard timepoints
        # 4. Convert to absolute concentrations
        # 5. Interpolate to query times t
        # 6. Superpose for multiple doses
        raise NotImplementedError(
            "Full PLM integration pending. Use AnalyticalPKEngine for now."
        )
