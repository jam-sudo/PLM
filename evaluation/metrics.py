"""
PLM Evaluation Metrics

Primary: Cmax AAFE (comparable to Sisyphus)
Secondary: AUC AAFE, tmax MAE, C(t) log-RMSE
"""

import numpy as np
from typing import Optional


def aafe(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Absolute Average Fold Error (geometric mean fold error)."""
    log_fe = np.abs(np.log10(predicted / observed))
    return 10 ** np.mean(log_fe)


def pct_within_nfold(observed: np.ndarray, predicted: np.ndarray, n: float = 2.0) -> float:
    """Percentage of predictions within n-fold of observed."""
    fold_error = np.maximum(predicted / observed, observed / predicted)
    return 100.0 * np.mean(fold_error <= n)


def ct_log_rmse(observed_ct: np.ndarray, predicted_ct: np.ndarray) -> float:
    """RMSE of log10 concentrations across all timepoints.
    
    Args:
        observed_ct: (N_drugs, N_timepoints) array
        predicted_ct: (N_drugs, N_timepoints) array
    """
    # Mask zeros/negatives
    mask = (observed_ct > 0) & (predicted_ct > 0)
    log_obs = np.log10(np.where(mask, observed_ct, 1.0))
    log_pred = np.log10(np.where(mask, predicted_ct, 1.0))
    diff = np.where(mask, log_obs - log_pred, 0.0)
    return np.sqrt(np.sum(diff**2) / np.sum(mask))


def cmax_from_ct(ct_profile: np.ndarray) -> float:
    """Extract Cmax from a concentration-time profile."""
    return np.max(ct_profile)


def tmax_from_ct(ct_profile: np.ndarray, timepoints: np.ndarray) -> float:
    """Extract tmax from a concentration-time profile."""
    return timepoints[np.argmax(ct_profile)]


def auc_trapezoidal(ct_profile: np.ndarray, timepoints: np.ndarray) -> float:
    """Calculate AUC using trapezoidal rule."""
    return np.trapz(ct_profile, timepoints)


def full_report(
    observed_cmax: np.ndarray,
    predicted_cmax: np.ndarray,
    observed_auc: Optional[np.ndarray] = None,
    predicted_auc: Optional[np.ndarray] = None,
) -> dict:
    """Generate full evaluation report."""
    report = {
        "cmax_aafe": float(aafe(observed_cmax, predicted_cmax)),
        "cmax_pct_2fold": float(pct_within_nfold(observed_cmax, predicted_cmax, 2.0)),
        "cmax_pct_3fold": float(pct_within_nfold(observed_cmax, predicted_cmax, 3.0)),
        "n_drugs": len(observed_cmax),
    }
    if observed_auc is not None and predicted_auc is not None:
        report["auc_aafe"] = float(aafe(observed_auc, predicted_auc))
        report["auc_pct_2fold"] = float(pct_within_nfold(observed_auc, predicted_auc, 2.0))
    return report
