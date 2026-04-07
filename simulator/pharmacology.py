"""Pharmacological effect models: adverse events (Cmax-driven) and efficacy (Ctrough-driven)."""

from __future__ import annotations

import math


def ae_probability(
    cmax: float,
    cmax_therapeutic: float,
    patient_sensitivity: float,
    steepness: float = 3.0,
) -> float:
    """Sigmoid AE probability as a function of Cmax.

    P(AE) = 1 / (1 + exp(-steepness * (Cmax/Cmax_therapeutic - sensitivity)))

    When Cmax/Cmax_therapeutic > sensitivity -> P(AE) > 0.5
    """
    if cmax_therapeutic <= 0:
        return 0.0
    ratio = cmax / cmax_therapeutic
    x = steepness * (ratio - patient_sensitivity)
    # Clip to avoid overflow
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def efficacy_probability(
    ctrough: float,
    ec50: float,
    emax: float = 0.95,
    hill: float = 1.0,
) -> float:
    """Emax model for efficacy based on trough concentration.

    P(response) = Emax * Ctrough^hill / (EC50^hill + Ctrough^hill)

    Args:
        ctrough: Trough concentration (at end of dosing interval).
        ec50: Concentration producing 50% of max effect.
        emax: Maximum response probability (default 0.95).
        hill: Hill coefficient / steepness (default 1.0).

    Returns:
        Probability of therapeutic response in [0, emax].
    """
    if ctrough <= 0 or ec50 <= 0:
        return 0.0
    c_h = ctrough ** hill
    ec_h = ec50 ** hill
    return emax * c_h / (ec_h + c_h)
