"""Multi-arm clinical trial simulator with PK-adherence-efficacy coupling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from simulator.patient import VirtualPatient, PatientState, generate_population
from simulator.pk_engine import (
    AnalyticalPKEngine,
    PKEngine,
    multi_dose_concentration,
)
from simulator.adherence import AdherenceModel
from simulator.pharmacology import ae_probability, efficacy_probability


@dataclass
class TrialArm:
    """One arm of a clinical trial."""

    label: str
    dose_mg: float
    n_patients: int


@dataclass
class TrialProtocol:
    """Defines a (possibly multi-arm) clinical trial."""

    drug_name: str
    arms: list[TrialArm]
    interval_h: float  # dosing interval
    n_doses: int  # total planned doses per patient
    cmax_therapeutic: float  # Cmax threshold for AE sigmoid (mg/L)
    ec50: float = 0.20  # Ctrough for 50% efficacy (mg/L)
    emax: float = 0.95  # maximum efficacy probability
    hill: float = 1.0  # Hill coefficient for efficacy
    observation_times_h: list = field(
        default_factory=lambda: [0, 0.5, 1, 2, 4, 8, 12, 24]
    )
    seed: int = 42
    drug_params: Optional[dict] = None  # population PK parameter overrides


@dataclass
class ArmResult:
    """Aggregated results for one trial arm."""

    label: str
    dose_mg: float
    n_patients: int
    n_completers: int
    n_dropouts: int
    dropout_rate: float
    mean_adherence_rate: float
    total_ae_events: int

    # Population PK at steady state (from completers)
    cmax_ss_mean: float
    cmax_ss_std: float
    cmax_ss_median: float
    cmax_ss_5th: float
    cmax_ss_95th: float

    # Trough concentration stats
    ctrough_ss_mean: float
    ctrough_ss_std: float

    # Efficacy
    response_rate: float  # fraction of completers with therapeutic response

    # Steady-state flag
    ss_fraction: float  # fraction of completers that reached steady state

    # Per-patient detail
    patient_summaries: list


@dataclass
class TrialResult:
    """Complete trial output across all arms."""

    protocol: TrialProtocol
    arm_results: dict  # label -> ArmResult


def _check_steady_state(cmax_values: list[float], threshold: float = 0.10) -> bool:
    """Check if Cmax has stabilized (< threshold relative change in last 2 intervals)."""
    if len(cmax_values) < 2:
        return False
    last, prev = cmax_values[-1], cmax_values[-2]
    if prev <= 0:
        return False
    return abs(last - prev) / prev < threshold


def simulate_arm(
    arm: TrialArm,
    protocol: TrialProtocol,
    adherence_model: Optional[AdherenceModel],
    pk_engine: PKEngine,
    enable_feedback: bool,
    seed: int,
) -> ArmResult:
    """Simulate one arm of a clinical trial."""
    rng = np.random.default_rng(seed)

    patients = generate_population(
        arm.n_patients, seed=seed, drug_params=protocol.drug_params
    )

    perfect_adherence = adherence_model is None
    adh = adherence_model or AdherenceModel(p11_base=1.0, p01_base=1.0)

    patient_summaries = []

    for patient in patients:
        state = PatientState()
        last_cmax = 0.0

        for dose_idx in range(protocol.n_doses):
            t_scheduled = dose_idx * protocol.interval_h

            if state.dropped_out:
                break

            # Behavioral decision
            if perfect_adherence or dose_idx == 0:
                action = "take"
            elif enable_feedback:
                action = adh.decide(
                    rng, patient, state,
                    last_cmax, protocol.cmax_therapeutic,
                )
            else:
                action = adh.decide_no_feedback(rng, patient, state)

            if action == "take":
                # Apply dose-timing jitter
                last_dose_t = state.doses_taken[-1][0] if state.doses_taken else None
                if not perfect_adherence and dose_idx > 0:
                    actual_time = adh.jitter_dose_time(rng, t_scheduled, last_dose_t)
                else:
                    actual_time = t_scheduled
                state.doses_taken.append((actual_time, arm.dose_mg))
            elif action == "skip":
                state.doses_skipped.append((t_scheduled, arm.dose_mg))
            elif action == "dropout":
                state.dropout_time_h = t_scheduled
                break

            # Compute Cmax and Ctrough for this interval
            if state.doses_taken:
                t_eval = np.linspace(
                    t_scheduled, t_scheduled + protocol.interval_h, 50
                )
                c_interval = pk_engine.concentration(t_eval, state.doses_taken, patient)
                last_cmax = float(np.max(c_interval))
                ctrough = float(c_interval[-1])
                state.cmax_per_interval.append(last_cmax)
                state.ctrough_per_interval.append(ctrough)

        # Patient summary
        n_taken = len(state.doses_taken)
        adherence_rate = n_taken / protocol.n_doses if protocol.n_doses > 0 else 0
        ss_cmax = state.cmax_per_interval[-1] if state.cmax_per_interval else 0.0
        ss_ctrough = state.ctrough_per_interval[-1] if state.ctrough_per_interval else 0.0
        reached_ss = _check_steady_state(state.cmax_per_interval)

        # Efficacy: based on last Ctrough
        p_response = efficacy_probability(
            ss_ctrough, protocol.ec50, protocol.emax, protocol.hill
        )
        responded = rng.random() < p_response if not state.dropped_out else False

        patient_summaries.append({
            "id": patient.id,
            "sex": patient.sex,
            "age": round(patient.age_yr, 1),
            "weight": round(patient.weight_kg, 1),
            "cyp3a4": round(patient.cyp3a4_activity, 3),
            "adherence_tendency": round(patient.adherence_tendency, 3),
            "doses_taken": n_taken,
            "doses_skipped": len(state.doses_skipped),
            "adherence_rate": round(adherence_rate, 3),
            "n_ae": len(state.ae_events),
            "dropped_out": state.dropped_out,
            "ss_cmax": round(ss_cmax, 6),
            "ss_ctrough": round(ss_ctrough, 6),
            "reached_ss": reached_ss,
            "responded": responded,
            "ke": round(patient.ke, 4),
            "vd_f": round(patient.vd_f, 1),
        })

    # Aggregate
    completers = [p for p in patient_summaries if not p["dropped_out"]]
    dropouts = [p for p in patient_summaries if p["dropped_out"]]

    ss_cmax_vals = np.array([p["ss_cmax"] for p in completers if p["ss_cmax"] > 0])
    ss_ctrough_vals = np.array([p["ss_ctrough"] for p in completers if p["ss_ctrough"] > 0])
    adherence_rates = [p["adherence_rate"] for p in patient_summaries]

    n_responded = sum(1 for p in completers if p["responded"])
    n_ss = sum(1 for p in completers if p["reached_ss"])

    return ArmResult(
        label=arm.label,
        dose_mg=arm.dose_mg,
        n_patients=arm.n_patients,
        n_completers=len(completers),
        n_dropouts=len(dropouts),
        dropout_rate=len(dropouts) / len(patient_summaries) if patient_summaries else 0,
        mean_adherence_rate=float(np.mean(adherence_rates)) if adherence_rates else 0,
        total_ae_events=sum(p["n_ae"] for p in patient_summaries),
        cmax_ss_mean=float(np.mean(ss_cmax_vals)) if len(ss_cmax_vals) > 0 else 0,
        cmax_ss_std=float(np.std(ss_cmax_vals)) if len(ss_cmax_vals) > 0 else 0,
        cmax_ss_median=float(np.median(ss_cmax_vals)) if len(ss_cmax_vals) > 0 else 0,
        cmax_ss_5th=float(np.percentile(ss_cmax_vals, 5)) if len(ss_cmax_vals) > 0 else 0,
        cmax_ss_95th=float(np.percentile(ss_cmax_vals, 95)) if len(ss_cmax_vals) > 0 else 0,
        ctrough_ss_mean=float(np.mean(ss_ctrough_vals)) if len(ss_ctrough_vals) > 0 else 0,
        ctrough_ss_std=float(np.std(ss_ctrough_vals)) if len(ss_ctrough_vals) > 0 else 0,
        response_rate=n_responded / len(completers) if completers else 0,
        ss_fraction=n_ss / len(completers) if completers else 0,
        patient_summaries=patient_summaries,
    )


def simulate_trial(
    protocol: TrialProtocol,
    adherence_model: Optional[AdherenceModel] = None,
    pk_engine: Optional[PKEngine] = None,
    enable_feedback: bool = True,
) -> TrialResult:
    """Run a full multi-arm clinical trial simulation.

    Args:
        protocol: Trial design with one or more arms.
        adherence_model: Markov adherence model. None = 100% adherence.
        pk_engine: PK prediction engine. None = analytical 1-compartment.
        enable_feedback: If True, AE feedback modifies adherence.
    """
    if pk_engine is None:
        pk_engine = AnalyticalPKEngine()

    arm_results = {}
    for i, arm in enumerate(protocol.arms):
        # Each arm gets a deterministic but distinct seed
        arm_seed = protocol.seed + i * 10000
        arm_results[arm.label] = simulate_arm(
            arm, protocol, adherence_model, pk_engine, enable_feedback, arm_seed
        )

    return TrialResult(protocol=protocol, arm_results=arm_results)
