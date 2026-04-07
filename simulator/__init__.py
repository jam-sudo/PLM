"""Clinical Trial Simulator — PK-driven adherence and efficacy modeling."""

from simulator.patient import VirtualPatient, PatientState, generate_population
from simulator.pk_engine import (
    AnalyticalPKEngine,
    PLMPKEngine,
    pk_concentration,
    multi_dose_concentration,
)
from simulator.adherence import AdherenceModel
from simulator.pharmacology import ae_probability, efficacy_probability
from simulator.trial import TrialProtocol, TrialArm, ArmResult, TrialResult, simulate_trial

__all__ = [
    "VirtualPatient",
    "PatientState",
    "generate_population",
    "AnalyticalPKEngine",
    "PLMPKEngine",
    "pk_concentration",
    "multi_dose_concentration",
    "AdherenceModel",
    "ae_probability",
    "efficacy_probability",
    "TrialProtocol",
    "TrialArm",
    "ArmResult",
    "TrialResult",
    "simulate_trial",
]
