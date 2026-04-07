"""Virtual patient generation with correlated demographics and allometric PK scaling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class VirtualPatient:
    """A single virtual patient with correlated physiology + behavior."""

    id: int
    weight_kg: float
    age_yr: float
    sex: str  # "M" or "F"
    cyp3a4_activity: float  # relative to reference (1.0 = normal)
    ka: float  # absorption rate constant (1/h)
    ke: float  # elimination rate constant (1/h)
    vd_f: float  # apparent volume of distribution (L)
    tlag: float  # absorption lag time (h), 0 for IV
    adherence_tendency: float  # 0-1, baseline probability of taking a dose
    ae_sensitivity: float  # individual threshold for AE-driven skip


@dataclass
class PatientState:
    """Mutable state tracked across the simulation for one patient."""

    doses_taken: list = field(default_factory=list)  # (time_h, dose_mg)
    doses_skipped: list = field(default_factory=list)  # (time_h, dose_mg)
    ae_events: list = field(default_factory=list)  # (time_h, severity)
    cmax_per_interval: list = field(default_factory=list)
    ctrough_per_interval: list = field(default_factory=list)
    dropped_out: bool = False
    dropout_time_h: Optional[float] = None
    consecutive_skips: int = 0


def generate_population(
    n: int,
    seed: int = 42,
    drug_params: Optional[dict] = None,
) -> list[VirtualPatient]:
    """Generate N virtual patients with correlated demographics.

    Allometric scaling:
    - Vd = Vd_ref70 * (weight/70)^0.7
    - CL = CL_ref70 * (weight/70)^0.75 * cyp_activity
    - ke = CL / Vd

    Weight, age, CYP activity, and PK parameters are correlated:
    - Heavier patients -> larger Vd (allometric, not linear)
    - Female -> ~15% lower CYP3A4 activity on average
    - Age -> slight CYP reduction after 65
    - Adherence tendency sampled from Beta distribution (realistic skew)
    """
    rng = np.random.default_rng(seed)

    dp = drug_params or {
        "ka_mean": 1.5,
        "ka_cv": 0.40,
        "cl_ref70": 10.0,  # L/h at 70kg reference
        "cl_cv": 0.35,
        "vd_f_ref70": 84.0,  # L at 70kg reference
        "vd_f_cv": 0.30,
        "tlag_mean": 0.3,
        "tlag_cv": 0.50,
    }

    patients = []
    for i in range(n):
        sex = "M" if rng.random() < 0.5 else "F"
        age = float(np.clip(rng.normal(45, 15), 18, 85))
        weight = float(np.clip(
            rng.normal(80 if sex == "M" else 68, 12), 40, 140
        ))

        # CYP3A4 activity: sex effect + age effect + IIV
        cyp_base = 1.0
        if sex == "F":
            cyp_base *= 0.85
        if age > 65:
            cyp_base *= max(0.6, 1.0 - 0.01 * (age - 65))
        cyp = float(np.clip(rng.lognormal(
            np.log(cyp_base) - 0.5 * 0.3**2, 0.3
        ), 0.2, 3.0))

        # Absorption rate constant (IIV only, not weight-dependent)
        ka = float(np.clip(rng.lognormal(
            np.log(dp["ka_mean"]) - 0.5 * dp["ka_cv"] ** 2, dp["ka_cv"]
        ), 0.1, 10.0))

        # Allometric Vd: Vd = Vd_ref70 * (wt/70)^0.7
        vd_ref = dp["vd_f_ref70"] * (weight / 70.0) ** 0.7
        vd_f = float(np.clip(rng.lognormal(
            np.log(vd_ref) - 0.5 * dp["vd_f_cv"] ** 2, dp["vd_f_cv"]
        ), 10, 500))

        # Allometric CL: CL = CL_ref70 * (wt/70)^0.75 * cyp_activity
        cl_ref = dp["cl_ref70"] * (weight / 70.0) ** 0.75 * cyp
        cl = float(np.clip(rng.lognormal(
            np.log(cl_ref) - 0.5 * dp["cl_cv"] ** 2, dp["cl_cv"]
        ), 0.5, 200))

        # ke = CL / Vd
        ke = cl / vd_f

        # Absorption lag time
        tlag_mean = dp.get("tlag_mean", 0.3)
        tlag_cv = dp.get("tlag_cv", 0.50)
        if tlag_mean > 0:
            tlag = float(np.clip(rng.lognormal(
                np.log(tlag_mean) - 0.5 * tlag_cv**2, tlag_cv
            ), 0.0, 2.0))
        else:
            tlag = 0.0

        # Adherence: Beta(4, 1.5) gives mean ~0.73, right-skewed
        adherence = float(np.clip(rng.beta(4, 1.5), 0.1, 1.0))

        # AE sensitivity: individual threshold (lower = more sensitive)
        ae_sensitivity = float(np.clip(rng.lognormal(0, 0.3), 0.3, 3.0))

        patients.append(VirtualPatient(
            id=i,
            weight_kg=weight,
            age_yr=age,
            sex=sex,
            cyp3a4_activity=cyp,
            ka=ka,
            ke=ke,
            vd_f=vd_f,
            tlag=tlag,
            adherence_tendency=adherence,
            ae_sensitivity=ae_sensitivity,
        ))

    return patients
