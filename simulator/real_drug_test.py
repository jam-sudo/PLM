"""Pick a random real drug and run the trial simulator with published PK parameters."""

from __future__ import annotations

import numpy as np

from simulator.trial import TrialProtocol, TrialArm, simulate_trial
from simulator.adherence import AdherenceModel
from simulator.visualize import generate_all_plots, plot_dose_response
import os

# ── Real drug PK parameter library (from FDA labels / published PopPK) ──

REAL_DRUGS = {
    "Metoprolol": {
        "description": "Beta-blocker, CYP2D6 substrate, IR tablet",
        "indication": "Hypertension, angina",
        "doses_mg": [25, 50, 100, 200],
        "interval_h": 12.0,  # BID
        "n_doses": 60,       # 30 days BID
        "drug_params": {
            "ka_mean": 1.2, "ka_cv": 0.40,
            "cl_ref70": 65.0,    # L/h (high first-pass, CL/F)
            "cl_cv": 0.50,       # high CV due to CYP2D6 polymorphism
            "vd_f_ref70": 290.0, # L
            "vd_f_cv": 0.30,
            "tlag_mean": 0.20,
            "tlag_cv": 0.40,
        },
        "cmax_therapeutic": 0.30,  # mg/L (~300 ng/mL)
        "ec50": 0.05,              # mg/L (BP lowering EC50)
    },
    "Sildenafil": {
        "description": "PDE5 inhibitor, CYP3A4 substrate, IR tablet",
        "indication": "Erectile dysfunction",
        "doses_mg": [25, 50, 100, 200],
        "interval_h": 24.0,  # PRN but modeled as QD
        "n_doses": 30,
        "drug_params": {
            "ka_mean": 1.8, "ka_cv": 0.45,
            "cl_ref70": 41.0,    # L/h
            "cl_cv": 0.35,
            "vd_f_ref70": 105.0, # L
            "vd_f_cv": 0.30,
            "tlag_mean": 0.25,
            "tlag_cv": 0.50,
        },
        "cmax_therapeutic": 0.45,  # mg/L (~450 ng/mL at 100mg)
        "ec50": 0.10,
    },
    "Metformin": {
        "description": "Biguanide, renal elimination (no CYP), IR tablet",
        "indication": "Type 2 diabetes",
        "doses_mg": [250, 500, 1000, 2000],
        "interval_h": 12.0,  # BID
        "n_doses": 60,
        "drug_params": {
            "ka_mean": 1.5, "ka_cv": 0.35,
            "cl_ref70": 30.0,    # L/h (renal, ~500 mL/min)
            "cl_cv": 0.25,       # lower CV (renal, less polymorphism)
            "vd_f_ref70": 63.0,  # L (~0.9 L/kg)
            "vd_f_cv": 0.25,
            "tlag_mean": 0.30,
            "tlag_cv": 0.40,
        },
        "cmax_therapeutic": 4.0,  # mg/L (4 ug/mL at 1000mg)
        "ec50": 1.0,              # mg/L (HbA1c effect)
    },
    "Warfarin": {
        "description": "VKA anticoagulant, CYP2C9/VKORC1, NARROW TI",
        "indication": "Anticoagulation (DVT/PE, AF)",
        "doses_mg": [1, 2, 5, 10],
        "interval_h": 24.0,  # QD
        "n_doses": 30,
        "drug_params": {
            "ka_mean": 2.0, "ka_cv": 0.30,    # rapid absorption
            "cl_ref70": 0.20,                   # L/h (very low, long t1/2)
            "cl_cv": 0.55,                      # HIGH CV: CYP2C9/VKORC1
            "vd_f_ref70": 10.0,                 # L (~0.14 L/kg, highly bound)
            "vd_f_cv": 0.20,
            "tlag_mean": 0.25,
            "tlag_cv": 0.40,
        },
        "cmax_therapeutic": 3.0,   # mg/L (narrow window)
        "ec50": 1.0,               # mg/L (INR target)
    },
    "Ibuprofen": {
        "description": "NSAID, CYP2C9 substrate, IR tablet",
        "indication": "Pain, inflammation",
        "doses_mg": [200, 400, 600, 800],
        "interval_h": 8.0,   # TID
        "n_doses": 90,       # 30 days TID
        "drug_params": {
            "ka_mean": 2.5, "ka_cv": 0.40,    # very rapid absorption
            "cl_ref70": 3.5,                    # L/h
            "cl_cv": 0.30,
            "vd_f_ref70": 10.0,                 # L (~0.12 L/kg, highly bound)
            "vd_f_cv": 0.25,
            "tlag_mean": 0.15,
            "tlag_cv": 0.50,
        },
        "cmax_therapeutic": 40.0,  # mg/L (40 ug/mL at 400mg)
        "ec50": 15.0,              # mg/L (COX inhibition)
    },
}


def run_random_drug(seed: int | None = None):
    """Randomly pick a drug and run the full simulation."""
    rng = np.random.default_rng(seed)
    drug_name = rng.choice(list(REAL_DRUGS.keys()))
    drug = REAL_DRUGS[drug_name]

    print("=" * 75)
    print(f"  RANDOMLY SELECTED: {drug_name}")
    print(f"  {drug['description']}")
    print(f"  Indication: {drug['indication']}")
    print("=" * 75)
    print()

    # Build protocol
    arms = [
        TrialArm(label=f"{d}mg", dose_mg=d, n_patients=300)
        for d in drug["doses_mg"]
    ]

    protocol = TrialProtocol(
        drug_name=drug_name,
        arms=arms,
        interval_h=drug["interval_h"],
        n_doses=drug["n_doses"],
        cmax_therapeutic=drug["cmax_therapeutic"],
        ec50=drug["ec50"],
        emax=0.95,
        hill=1.0,
        seed=123,
        drug_params=drug["drug_params"],
    )

    regimen_map = {8.0: "TID", 12.0: "BID", 24.0: "QD"}
    regimen = regimen_map.get(drug["interval_h"], f"q{drug['interval_h']}h")
    days = int(drug["n_doses"] * drug["interval_h"] / 24)

    print(f"Regimen: {regimen} x {days} days ({drug['n_doses']} doses)")
    print(f"Arms: {', '.join(a.label for a in arms)}")
    print(f"N per arm: 300")
    print(f"Cmax therapeutic: {drug['cmax_therapeutic']} mg/L")
    print(f"EC50: {drug['ec50']} mg/L")
    print()

    # ── Scenario 1: Perfect adherence ──
    print(">>> Scenario 1: Perfect adherence")
    result_perfect = simulate_trial(protocol, adherence_model=None)
    _print_table(result_perfect)

    # ── Scenario 2: Markov + Feedback ──
    print(">>> Scenario 2: Markov adherence + PK-AE feedback")
    adh = AdherenceModel(
        p11_base=0.92, p01_base=0.50,
        dropout_threshold=7,
        timing_jitter_h=1.5,
    )
    result_feedback = simulate_trial(protocol, adherence_model=adh, enable_feedback=True)
    _print_table(result_feedback)

    # ── Scenario 3: Markov only (no feedback) ──
    print(">>> Scenario 3: Markov adherence only (no PK-AE feedback)")
    result_markov = simulate_trial(protocol, adherence_model=adh, enable_feedback=False)
    _print_table(result_markov)

    # ── Head-to-head comparison ──
    print()
    print("=" * 75)
    print(f"HEAD-TO-HEAD: {drug['doses_mg'][2]}mg arm across scenarios")
    print("=" * 75)
    mid_label = f"{drug['doses_mg'][2]}mg"
    p = result_perfect.arm_results[mid_label]
    m = result_markov.arm_results[mid_label]
    f = result_feedback.arm_results[mid_label]

    header = f"{'Metric':<28} {'Perfect':>14} {'Markov':>14} {'Feedback':>14}"
    print(header)
    print("-" * len(header))
    rows = [
        ("Completers", f"{p.n_completers}/300", f"{m.n_completers}/300", f"{f.n_completers}/300"),
        ("Dropout %", f"{p.dropout_rate:.1%}", f"{m.dropout_rate:.1%}", f"{f.dropout_rate:.1%}"),
        ("Mean adherence", f"{p.mean_adherence_rate:.1%}", f"{m.mean_adherence_rate:.1%}", f"{f.mean_adherence_rate:.1%}"),
        ("AE events", f"{p.total_ae_events}", f"{m.total_ae_events}", f"{f.total_ae_events}"),
        ("Response rate", f"{p.response_rate:.1%}", f"{m.response_rate:.1%}", f"{f.response_rate:.1%}"),
        ("SS Cmax mean (mg/L)", f"{p.cmax_ss_mean:.4f}", f"{m.cmax_ss_mean:.4f}", f"{f.cmax_ss_mean:.4f}"),
        ("SS Ctrough mean (mg/L)", f"{p.ctrough_ss_mean:.4f}", f"{m.ctrough_ss_mean:.4f}", f"{f.ctrough_ss_mean:.4f}"),
        ("SS Cmax CV%", _cv(p), _cv(m), _cv(f)),
    ]
    for name, *vals in rows:
        print(f"{name:<28}" + "".join(f"{v:>14}" for v in vals))

    # ── Clinical interpretation ──
    print()
    print("=" * 75)
    print("CLINICAL INTERPRETATION")
    print("=" * 75)

    # Find optimal dose (highest response rate with <20% dropout)
    feedback_arms = sorted(result_feedback.arm_results.values(), key=lambda a: a.dose_mg)
    optimal = None
    for arm in reversed(feedback_arms):
        if arm.dropout_rate < 0.20:
            optimal = arm
            break
    if optimal is None:
        optimal = feedback_arms[0]

    print(f"  Optimal dose (feedback model): {optimal.label}")
    print(f"    Response rate: {optimal.response_rate:.1%}")
    print(f"    Dropout rate:  {optimal.dropout_rate:.1%}")
    print(f"    Mean adherence: {optimal.mean_adherence_rate:.1%}")
    print(f"    AE events/patient: {optimal.total_ae_events / optimal.n_patients:.1f}")

    # Feedback vs perfect delta
    opt_perfect = result_perfect.arm_results[optimal.label]
    rr_delta = optimal.response_rate - opt_perfect.response_rate
    print(f"  Response rate delta (feedback vs perfect): {rr_delta:+.1%}")
    print(f"  → {'Adherence reduces real-world efficacy' if rr_delta < 0 else 'Comparable efficacy'}")

    # Generate plots
    print()
    plot_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "trial_sim_plots", drug_name.lower()
    )
    print(f"Generating plots in {plot_dir}...")
    paths = generate_all_plots(result_feedback, plot_dir, reference_arm_label=mid_label)
    for path in paths:
        print(f"  {path}")

    return drug_name, result_perfect, result_markov, result_feedback


def _cv(arm) -> str:
    if arm.cmax_ss_mean > 0:
        return f"{arm.cmax_ss_std / arm.cmax_ss_mean * 100:.1f}%"
    return "N/A"


def _print_table(result):
    arms = sorted(result.arm_results.values(), key=lambda a: a.dose_mg)
    header = f"  {'Arm':<10} {'Comp':>6} {'Drop%':>7} {'Adh%':>7} {'AE':>6} {'Resp%':>7} {'Cmax':>10} {'Ctrough':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for a in arms:
        print(
            f"  {a.label:<10} {a.n_completers:>6} {a.dropout_rate:>6.1%} "
            f"{a.mean_adherence_rate:>6.1%} {a.total_ae_events:>6} "
            f"{a.response_rate:>6.1%} {a.cmax_ss_mean:>10.4f} {a.ctrough_ss_mean:>10.4f}"
        )
    print()


if __name__ == "__main__":
    # Use current time as seed for true randomness
    import time
    run_random_drug(seed=int(time.time()))
