"""CLI entry point: 4-arm dose-finding demo with comparison scenarios."""

from __future__ import annotations

import json
import os

from simulator.trial import TrialProtocol, TrialArm, simulate_trial
from simulator.adherence import AdherenceModel
from simulator.visualize import generate_all_plots


# Atorvastatin-like PK parameters (allometric scaling)
ATORVASTATIN_PARAMS = {
    "ka_mean": 1.2,
    "ka_cv": 0.40,
    "cl_ref70": 35.0,   # L/h at 70kg (atorvastatin CL/F ~ 625 mL/min)
    "cl_cv": 0.35,
    "vd_f_ref70": 380.0,  # L at 70kg
    "vd_f_cv": 0.30,
    "tlag_mean": 0.25,
    "tlag_cv": 0.50,
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def make_protocol(
    drug_name: str = "Atorvastatin-like",
    doses: list[float] | None = None,
    n_per_arm: int = 200,
    n_doses: int = 30,
    cmax_therapeutic: float = 0.50,
    ec50: float = 0.05,
    drug_params: dict | None = None,
    seed: int = 42,
) -> TrialProtocol:
    """Build a multi-arm protocol."""
    if doses is None:
        doses = [10, 20, 40, 80]

    arms = [
        TrialArm(label=f"{d}mg", dose_mg=d, n_patients=n_per_arm)
        for d in doses
    ]

    return TrialProtocol(
        drug_name=drug_name,
        arms=arms,
        interval_h=24.0,
        n_doses=n_doses,
        cmax_therapeutic=cmax_therapeutic,
        ec50=ec50,
        emax=0.95,
        hill=1.0,
        seed=seed,
        drug_params=drug_params or ATORVASTATIN_PARAMS,
    )


def print_arm_table(label: str, results: dict) -> None:
    """Print a formatted comparison table across arms."""
    arms = sorted(results.values(), key=lambda a: a.dose_mg)
    header = f"{'Metric':<25}" + "".join(f"{a.label:>14}" for a in arms)
    print(header)
    print("-" * len(header))

    rows = [
        ("Completers", [f"{a.n_completers}/{a.n_patients}" for a in arms]),
        ("Dropout %", [f"{a.dropout_rate:.1%}" for a in arms]),
        ("Mean adherence", [f"{a.mean_adherence_rate:.1%}" for a in arms]),
        ("AE events", [f"{a.total_ae_events}" for a in arms]),
        ("Response rate", [f"{a.response_rate:.1%}" for a in arms]),
        ("SS Cmax mean", [f"{a.cmax_ss_mean:.4f}" for a in arms]),
        ("SS Cmax CV%", [
            f"{a.cmax_ss_std / a.cmax_ss_mean * 100:.1f}%" if a.cmax_ss_mean > 0 else "N/A"
            for a in arms
        ]),
        ("SS Cmax 90% PI", [
            f"[{a.cmax_ss_5th:.3f}, {a.cmax_ss_95th:.3f}]" for a in arms
        ]),
        ("SS Ctrough mean", [f"{a.ctrough_ss_mean:.4f}" for a in arms]),
        ("SS reached %", [f"{a.ss_fraction:.0%}" for a in arms]),
    ]

    for name, vals in rows:
        print(f"{name:<25}" + "".join(f"{v:>14}" for v in vals))


def print_scenario_comparison(
    arm_label: str,
    perfect: dict,
    markov: dict,
    feedback: dict,
) -> None:
    """Compare 3 scenarios for a single arm."""
    p = perfect[arm_label]
    m = markov[arm_label]
    f = feedback[arm_label]

    header = f"{'Metric':<25} {'Perfect':>14} {'Markov':>14} {'Feedback':>14}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Completers", p.n_completers, m.n_completers, f.n_completers),
        ("Dropout %", f"{p.dropout_rate:.1%}", f"{m.dropout_rate:.1%}", f"{f.dropout_rate:.1%}"),
        ("Mean adherence", f"{p.mean_adherence_rate:.1%}", f"{m.mean_adherence_rate:.1%}", f"{f.mean_adherence_rate:.1%}"),
        ("AE events", p.total_ae_events, m.total_ae_events, f.total_ae_events),
        ("Response rate", f"{p.response_rate:.1%}", f"{m.response_rate:.1%}", f"{f.response_rate:.1%}"),
        ("SS Cmax mean", f"{p.cmax_ss_mean:.4f}", f"{m.cmax_ss_mean:.4f}", f"{f.cmax_ss_mean:.4f}"),
    ]

    for row in rows:
        name = row[0]
        vals = row[1:]
        print(f"{name:<25}" + "".join(f"{v!s:>14}" for v in vals))


def run_demo() -> None:
    """Run the full 4-arm dose-finding demo."""
    print("=" * 75)
    print("CLINICAL TRIAL SIMULATOR — Iterated POC")
    print("=" * 75)

    protocol = make_protocol()
    print(f"Drug: {protocol.drug_name}")
    print(f"Regimen: QD x {protocol.n_doses} days")
    print(f"Arms: {', '.join(a.label for a in protocol.arms)}")
    print(f"N per arm: {protocol.arms[0].n_patients}")
    print()

    # Scenario A: Perfect adherence
    print("Running Scenario A: Perfect adherence...")
    result_perfect = simulate_trial(protocol, adherence_model=None)

    # Scenario B: Markov, no feedback
    print("Running Scenario B: Markov adherence (no feedback)...")
    markov = AdherenceModel(p11_base=0.92, p01_base=0.50, dropout_threshold=7)
    result_markov = simulate_trial(protocol, adherence_model=markov, enable_feedback=False)

    # Scenario C: Markov + PK-AE feedback
    print("Running Scenario C: Markov + PK-AE feedback...")
    result_feedback = simulate_trial(protocol, adherence_model=markov, enable_feedback=True)

    # --- Tables ---
    print()
    print("=" * 75)
    print("DOSE-FINDING: Perfect Adherence")
    print("=" * 75)
    print_arm_table("Perfect", result_perfect.arm_results)

    print()
    print("=" * 75)
    print("DOSE-FINDING: With PK-AE Feedback")
    print("=" * 75)
    print_arm_table("Feedback", result_feedback.arm_results)

    print()
    print("=" * 75)
    print("SCENARIO COMPARISON (40mg arm)")
    print("=" * 75)
    print_scenario_comparison(
        "40mg",
        result_perfect.arm_results,
        result_markov.arm_results,
        result_feedback.arm_results,
    )

    # --- Narrow TI drug ---
    print()
    print("=" * 75)
    print("NARROW THERAPEUTIC INDEX (Cmax_therapeutic = 0.20)")
    print("=" * 75)
    protocol_narrow = make_protocol(
        drug_name="Narrow-TI Drug",
        cmax_therapeutic=0.20,
        ec50=0.03,
    )
    result_narrow = simulate_trial(protocol_narrow, adherence_model=markov, enable_feedback=True)
    print_arm_table("Narrow-TI Feedback", result_narrow.arm_results)

    # Key insight
    fb_40 = result_feedback.arm_results["40mg"]
    nb_40 = result_narrow.arm_results["40mg"]
    print()
    print("KEY INSIGHT: TI-dependent dropout at 40mg")
    print(f"  Wide TI  feedback dropout: {fb_40.dropout_rate:.1%}")
    print(f"  Narrow TI feedback dropout: {nb_40.dropout_rate:.1%}")
    print(f"  Delta: {nb_40.dropout_rate - fb_40.dropout_rate:+.1%}")

    # --- Plots ---
    print()
    print("Generating plots...")
    plot_dir = os.path.join(OUTPUT_DIR, "trial_sim_plots")

    paths = generate_all_plots(result_feedback, plot_dir, reference_arm_label="40mg")
    for p in paths:
        print(f"  Saved: {p}")

    # Also make dose-response for narrow TI
    from simulator.visualize import plot_dose_response
    p = plot_dose_response(result_narrow, plot_dir, "dose_response_narrow_ti.png")
    print(f"  Saved: {p}")

    # --- JSON export ---
    export_path = os.path.join(OUTPUT_DIR, "trial_sim_results.json")
    export = {}
    for name, result in [
        ("perfect", result_perfect),
        ("markov_only", result_markov),
        ("feedback", result_feedback),
        ("narrow_ti_feedback", result_narrow),
    ]:
        export[name] = {}
        for arm_label, arm in result.arm_results.items():
            export[name][arm_label] = {
                "n_completers": arm.n_completers,
                "n_dropouts": arm.n_dropouts,
                "dropout_rate": round(arm.dropout_rate, 4),
                "mean_adherence": round(arm.mean_adherence_rate, 4),
                "total_ae": arm.total_ae_events,
                "response_rate": round(arm.response_rate, 4),
                "cmax_mean": round(arm.cmax_ss_mean, 6),
                "cmax_std": round(arm.cmax_ss_std, 6),
                "cmax_5th": round(arm.cmax_ss_5th, 6),
                "cmax_95th": round(arm.cmax_ss_95th, 6),
                "ctrough_mean": round(arm.ctrough_ss_mean, 6),
                "ctrough_std": round(arm.ctrough_ss_std, 6),
            }

    with open(export_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nResults exported to {export_path}")


if __name__ == "__main__":
    run_demo()
