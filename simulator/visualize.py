"""Publication-quality trial simulation plots."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from simulator.trial import TrialResult, ArmResult


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_pk_fan_chart(
    result: TrialResult,
    output_dir: str,
    filename: str = "pk_fan_chart.png",
) -> str:
    """Population PK fan chart: median + 90% PI of Cmax per arm."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    arms = sorted(result.arm_results.values(), key=lambda a: a.dose_mg)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(arms)))

    for arm, color in zip(arms, colors):
        cmax_vals = [p["ss_cmax"] for p in arm.patient_summaries if p["ss_cmax"] > 0 and not p["dropped_out"]]
        if not cmax_vals:
            continue

        vals = np.array(cmax_vals)
        # Box-style summary at each dose
        bp = ax.boxplot(
            [vals],
            positions=[arm.dose_mg],
            widths=arm.dose_mg * 0.15,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_xlabel("Dose (mg)", fontsize=12)
    ax.set_ylabel("Steady-State Cmax (mg/L)", fontsize=12)
    ax.set_title(f"{result.protocol.drug_name} — Population PK by Dose", fontsize=14)
    ax.axhline(result.protocol.cmax_therapeutic, color="red", ls="--", alpha=0.7, label="Cmax therapeutic")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_adherence_waterfall(
    arm_result: ArmResult,
    output_dir: str,
    filename: str = "adherence_waterfall.png",
) -> str:
    """Sorted bar chart of per-patient adherence rates, colored by dropout."""
    import matplotlib.pyplot as plt

    summaries = sorted(arm_result.patient_summaries, key=lambda p: p["adherence_rate"])
    rates = [p["adherence_rate"] for p in summaries]
    colors = ["#d62728" if p["dropped_out"] else "#2ca02c" for p in summaries]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(rates)), rates, color=colors, width=1.0, edgecolor="none")
    ax.set_xlabel("Patients (sorted)", fontsize=12)
    ax.set_ylabel("Adherence Rate", fontsize=12)
    ax.set_title(
        f"{arm_result.label} ({arm_result.dose_mg}mg) — Adherence Waterfall "
        f"(red = dropout, N={arm_result.n_patients})",
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color="orange", ls="--", alpha=0.6, label="80% threshold")
    ax.legend()

    _ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dose_response(
    result: TrialResult,
    output_dir: str,
    filename: str = "dose_response.png",
) -> str:
    """Dose-response curve: efficacy and AE rate vs dose across arms."""
    import matplotlib.pyplot as plt

    arms = sorted(result.arm_results.values(), key=lambda a: a.dose_mg)
    doses = [a.dose_mg for a in arms]
    response_rates = [a.response_rate for a in arms]
    ae_rates = [
        a.total_ae_events / max(1, a.n_completers) for a in arms
    ]
    dropout_rates = [a.dropout_rate for a in arms]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(doses, response_rates, "o-", color="#1f77b4", linewidth=2, markersize=8, label="Response rate")
    ax1.set_xlabel("Dose (mg)", fontsize=12)
    ax1.set_ylabel("Response Rate", fontsize=12, color="#1f77b4")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2.plot(doses, ae_rates, "s--", color="#d62728", linewidth=2, markersize=8, label="AE/completer")
    ax2.plot(doses, dropout_rates, "^:", color="#ff7f0e", linewidth=2, markersize=8, label="Dropout rate")
    ax2.set_ylabel("AE Rate / Dropout Rate", fontsize=12, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title(
        f"{result.protocol.drug_name} — Dose-Response (Efficacy vs Safety)",
        fontsize=14,
    )
    ax1.grid(axis="both", alpha=0.3)

    _ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ae_adherence_scatter(
    arm_result: ArmResult,
    output_dir: str,
    filename: str = "ae_adherence_scatter.png",
) -> str:
    """Scatter plot: AE count vs adherence rate, showing the feedback loop."""
    import matplotlib.pyplot as plt

    summaries = arm_result.patient_summaries
    ae_counts = [p["n_ae"] for p in summaries]
    adherence = [p["adherence_rate"] for p in summaries]
    dropped = [p["dropped_out"] for p in summaries]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Separate completers and dropouts
    for is_drop, label, color, marker in [
        (False, "Completer", "#2ca02c", "o"),
        (True, "Dropout", "#d62728", "x"),
    ]:
        idx = [i for i, d in enumerate(dropped) if d == is_drop]
        if idx:
            ax.scatter(
                [ae_counts[i] for i in idx],
                [adherence[i] for i in idx],
                c=color, marker=marker, alpha=0.5, label=label, s=30,
            )

    ax.set_xlabel("Number of AE Events", fontsize=12)
    ax.set_ylabel("Adherence Rate", fontsize=12)
    ax.set_title(
        f"{arm_result.label} ({arm_result.dose_mg}mg) — AE-Adherence Feedback",
        fontsize=13,
    )
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)

    _ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(
    result: TrialResult,
    output_dir: str = "data/trial_sim_plots",
    reference_arm_label: str | None = None,
) -> list[str]:
    """Generate all 4 plot types. Returns list of saved file paths."""
    paths = []
    paths.append(plot_pk_fan_chart(result, output_dir))
    paths.append(plot_dose_response(result, output_dir))

    # Pick a reference arm for per-arm plots (default: highest dose)
    if reference_arm_label and reference_arm_label in result.arm_results:
        ref_arm = result.arm_results[reference_arm_label]
    else:
        ref_arm = max(result.arm_results.values(), key=lambda a: a.dose_mg)

    paths.append(plot_adherence_waterfall(ref_arm, output_dir))
    paths.append(plot_ae_adherence_scatter(ref_arm, output_dir))

    return paths
