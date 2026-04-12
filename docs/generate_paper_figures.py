"""Generate publication figures and tables for PLM paper.

Figures:
  2: Predicted vs observed Cmax scatter (holdout)
  3: PLM vs Sisyphus per-drug comparison
  4: Conformal prediction intervals forest plot
  5: Conditional coverage by error quartile
  6: Error decomposition (NL PK, bias, Tanimoto)

Tables:
  6: AAFE by drug class (from I7 diagnostic)
  7: Negative results summary
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path("/home/jam/PLM")
OUTDIR = ROOT / "docs/figures"
OUTDIR.mkdir(exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Color palette
C_PLM = "#2563EB"      # blue
C_SIS = "#DC2626"      # red
C_COVER = "#16A34A"    # green (covered)
C_MISS = "#DC2626"     # red (not covered)
C_GRAY = "#6B7280"


def load_data():
    s13 = json.load(open(ROOT / "models/b1/s13_uq_results.json"))
    diag = json.load(open(ROOT / "data/validation/plm_vs_sisyphus_diagnostic.json"))
    ho = json.load(open(ROOT / "data/validation/holdout_definition.json"))
    return s13, diag, ho


def fig2_scatter(s13):
    """Predicted vs observed Cmax scatter plot."""
    drugs = s13["per_drug"]
    y_true = np.array([d["y_true"] for d in drugs])
    y_pred = np.array([d["y_pred"] for d in drugs])
    errors = np.array([d["error"] for d in drugs])

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Fold boundaries
    lim = [-2.5, 3.0]
    ax.plot(lim, lim, "k-", lw=1, alpha=0.5, label="Perfect prediction")
    for fold, ls in [(2, "--"), (3, ":")]:
        offset = math.log10(fold)
        ax.plot(lim, [lim[0]+offset, lim[1]+offset], color=C_GRAY, ls=ls, lw=0.7, alpha=0.6)
        ax.plot(lim, [lim[0]-offset, lim[1]-offset], color=C_GRAY, ls=ls, lw=0.7, alpha=0.6)

    # Color by error magnitude
    colors = np.where(errors < math.log10(2), C_COVER,
             np.where(errors < math.log10(3), "#F59E0B", C_MISS))

    ax.scatter(y_true, y_pred, c=colors, s=30, alpha=0.75, edgecolors="white",
               linewidths=0.3, zorder=3)

    # Annotate worst outliers
    for d in drugs:
        if d["error"] > 1.1:
            ax.annotate(d["drug"], (d["y_true"], d["y_pred"]),
                       fontsize=6, alpha=0.7, ha="left",
                       xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Observed log$_{10}$(Cmax/dose)")
    ax.set_ylabel("Predicted log$_{10}$(Cmax/dose)")
    ax.set_title("PLM Holdout Predictions (N=97)")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")

    # Legend
    handles = [
        mpatches.Patch(color=C_COVER, label=f"<2-fold ({sum(errors < math.log10(2))})"),
        mpatches.Patch(color="#F59E0B", label=f"2-3-fold ({sum((errors >= math.log10(2)) & (errors < math.log10(3)))})"),
        mpatches.Patch(color=C_MISS, label=f">3-fold ({sum(errors >= math.log10(3))})"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    # Metrics annotation
    aafe = s13["holdout"]["ensemble_aafe"]
    f2 = s13["holdout"]["fold2_pct"]
    ax.text(0.97, 0.03, f"AAFE = {aafe:.2f}\n2-fold = {f2:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRAY, alpha=0.9))

    fig.savefig(OUTDIR / "fig2_scatter.png")
    fig.savefig(OUTDIR / "fig2_scatter.pdf")
    plt.close(fig)
    print("  Fig 2: scatter plot saved")


def fig3_plm_vs_sisyphus(s13, diag):
    """Per-drug |error| comparison: PLM vs Sisyphus."""
    rows = diag["per_drug_rows"]
    rows_sorted = sorted(rows, key=lambda r: r["plm_abs_err"] - r["sis_abs_err"])

    names = [r["name"] for r in rows_sorted]
    plm_err = np.array([r["plm_abs_err"] for r in rows_sorted])
    sis_err = np.array([r["sis_abs_err"] for r in rows_sorted])
    delta = plm_err - sis_err  # negative = PLM better

    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(len(names))

    colors = np.where(delta < 0, C_PLM, C_SIS)
    ax.barh(y, delta, color=colors, height=0.7, alpha=0.8)
    ax.axvline(0, color="black", lw=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("|PLM error| - |Sisyphus error| (log$_{10}$ units)")
    ax.set_title(f"Per-Drug Error Comparison (PLM wins {sum(delta < 0)}/97)")
    ax.invert_yaxis()

    # Legend
    handles = [
        mpatches.Patch(color=C_PLM, label="PLM better"),
        mpatches.Patch(color=C_SIS, label="Sisyphus better"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.savefig(OUTDIR / "fig3_plm_vs_sis.png")
    fig.savefig(OUTDIR / "fig3_plm_vs_sis.pdf")
    plt.close(fig)
    print("  Fig 3: PLM vs Sisyphus saved")


def fig4_forest(s13):
    """Conformal prediction intervals forest plot."""
    drugs = s13["per_drug"]
    # Sort by predicted value
    drugs_sorted = sorted(drugs, key=lambda d: d["y_pred"])

    fig, ax = plt.subplots(figsize=(6, 10))
    y = np.arange(len(drugs_sorted))

    for i, d in enumerate(drugs_sorted):
        lo, hi = d["ci90"]
        color = C_COVER if d["covered"] else C_MISS
        ax.plot([lo, hi], [i, i], color=color, lw=0.8, alpha=0.6)
        ax.plot(d["y_pred"], i, "o", color=color, markersize=2.5, alpha=0.7)
        ax.plot(d["y_true"], i, "s", color="black", markersize=2, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels([d["drug"] for d in drugs_sorted], fontsize=4.5)
    ax.set_xlabel("log$_{10}$(Cmax/dose)")
    ax.set_title(f"90% Conformal Intervals — Coverage {s13['holdout']['coverage']*100:.1f}%")

    handles = [
        plt.Line2D([0], [0], color=C_COVER, lw=2, label="Covered"),
        plt.Line2D([0], [0], color=C_MISS, lw=2, label="Not covered"),
        plt.Line2D([0], [0], marker="s", color="black", lw=0, markersize=4, label="True value"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7)

    fig.savefig(OUTDIR / "fig4_forest.png")
    fig.savefig(OUTDIR / "fig4_forest.pdf")
    plt.close(fig)
    print("  Fig 4: forest plot saved")


def fig5_conditional_coverage(s13):
    """Conditional coverage by error quartile."""
    cc = s13["conditional_coverage"]
    labels = list(cc.keys())
    coverages = [cc[l]["coverage"] for l in labels]
    ns = [cc[l]["n"] for l in labels]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, coverages, color=[C_COVER, C_COVER, C_COVER, C_MISS],
                  alpha=0.8, edgecolor="white", linewidth=0.5)

    ax.axhline(0.9, color=C_GRAY, ls="--", lw=1, label="Nominal 90%")
    ax.axhline(0.85, color=C_MISS, ls=":", lw=1, alpha=0.5, label="Target 85%")

    for i, (cov, n) in enumerate(zip(coverages, ns)):
        ax.text(i, cov + 0.02, f"{cov:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\n(lowest error)", "Q2", "Q3", "Q4\n(highest error)"],
                       fontsize=8)
    ax.set_ylabel("Empirical Coverage")
    ax.set_ylim(0, 1.15)
    ax.set_title("Conditional Coverage by Error Quartile")
    ax.legend(fontsize=8, loc="center right")

    fig.savefig(OUTDIR / "fig5_conditional.png")
    fig.savefig(OUTDIR / "fig5_conditional.pdf")
    plt.close(fig)
    print("  Fig 5: conditional coverage saved")


def fig6_error_decomposition(diag):
    """Multi-panel error decomposition."""
    rows = diag["per_drug_rows"]
    plm_err = np.array([r["plm_log_err"] for r in rows])  # signed
    plm_abs = np.array([r["plm_abs_err"] for r in rows])
    tanimoto = np.array([r["tanimoto_max_train"] for r in rows])
    is_nl = np.array([r["is_nonlinear_pk"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    # Panel A: NL vs Linear PK
    ax = axes[0]
    nl_err = plm_abs[is_nl]
    lin_err = plm_abs[~is_nl]
    bp = ax.boxplot([lin_err, nl_err], labels=["Linear\n(N=86)", "Nonlinear\n(N=11)"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(C_PLM)
    bp["boxes"][1].set_facecolor("#F59E0B")
    for b in bp["boxes"]:
        b.set_alpha(0.6)
    ax.set_ylabel("|log$_{10}$ error|")
    ax.set_title("A. Error by PK Linearity")
    nl_aafe = 10**np.mean(nl_err)
    lin_aafe = 10**np.mean(lin_err)
    ax.text(0.95, 0.95, f"Linear AAFE: {lin_aafe:.2f}\nNonlinear AAFE: {nl_aafe:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(fc="white", ec=C_GRAY, alpha=0.8))

    # Panel B: Error vs Tanimoto
    ax = axes[1]
    ax.scatter(tanimoto, plm_abs, c=C_PLM, s=20, alpha=0.5, edgecolors="white", linewidths=0.3)
    # Fit line
    z = np.polyfit(tanimoto, plm_abs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(tanimoto.min(), tanimoto.max(), 50)
    ax.plot(x_line, p(x_line), "--", color=C_MISS, lw=1.2, alpha=0.8)
    from scipy.stats import pearsonr
    r, pval = pearsonr(tanimoto, plm_abs)
    ax.set_xlabel("Max Tanimoto to Training Set")
    ax.set_ylabel("|log$_{10}$ error|")
    ax.set_title("B. Error vs Chemical Similarity")
    ax.text(0.95, 0.95, f"r = {r:.3f}\np = {pval:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(fc="white", ec=C_GRAY, alpha=0.8))

    # Panel C: Bias distribution
    ax = axes[2]
    ax.hist(plm_err, bins=25, color=C_PLM, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=1)
    mean_bias = np.mean(plm_err)
    ax.axvline(mean_bias, color=C_MISS, lw=1.5, ls="--")
    ax.set_xlabel("Signed log$_{10}$ error (+ = overprediction)")
    ax.set_ylabel("Count")
    ax.set_title("C. Prediction Bias Distribution")
    n_over = sum(plm_err > 0)
    ax.text(0.95, 0.95, f"Mean bias: +{mean_bias:.3f}\n{n_over}/97 overpredicted",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(fc="white", ec=C_GRAY, alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig6_error_decomp.png")
    fig.savefig(OUTDIR / "fig6_error_decomp.pdf")
    plt.close(fig)
    print("  Fig 6: error decomposition saved")


def table6_drug_class(diag):
    """Compute per-class AAFE from diagnostic data."""
    # Manual drug class assignments for 97 holdout drugs
    # Based on therapeutic class / mechanism
    class_map = {
        "paroxetine": "SSRI/SNRI", "sertraline": "SSRI/SNRI",
        "venlafaxine": "SSRI/SNRI", "duloxetine": "SSRI/SNRI",
        "fluticasone": "Steroid", "budesonide": "Steroid",
        "prednisolone": "Steroid",
        "atorvastatin": "Statin", "rosuvastatin": "Statin",
        "simvastatin": "Statin", "pravastatin": "Statin",
        "lovastatin": "Statin",
        "ibuprofen": "NSAID", "naproxen": "NSAID",
        "celecoxib": "NSAID", "diclofenac": "NSAID",
        "amlodipine": "Antihypertensive", "losartan": "Antihypertensive",
        "valsartan": "Antihypertensive", "irbesartan": "Antihypertensive",
        "telmisartan": "Antihypertensive", "olmesartan": "Antihypertensive",
        "imatinib": "TKI", "erlotinib": "TKI",
        "sorafenib": "TKI", "sunitinib": "TKI",
        "phenytoin": "Antiepileptic", "carbamazepine": "Antiepileptic",
        "lamotrigine": "Antiepileptic", "levetiracetam": "Antiepileptic",
    }

    rows = diag["per_drug_rows"]
    class_errors = {}
    for r in rows:
        cls = class_map.get(r["name"], "Other")
        if cls == "Other":
            continue
        class_errors.setdefault(cls, []).append(r["plm_abs_err"])

    print("\n  Table 6: Drug Class AAFE")
    print(f"  {'Class':<20} {'N':>3} {'AAFE':>7} {'Worst Drug'}")
    print(f"  {'-'*50}")

    table_data = {}
    for cls in sorted(class_errors.keys(), key=lambda c: -10**np.mean(class_errors[c])):
        errs = np.array(class_errors[cls])
        aafe = 10**np.mean(errs)
        # Find worst drug in class
        worst = max((r for r in rows if class_map.get(r["name"]) == cls),
                    key=lambda r: r["plm_abs_err"])
        table_data[cls] = {"n": len(errs), "aafe": round(aafe, 2),
                           "worst": worst["name"],
                           "worst_aafe": round(10**worst["plm_abs_err"], 2)}
        print(f"  {cls:<20} {len(errs):>3} {aafe:>7.2f} {worst['name']}")

    # Save
    out_path = OUTDIR / "table6_drug_class.json"
    with open(out_path, "w") as f:
        json.dump(table_data, f, indent=2)
    print(f"  Table 6 data saved to {out_path}")
    return table_data


def table7_negative_results():
    """Print negative results summary table."""
    print("\n  Table 7: Negative Results Summary")
    table = [
        ("Chemical repr.", "MolFormer (F2), Tanimoto retrieval (F3)", "0", "PK != SAR"),
        ("ADME auxiliary", "t1/2 (F12/F13), Vd (F14), DailyMed (F11)", "0", "Context mismatch"),
        ("Data augment.", "DrugBank (F1), ChEMBL conserv. (F10)", "-0.11", "Quality > quantity"),
        ("Loss/calibr.", "Asymmetric (F4), isotonic (F5)", "-0.09", "Data insufficient"),
        ("Adaptive UQ", "Difficulty model (F15), Tanimoto (F7)", "N/A", "Not predictable"),
        ("Bioavailability", "F feature (F8)", "-0.05", "Weak classifier"),
        ("Data expansion", "ChEMBL v2 strict (S12)", "-0.04*", "Only lever that worked"),
    ]
    print(f"  {'Dimension':<18} {'Approaches':<42} {'Delta':>6} {'Conclusion'}")
    print(f"  {'-'*85}")
    for dim, approaches, delta, conclusion in table:
        print(f"  {dim:<18} {approaches:<42} {delta:>6} {conclusion}")


def main():
    print("Generating paper figures and tables...", flush=True)
    s13, diag, ho = load_data()

    print("\nFigures:", flush=True)
    fig2_scatter(s13)
    fig3_plm_vs_sisyphus(s13, diag)
    fig4_forest(s13)
    fig5_conditional_coverage(s13)
    fig6_error_decomposition(diag)

    print("\nTables:", flush=True)
    table6_drug_class(diag)
    table7_negative_results()

    print(f"\nAll outputs in {OUTDIR}/")


if __name__ == "__main__":
    main()
