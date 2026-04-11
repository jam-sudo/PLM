# S12 Pre-registration — v12 retrain (v11 + ChEMBL v2 strict)

**Date**: 2026-04-11 (before running)
**Status**: PRE-REGISTERED

## Context

F10/I4 (2026-04-07) previously tried adding ChEMBL Cmax data and failed due to three contamination modes:
1. Animal data contamination (rat/mouse/dog PK mixed with human)
2. `mg/kg` dose parsing error (regex captures "10 mg" from "10 mg/kg")
3. Persistent log_cd shift from species/unit confusion

S12 attempts a **re-extraction with stricter filters** to address each mode. The new pipeline (`pipeline/chembl_v2_strict.py`) enforces:

- **Animal rejection** (any description containing rat/mice/mouse/dog/canine/monkey/rabbit/rodent/murine/beagle/primate/porcine/ovine/bovine/sprague-dawley/wistar/c57/balb/cd1/irc/icr/cynomolgus/rhesus/macaque)
- **Positive human requirement** (description must mention human/healthy/patient/clinical/homo sapiens OR assay_organism="Homo sapiens")
- **Oral-only** (description must mention po/oral/orally/tablet/capsule/suspension)
- **Non-oral rejection** (iv/intravenous/infusion/im/sc/topical/nasal/inhalation/sublingual/buccal/etc.)
- **mg/kg rejection** (explicit regex for "N mg/kg" patterns)
- **Unit whitelist** (ng/mL, µg/mL, mg/L, µg/L, or nM/µM with MW conversion)
- **log_cd sanity** within v11 p1-p99 ± 0.5 buffer
- **Per-(drug, dose) grouping** — no aggregation across different doses of the same drug

## Extraction results (before retrain)

- Queried 25,002 ChEMBL Cmax activities
- Rejected: 22,443 animal, 592 no-human marker, 545 missing fields, 533 no-oral marker, 268 already in v11/holdout, 166 non-oral route, 106 mg/kg, 32 no dose pattern, 7 log_cd out of range
- **Accepted: 290 rows → 164 unique (drug, dose) pairs → 91 new unique drugs**
- v12 = v11 + 164 new rows = **4704 rows (+3.6%), 1264 drugs (+7.8%)**

## Hypothesis

**H1**: v12 (v11 + 164 strictly-filtered ChEMBL rows) improves HO AAFE vs v11 baseline.

**H1 rationale**: The new rows add 91 novel drugs (+7.8% drug diversity) with human oral Cmax data that survived animal/mg-kg/route/unit filtering. If the strict filters successfully excluded noise, HO should improve modestly. If residual contamination exists, HO may stay null or worsen.

## Baseline

S11 fp_enc_base: **HO AAFE 3.372 ± 0.010** (3 seeds, 5-fold GroupKFold by IK14)

## Success criteria

Applied to **ΔHO = mean(HO_v12) − mean(HO_v11)** (paired across 3 seeds):

| Verdict | Range | Interpretation |
|---|---|---|
| **PASS** | ΔHO ≤ −0.05 | v12 improves HO (new_HO ≤ 3.322) |
| **PARTIAL** | −0.05 < ΔHO ≤ −0.02 | Modest improvement |
| **NULL** | −0.02 < ΔHO < +0.05 | No detectable change |
| **HARM** | ΔHO ≥ +0.05 | v12 worsens HO (new_HO ≥ 3.422) |

## Design

- **Model**: fp_enc_base (Morgan FP4096 + frozen ADME encoder 128 + physchem 20 + TDC 9 + μPBPK 6 + log_dose) — same as S11
- **CV**: 5-fold GroupKFold on IK14
- **Seeds**: 42, 137, 2024
- **Comparison**: same 3 seeds run on both v11 and v12, paired (A=v11 baseline, B=v12)
- **Evaluation**: 97-drug holdout, AAFE metric

## Holdout protection

Verified 2026-04-11 in `pipeline/build_v12_chembl.py`: all 164 ChEMBL rows filtered against the 97 holdout IK14s before merging. Additionally, `chembl_v2_strict.py` pre-filtered against v11+holdout during extraction.

## Null is still informative

- If PASS: validates that strict re-extraction with species/route/dose filters unlocks ChEMBL for PLM training. Data quantity expansion via ChEMBL becomes a viable path.
- If NULL: confirms that 164 rows / 4540 = 3.6% is below signal threshold for AAFE change. Expansion needs ~10x more rows to move the needle.
- If HARM: the strict filter was not strict enough; residual contamination remains. Need further filter refinement or abandon ChEMBL path.

## Pre-registered before: `models/s12_v12_retrain.py` execution on 2026-04-11
