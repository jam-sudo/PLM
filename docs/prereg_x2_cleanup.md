# X2 Pre-registration — Data Cleanup + Visual Expansion

**Date**: 2026-04-10
**Status**: PRE-REGISTERED (before running)

## Hypothesis

**H1**: Removing 18 high-confidence contaminated entries from v0.5 will improve HO AAFE (vs S11 baseline 3.372 ± 0.010).

**H2**: Additionally adding 10 visually-validated new profiles (26% yield from batches 2-5 of auto_digitized_full queue) will further improve or maintain HO AAFE relative to H1 result.

## Contamination identification evidence (batches 2-5, indices 48-85)

Visual inspection of 38 figure PNGs revealed systematic auto-digitization errors in v0.5 where the source figure was NOT a single-dose oral parent-drug PK profile. Flagged categories:

1. **Wrong analyte (DDI victims/metabolites)**: netupitant→digoxin, nitroglycerin→1,2-GDN, rolapitant→DEX
2. **Wrong matrix (PD response curves)**: motixafortide→CD34+ cells
3. **Wrong route/dose (>100x magnitude errors)**: oxymetazoline 18mg oral (real 0.05-0.2mg nasal), nitroglycerin 6.5mg oral (real 0.4-0.8mg SL)
4. **Implausible Cmax (>5x off literature)**: nirmatrelvir 100mg Cmax 2.0 (real ~1000), rucaparib 600mg Cmax 4.3 (real ~1900), sarecycline 100mg Cmax 6912 (real ~1000)
5. **Multi-dose steady-state stored as single-dose**: naloxone (oral bioavailability ~2% would give Cmax <1)

Total high-confidence removals: 18 entries across 9 drugs.

## Success criteria

**Primary metric**: ΔHO AAFE = AAFE(v0.6 or v0.7) − AAFE(v0.5 baseline 3.372)

Pre-registered verdict bands (applied to both H1 and H2):
- **PASS**: ΔHO ≤ −0.10 (improvement; note: AAFE is error-like so lower=better, ΔHO is S11 − new)
  - Restatement: new_AAFE ≤ 3.272
- **PARTIAL**: −0.10 < ΔHO ≤ −0.05 (3.272 < new_AAFE ≤ 3.322)
- **NULL**: −0.05 < ΔHO < +0.05 (3.322 < new_AAFE < 3.422)
- **HARM**: ΔHO ≥ +0.05 (new_AAFE ≥ 3.422)

## Evaluation protocol

- Config: fp_enc_base (Morgan FP4096 + physchem + μPBPK + ADME encoder) — same as S11
- Training data: v0.6 (for H1), v0.7 (for H2)
- CV: 5-fold GroupKFold on IK14, 3 seeds (42, 123, 456) — same as S11
- Holdout: 97 drugs (unchanged, uncontaminated)
- Seed averaging, paired across baseline and new

## Null results are valid

If cleanup produces NULL (no HO change), the interpretation is:
- Contaminated entries were tolerated by XGB regularization (gradient boosting robust to label noise)
- OR contamination magnitude was too small to affect 199→181 row training
- OR HO performance is bottlenecked by distribution shift, not training noise
- This is still a **data quality improvement** and should be kept for downstream work

If cleanup produces HARM (unexpected):
- Investigate whether the "contaminated" entries were actually correct for a different reason (e.g., extracted from a different figure on same PDF)
- Re-audit removal list
- Retract v0.6

## Holdout protection

Verified 2026-04-10: all 18 flagged removals and 10 new additions are NOT in the 97-drug holdout. No contamination of evaluation set.

## Pre-registered on: 2026-04-10 (before v0.6 build)
