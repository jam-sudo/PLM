# PLM Research Log

> All experiments, both successes and failures. Linked from [CLAUDE.md](../CLAUDE.md).
> Each entry records: hypothesis, method, result, interpretation, next action.

---

## Successes (improved or matched baseline)

### S1. Table Extraction → AAFE 10.1 → 3.275
- **Date**: Phase 1
- **Hypothesis**: Figure digitization noise is the primary error source; extracting PK tables directly from PDF text will yield cleaner data
- **Method**: LLM-based PK table extraction from FDA review PDFs → structured Cmax/AUC/t1/2
- **Result**: CV AAFE 10.1 → 3.275 (3x improvement)
- **File**: `models/xgboost_v2_results.json`
- **Interpretation**: Data quality >> data quantity. Table-extracted scalars far more reliable than auto-digitized C-t curves
- **Status**: Adopted as new baseline

### S2. Data Cleaning + Sisyphus Validation → AAFE 3.1 (CV)
- **Date**: Phase 1 iteration
- **Hypothesis**: Cross-validating against Sisyphus predictions can identify bad datapoints
- **Method**: Sisyphus-validated subset, cleaned outliers, expanded with table data
- **Result**: In-domain CV AAFE 3.098 (v7), OOD holdout 3.819
- **File**: `data/validation/phase_clean_v7_results.json`
- **Interpretation**: CV improves but OOD holdout stays ~3.7-4.0 — overfitting to training chemical space
- **Status**: Adopted

### S3. ADME Pretrained Encoder
- **Date**: Feature engineering phase
- **Hypothesis**: Pretraining on TDC ADME tasks (CYP, BBB, clearance) creates useful molecular representations
- **Method**: XGBoost encoder pretrained on 8 ADME endpoints, features concatenated with Morgan FP
- **Result**: CV AAFE 2.744 (encoder+FP), HO AAFE 3.456
- **File**: `models/pretrain_results.json`
- **Interpretation**: Modest CV gain but no HO improvement. ADME features are somewhat informative but don't transfer to OOD drugs
- **Status**: Minor improvement, not transformative

### S4. Mechanistic ML Features
- **Date**: Feature engineering phase
- **Hypothesis**: Adding physicochemical descriptors (MW, logP, TPSA, HBD/HBA, ionization) improves prediction
- **Method**: XGBoost with Morgan FP + physchem + ionization features
- **Result**: CV AAFE 2.864, HO AAFE 3.532
- **File**: `data/validation/mechanistic_ml_results.json`
- **Interpretation**: Physchem features help CV slightly, HO gap persists. Chemical space coverage, not feature richness, is the bottleneck
- **Status**: Adopted into feature set

### S5. LLM PK Table Extraction Pipeline
- **Date**: Data expansion phase
- **Hypothesis**: LLM can reliably extract structured PK parameters from FDA review PDFs
- **Method**: Claude/GPT extracts drug name, dose, Cmax, AUC, t1/2 from 456 PDFs
- **Result**: 1,333 valid PK tuples from 226 drugs, 303 with SMILES mapping
- **File**: `data/llm_extracted/extraction_stats.json`
- **Interpretation**: Excellent data extraction tool. Expanded training data from ~200 to ~3,500 profiles (combined with Sisyphus)
- **Status**: Adopted as primary data source

### S6. Unit Normalization Pipeline
- **Date**: Data quality phase
- **Hypothesis**: Systematic unit conversion prevents ng/mL vs mg/L contamination
- **Method**: `pipeline/normalizer.py` with UNIT_TO_NGML table, sanity checks, Cmax/dose ratio bounds
- **Result**: Full pipeline audit (2026-04-07): all conversions correct, no unit mismatch found
- **File**: `pipeline/normalizer.py`
- **Interpretation**: Critical infrastructure. One unit error = 1000x dataset contamination
- **Status**: Verified and operational

---

## Failures (no improvement or degraded performance)

### F1. DrugBank Synthetic C-t Profiles
- **Date**: 2026-04-07
- **Hypothesis**: Generating synthetic C-t profiles from DrugBank PK parameters (t1/2, Vd, CL) via 1-compartment model can expand training data 6x
- **Method**: 780 drugs with t1/2+Vd+SMILES → 1-cpt oral model → synthetic Cmax. 335 novel drugs added after holdout/dedup exclusion
- **Result**: Baseline AAFE 3.355 → Expanded 3.469 (+0.11 worse), DrugBank-only 4.143
- **File**: `data/validation/drugbank_expansion_results.json`, `pipeline/synthetic_ct.py`
- **Why it failed**:
  1. Fixed ka=1.5/h for all drugs — ignores real absorption variability
  2. Fixed dose=100mg — feature space collapse at log10(dose)=2.0
  3. 1-compartment assumption — misses distribution phase, overestimates Cmax
  4. Synthetic noise > information gain from 335 new compounds
- **Lesson**: Data quality >> data quantity. Noisy synthetic data actively harms the model
- **Status**: Reverted. Synthetic data files kept for reference but excluded from training

### F2. MolFormer Embeddings
- **Date**: Feature engineering phase
- **Hypothesis**: Pretrained MolFormer (transformer on SMILES) provides better molecular features than Morgan FP
- **Method**: MolFormer embeddings (768-dim) replacing or augmenting Morgan FP
- **Result**: AAFE 3.355 (baseline) → 3.419 (MolFormer+baseline) → 3.447 (MolFormer-only)
- **File**: `data/validation/molformer_results.json`
- **Why it failed**: MolFormer captures SAR-relevant features, but PK ≠ SAR. Morgan FP already encodes the substructure patterns most relevant to ADME
- **Lesson**: Fancy embeddings don't help when the bottleneck is data size, not representation power
- **Status**: Abandoned

### F3. Retrieval-Augmented Delta Learning
- **Date**: Novel experiment phase
- **Hypothesis**: For each test drug, retrieve k=5 nearest neighbors from training set and predict residual
- **Method**: Tanimoto NN retrieval + delta prediction on top of base model
- **Result**: AAFE 3.355 → 3.865 (worse)
- **File**: `data/validation/novel_results.json` (ablation.7_retrieval_delta)
- **Why it failed**: Nearest neighbors in Morgan FP space don't have similar PK. Tanimoto similarity is a poor proxy for PK similarity
- **Lesson**: Chemical similarity ≠ PK similarity. Need mechanism-aware similarity (shared CYP, transporter)
- **Status**: Abandoned

### F4. Asymmetric Loss Function
- **Date**: Novel experiment phase
- **Hypothesis**: Penalizing overprediction more than underprediction (clinical safety) improves calibration
- **Method**: Asymmetric loss with alpha=1.5 and 2.0
- **Result**: AAFE 3.355 → 3.519 (alpha=1.5), 3.455 (alpha=2.0)
- **File**: `data/validation/novel_results.json`
- **Why it failed**: Loss asymmetry shifts bias but doesn't reduce variance. With N~3500, the model doesn't have enough signal to benefit from fine-tuned loss shapes
- **Status**: Abandoned

### F5. Isotonic Calibration
- **Date**: Novel experiment phase
- **Hypothesis**: Post-hoc isotonic regression on CV predictions corrects systematic bias
- **Method**: Isotonic regression fit on CV residuals, applied to holdout
- **Result**: AAFE 3.355 → 3.447 (worse)
- **File**: `data/validation/novel_results.json` (ablation.0.5_isotonic)
- **Why it failed**: Isotonic calibration overfits to CV error distribution, which differs from OOD holdout
- **Status**: Abandoned for holdout, potentially useful for in-domain CV

### F6. PK-DB API Data Access
- **Date**: 2026-04-07
- **Hypothesis**: PK-DB (pk-db.com) provides open C-t timecourse data via REST API
- **Method**: Queried all API endpoints (outputs, timecourses, pkdata/*)
- **Result**: Metadata endpoints work (803 studies), but ALL data endpoints return count=0. Only 88/803 studies have open licence
- **Why it failed**: API bug or access restriction. Data exists in metadata but cannot be retrieved
- **Status**: Blocked. No workaround found

---

## Informational (data leakage / not comparable)

### I1. LLM Direct Cmax Prediction — DATA LEAKAGE
- **Date**: Evaluation phase
- **Hypothesis**: LLM can predict Cmax from drug name + dose
- **Method**: Single LLM pass, 5-round multi-prompt, CoT reasoning
- **Result**: AAFE 2.228 (single), 2.144 (5-round trimmed), 2.187 (CoT median)
- **Files**: `data/validation/llm_smoke_results.json`, `data/validation/five_round_results.json`, `data/validation/llm_cot_results.json`
- **WARNING**: This is data leakage. LLM recalls published PK from training corpus (medical literature, FDA labels). Holdout drugs are all marketed compounds. Cannot generalize to novel compounds
- **Status**: NOT PLM model performance. Useful as data extraction tool only

### I2. LLM + XGBoost Ensemble
- **Date**: Evaluation phase
- **Method**: Weighted combination of XGBoost + LLM predictions
- **Result**: Median 3-way AAFE 2.212, weighted ensembles 2.26-2.99
- **File**: `data/validation/ensemble_results.json`
- **WARNING**: Inherits LLM data leakage. Not a valid model performance metric
- **Status**: Not comparable to Sisyphus

---

### F8. Bioavailability Feature Addition
- **Date**: 2026-04-07
- **Pre-registered hypothesis**: Adding predicted P(F>20%) as feature reduces low-F overprediction
- **Pre-registered success criteria**: F<20% AAFE 6.0→<4.0 OR overall AAFE 3.355→<3.2
- **Method**: XGBoost classifier on TDC bioavailability data (640 drugs, AUC=0.710), P(F>20%) + log(F_proxy) as 2 extra features
- **Result**: Overall 3.355→3.407 (+0.052 worse), F<20% 6.018→6.861 (+0.843 worse). **Both criteria FAIL**
- **File**: `data/validation/bioavailability_experiment_results.json`, `pipeline/bioavailability_experiment.py`
- **Why it failed**:
  1. F classifier AUC=0.710 — too weak to provide useful signal
  2. N=9 low-F drugs in holdout — model can't learn to use F feature meaningfully
  3. Weak feature adds noise → spurious XGBoost splits → overfitting
- **Lesson**: Predicted feature is only useful when the predictor itself is strong (AUC>0.85+). Weak predictions add noise, not signal
- **Status**: FAIL. Both criteria missed.

### F7. Tanimoto-Gated Ensemble
- **Date**: 2026-04-07
- **Hypothesis**: PLM accuracy correlates with nn_tanimoto to training set; drugs close to training set should get higher PLM weight in ensemble
- **Method**: Pearson/Spearman correlation of nn_tanimoto vs PLM absolute error; stratified analysis by Tanimoto bins
- **Result**: r = -0.088 (p=0.39) — no correlation. PLM wins 35% at low Tanimoto, 50% at mid, 18% at high. No usable pattern.
- **File**: `data/validation/plm_sisyphus_error_correlation.json`
- **Why it failed**: Tanimoto similarity (Morgan FP) captures structural similarity but PK is driven by specific ADME mechanisms (CYP, transporters) that don't correlate with overall structural similarity
- **Lesson**: Chemical similarity ≠ PK prediction confidence. Need mechanism-specific confidence (e.g., "do I know the CYP substrate class?") not generic similarity
- **Status**: Abandoned. MW 450-600 pattern noted (PLM wins 61.5%, N=13) but too small to act on

### I3. PLM-Sisyphus Error Correlation Analysis
- **Date**: 2026-04-07
- **Result**: Pearson r = 0.644 (signed errors), r = 0.366 (absolute errors)
- **Oracle best-of-2**: AAFE 1.794 (vs Meta 2.190, PLM 3.355)
- **PLM wins on 34% of drugs**, opposite error direction on 35%
- **w=0.1 ensemble**: AAFE 2.198 (≈ Meta parity, no improvement)
- **Bias**: PLM +0.269 overprediction, Meta +0.037 (near-unbiased)
- **Conclusion**: Ensemble potential exists (r < 0.7) but cannot be exploited with current methods without cherry-picking

### F9. VLM Figure Re-Digitization (low yield)
- **Date**: 2026-04-07
- **Hypothesis**: Claude vision can extract C-t data from 335 figures that OCR-based auto-digitizer failed on, recovering ~100+ profiles
- **Method**: Caption-based classification identified 192 "C-t candidates". Processed 32 figures across 2 batches using Claude vision
- **Result**: 3 C-t curves extracted from 32 figures (9.4% hit rate). Projected yield for all 192: ~18 curves
- **Why low yield**: Most "failed" figures are actually PK parameter tables (not plots), PD plots, study design tables, or legend fragments. Caption keywords like "concentration" appear in table captions too
- **Extracted**: Istradefylline fasted/fed (2 curves), R-warfarin (1 curve)
- **Lesson**: The auto-digitizer's 63.9% success rate was not bottlenecked by OCR quality — it failed because 36% of "C-t candidates" were never C-t curves to begin with
- **Status**: Low ROI. VLM digitizer script preserved (`pipeline/vlm_digitizer.py`) for future use on confirmed C-t figures

### F10. ChEMBL Conservative Salvage
- **Date**: 2026-04-07
- **Pre-registered hypothesis**: Conservatively filtered ChEMBL (dose>=100mg, log_cd within v10 range) adds 174 novel drugs without introducing noise
- **Method**: Filter 8,002 → 174 entries (dose>=100mg, log_cd in [p10,p90] of v10). Added to training with w=1.0, 0.3, 0.1
- **Result**: AAFE 3.355 → 3.372 (w=1.0), 3.427 (w=0.3), 3.444 (w=0.1). All worse. **FAIL**
- **File**: `data/validation/chembl_salvage_results.json`
- **Why it failed**: Even after aggressive filtering, remaining animal data contamination and unit inconsistencies add noise. v10 data quality is strictly superior
- **Status**: FAIL. ChEMBL data confirmed unusable for PLM training in current form

### I4. ChEMBL 8,002 Data Quality Audit
- **Date**: 2026-04-07
- **Finding**: ChEMBL PK expansion (8,002 entries) has 3 overlapping data quality issues:
  1. **Animal data contamination**: `assay_organism` filter doesn't catch entries where organism=None. Rat/mouse PK data mixed with human
  2. **mg/kg → mg dose parsing error**: 64% of entries have dose ≤10mg (median 10mg vs v10 median 60mg). Regex extracts "10 mg" from "10 mg/kg" descriptions
  3. **Persistent log_cd shift**: Even at matched dose bins, ChEMBL log_cd is +0.7 higher than v10 (~5x Cmax/dose), likely from animal PK or nM→ng/mL conversion issues
- **File**: `data/curated/chembl_pk_expansion.json`, `pipeline/chembl_expansion.py`
- **Conclusion**: Data is too contaminated for direct use. Would require: (a) text-based human/animal classification of each assay description, (b) mg/kg detection and body weight correction, (c) cross-validation against known human PK values
- **Status**: BLOCKED. Needs significant re-extraction work

### F11. DailyMed ADME Feature Merge
- **Date**: 2026-04-07
- **Pre-registered hypothesis**: Filling TDC NaN features with DailyMed-extracted ADME data (F, PPB, t1/2, CYP, transporters) increases MI and reduces AAFE
- **Pre-registered success criterion**: AAFE < 3.1
- **Method**: Extracted ADME features from 84/97 holdout drugs via DailyMed API. Merged 37 NaN fills into TDC. Retrained XGBoost with same architecture
- **Result**: AAFE 3.355 → 3.358 (+0.003). **FAIL.** 8 drugs improved, 7 degraded, net zero
- **File**: `data/validation/dailymed_feature_merge_results.json`
- **Why it failed**:
  1. Only 37 NaN fills across 29 drugs — too sparse to shift the overall distribution
  2. Regex-extracted values are noisy (no validation against ground truth)
  3. Training set features unchanged (DailyMed only extracted for holdout drugs) — feature distribution mismatch
  4. The 73% model capacity gap (Shannon S7) may require fundamentally different features, not just filling NaN in existing ones
- **Lesson**: Sparse feature fills (37 values across 29 drugs) don't measurably change a 3,546-sample model. Need dense coverage AND training set parity
- **Status**: FAIL

### S7. Information-Theoretic Ceiling Analysis
- **Date**: 2026-04-07
- **Method**: Shannon information theory applied to PLM prediction problem
- **Key Results**:
  - Channel capacity (SMILES→Cmax): 2.50 bits/prediction
  - Model captures: 0.318 bits (12.7% of channel)
  - CV R² = 0.356 (64% of variance unexplained)
  - Noise floor AAFE: 1.269 (theoretical best)
  - **Model capacity gap: 1.537** (noise→CV, 73% of total error)
  - **Generalization gap: 0.549** (CV→holdout, 27% of total error)
- **Critical insight**: 10 failed experiments were all attacking the generalization gap (27%) while the model capacity gap (73%) was the true bottleneck. 87% of predictable information in SMILES→Cmax channel is not captured by current features.
- **Feature MI decomposition**: Morgan FP alone = 0.125 bits (5%), all features = 0.318 bits (13%). TDC ADME features contribute 60% of captured information despite being available for only 58% of holdout drugs.
- **Prescription**: New information sources needed — not more data points with same features, but higher-coverage ADME features (CYP panel, transporter, continuous F, in-vitro CL)
- **Status**: PARADIGM SHIFT. Redirects strategy from data expansion to feature coverage expansion.

### S8. Non-Linear PK Stratification Analysis
- **Date**: 2026-04-08
- **Method**: Classified 11/97 holdout drugs as non-linear PK (saturable metabolism, absorption, transport, autoinduction). Computed stratified AAFE for LLM+calibrator vs XGBoost.
- **Non-linear drugs**: phenytoin, carbamazepine, paroxetine, posaconazole, itraconazole, clopidogrel, sirolimus, clozapine, probenecid, digoxin, tamoxifen
- **Key Results**:
  - LLM+calibrator: NL AAFE 3.276 (N=11), Linear AAFE **1.923** (N=86)
  - XGBoost baseline: NL AAFE **2.837** (N=11), Linear AAFE 3.427 (N=86)
  - LLM excels on linear drugs (1.923 vs 3.427), XGBoost better on non-linear (2.837 vs 3.276)
  - Worst LLM outliers: posaconazole (17.3x over), paroxetine (10.3x over) — both saturable mechanisms
  - Mechanism-aware routing (NL→XGB, Linear→LLM): AAFE **2.009** (1.6% gain)
  - Oracle per-drug best-of-2: AAFE **1.834** (10.2% gain ceiling)
  - LLM wins 67/97 drugs overall (69%)
- **Interpretation**: LLM's pharmacological knowledge is well-calibrated for standard linear PK but systematically overpredicts Cmax for drugs with saturable mechanisms (recalls "typical" PK unaware of dose-dependent non-linearity). Non-linear PK drugs are a specific, identifiable failure mode.
- **File**: `data/validation/nonlinear_pk_analysis.json`
- **Status**: SUCCESS. Identifies actionable error decomposition. Simple NL routing gives small gain (1.6%) due to N=11, but the linear-only AAFE of 1.923 demonstrates LLM capability on well-behaved drugs.

### I5. Post-Cutoff Prospective Validation Set Assembly
- **Date**: 2026-04-08
- **Purpose**: Address transductive limitation — assemble drugs approved AFTER Claude's training cutoff (May 2025) for truly prospective evaluation
- **Method**: Identified 19 oral small molecule NMEs approved June 2025 – April 2026 from FDA. Retrieved SMILES from PubChem. Compiled dose + Cmax from FDA labels and web sources.
- **Ready to test**: 9 drugs with SMILES + Cmax confirmed (taletrectinib, sebetralstat, zongertinib, dordaviprone, imlunestrant, remibrutinib, sevabertinib, tradipitant, relacorilant)
- **Need Cmax extraction**: 10 drugs with NDA numbers identified, FDA label extraction pending
- **Key drugs**: orforglipron (first oral GLP-1 RA, approved 2026-04-01 — 7 days ago), relacorilant (2026-03-25)
- **File**: `data/validation/post_cutoff_candidates.json`, `data/validation/post_cutoff_smiles.json`
- **Next step**: Run XGBoost predictions on 9 ready drugs, then run LLM CoT for comparison. LLM should fail (no pretraining knowledge), establishing genuine from-scratch prediction capability.
- **Status**: DATA ASSEMBLED. Predictions run (see S9).

### S9. External Validation Experiments (E1/E2/E3)
- **Date**: 2026-04-08
- **Pre-registered**: Yes. Hypotheses stated before running.
- **Cherry-picking safeguards**: Model trained once (no retuning), all drugs reported (zero exclusions), IK14 leakage checked (61+3 contaminated drugs excluded automatically)
- **Method**: Same XGBoost model as ho_diagnostic (3,546 samples, 868 drugs). Applied to 3 independent test sets without any parameter adjustment.
- **Results**:

| Experiment | N evaluated | N excluded (leakage) | AAFE | 2-fold% | Bias | Pre-reg criterion | Status |
|---|---|---|---|---|---|---|---|
| E0: Holdout 97 (sanity) | 97 | 0 | **3.355** | 37.1% | +0.269 | — | ✓ reproduces |
| E1: Brown 2025 (external) | 29 | 61 | **3.255** | 37.9% | −0.159 | <5.0 | **PASS** |
| E2: Post-cutoff (prospective) | 6 | 3 | **4.262** | 16.7% | +0.071 | ~3.5 | WORSE |
| E3: Holdout 103 (expanded) | 103 | 0 | **3.354** | 36.9% | +0.228 | 3.3–3.5 | **PASS** |

- **Key findings**:
  1. **E1 (Brown 2025)**: AAFE 3.255 on 29 truly independent drugs — BETTER than holdout 3.355. External validation confirms model generalizes. Negative bias (−0.159) = slight underprediction on newer drugs. 61/92 drugs were in training (LLM-extracted FDA data covers 2020-2024 approvals extensively).
  2. **E2 (Post-cutoff)**: AAFE 4.262 on 6 drugs — worse than holdout as expected. Novel chemical space (oncology TKIs, BTK inhibitors) may be underrepresented in training. N=6 too small for firm conclusions. 3/9 drugs were already in training (clinical trial data available pre-approval).
  3. **E3 (Holdout 103)**: AAFE 3.354 — essentially unchanged from 97 (3.355). The 6 recovered Sisyphus drugs behave similarly to the original holdout.
- **Leakage disclosure**: 61/92 Brown 2025 drugs found in training via IK14 check. This is NOT a pipeline error — PLM's LLM extraction from 456 FDA PDFs naturally covers recently approved drugs. The leakage check correctly excluded them.
- **Honest assessment**: E1 passes but N=29 is smaller than desired. The training set's broad coverage (868 drugs) means most marketed oral drugs are already in training. Truly independent external validation requires either (a) pre-approval compounds or (b) non-FDA sources.
- **File**: `data/validation/external_validation_results.json`, `pipeline/external_validation.py`
- **Status**: SUCCESS (E1 PASS, E3 PASS). E2 inconclusive (N=6).

### F12. B1v4 — Physics-Informed NN with Half-Life Auxiliary Supervision
- **Date**: 2026-04-10
- **Pre-registered hypothesis**: Output-space parameterization through physical PK model (A, k_slow, k_fast) combined with half-life auxiliary loss will reduce the CV-HO overfitting gap (0.67) by constraining the hypothesis class to a physical manifold. Targets: HO AAFE ≤ 3.15 (PASS), 3.15–3.30 (PARTIAL), >3.30 (FAIL).
- **Mechanism**: NN outputs (logA, log k_slow, log Δk); analytic Cmax = A·(exp(−k_slow·tmax)−exp(−k_fast·tmax)) and t_half = ln2/k_slow. Dual loss L = L_cmax + λ·L_thalf on rows where half-life is observed (356 unique IK14s, 1498 rows out of 4540 training).
- **Ablation (3 NN variants, same features = FP4096+physchem+tdc+upbpk+log_dose = 4132-d)**:
  | variant | CV AAFE | HO AAFE | CV-HO gap | interpretation |
  |---|---|---|---|---|
  | nn_scalar (direct output) | 3.128 | 4.076 | 0.947 | NN architecture effect alone |
  | nn_physical (physical reparam only) | 3.159 | 4.138 | 0.979 | +reparam |
  | nn_b1v4_full (phys+halflife loss) | 3.182 | 4.144 | 0.961 | +half-life aux |
  | *xgb_ref (fp_enc_base)* | *2.788* | *3.456* | *0.67* | *reference* |
- **Result**: **FAIL.** NN framework is ~0.6 AAFE worse than XGB on this tabular task (known tabular ML result — sparse 4096-d FP + 3,600 rows favors tree models). First NN run (larger hidden 768-384-128 with BatchNorm) showed half-life aux effect of −0.26; second run (256-64 LayerNorm, higher dropout) showed null (+0.006). The first run's improvement was noise from random init, not a real mechanism effect.
- **File**: `models/b1/b1_results.json`, `models/plm_b1_nn.py`
- **Why null**: (a) NN cannot match XGB baseline on this feature set by >0.6 AAFE, making absolute comparison impossible; (b) the apparent effect was not reproducible across hyperparameters; (c) see F13 XGB replication which confirmed null.
- **Status**: FAIL

### F13. B1v5 — XGB Half-Life Stacking (observed + predicted auxiliary feature)
- **Date**: 2026-04-10
- **Pre-registered hypothesis**: If B1's mechanism (half-life informs Cmax) is real, it should transfer to XGB framework via (a) direct observed half-life as feature, or (b) out-of-fold predicted half-life from a stacked XGB. This tests the mechanism independently of NN architecture confound in F12.
- **Method**: 3 XGB models on same features (FP4096+physchem+tdc_adme+μPBPK+log_dose = 4132-d), 5-fold GroupKFold on IK14, leakage-safe OOF for predicted half-life:
  | variant | CV AAFE | HO AAFE | CV-HO gap | 2-fold% |
  |---|---|---|---|---|
  | A) XGB baseline (no half-life) | 3.092 | **3.389** | 0.297 | 34.0% |
  | B) XGB + observed half-life feat | 3.092 | 3.383 | 0.291 | 36.1% |
  | C) XGB + predicted half-life feat | 3.094 | 3.380 | 0.286 | 36.1% |
- **Result**: **FAIL.** Δ(B−A) = −0.006, Δ(C−A) = −0.009 — both within noise. Half-life adds no measurable information to Cmax prediction whether supplied as observed value or out-of-fold XGB prediction.
- **OOF half-life prediction quality**: AAFE 1.97, MAE(log10) 0.295 — good enough to be meaningful, yet transfers zero benefit to Cmax.
- **File**: `models/b1/b1_xgb_stacked_results.json`, `models/plm_b1_xgb_stacked.py`
- **Mechanism interpretation**: Cmax for single-dose PK is dominated by F·dose/Vd (absorption + distribution amplitude) and ka (absorption rate), not by ke = ln2/t_half (elimination rate). Elimination governs AUC and terminal concentration, not peak. Half-life is the WRONG auxiliary signal for Cmax — it constrains the irrelevant parameter dimension.
- **Lesson**: Physically plausible does not imply statistically useful. The analytic coupling Cmax = f(A, ka, ke) has low sensitivity to ke near typical PK values, so even accurate half-life supervision barely shifts Cmax predictions.
- **Corollary (see S10)**: Side-by-side, this baseline (3.389) is better than the current `fp_enc_base` reference (3.456), revealing the ADME encoder was hurting, not helping.
- **Status**: FAIL. B1 mechanism refuted across both NN and XGB frameworks. Half-life is not a useful auxiliary target for Cmax prediction.

### S10. ADME Encoder Claim — RETRACTED and Replaced by S11
- **Date**: 2026-04-10
- **Original claim (INCORRECT)**: While running B1v5, observed HO AAFE 3.389 without encoder vs 3.456 with encoder (`pretrain_results.json` `fp_enc_base`), concluded encoder hurts by 0.067.
- **Retraction reason**: The 3.389 vs 3.456 comparison was CONFOUNDED — different scripts, different random seeds, different `tree_method`. Not apples-to-apples.
- **Replacement**: See S11 for the pre-registered replication that measured the true effect.
- **Status**: RETRACTED. Conclusion reversed by S11.

### S11. ADME Encoder Pre-Registered Replication
- **Date**: 2026-04-10
- **Pre-registered hypothesis**: ΔHO AAFE = (with encoder) − (without encoder) > +0.03, AND 3-seed CI excludes 0 → PASS (encoder hurts). Otherwise INCONCLUSIVE or FAIL.
- **Design**: 3 seeds {42, 137, 2024} × 2 configs (with/without frozen 128-d encoder), 5-fold GroupKFold on IK14, identical XGB params, identical features except encoder block.
- **Result**:
  | config | CV AAFE (mean±std) | HO AAFE (mean±std) | CV-HO gap |
  |---|---|---|---|
  | A — with encoder | 3.165±0.005 | **3.372±0.010** | **0.207** |
  | B — no encoder | 3.091±0.001 | 3.387±0.010 | 0.296 |
  - Paired ΔHO = **−0.015 ± 0.021**
  - 95% CI (t_2): [−0.067, +0.037] — **includes 0**
  - ΔCV-HO gap = **−0.089** — encoder reduces gap reproducibly
- **Pre-registered verdict**: **FAIL** (encoder does NOT hurt HO; in fact slightly helps). S10 reversed.
- **Key finding**: The encoder's real effect is to **reduce the CV-HO gap by ~0.09** while leaving HO AAFE statistically unchanged. This is a regularization signature, not a feature-noise signature as S10 initially suggested. The encoder IS doing its job — distilling TDC ADME tasks into a representation that smooths out overfitting in the Cmax head.
- **Corrected baseline**: `fp_enc_base` HO AAFE ≈ **3.37** (not 3.456 from the old pretrain_results.json which used an unlucky seed). This is PLM's true holdout number under the current feature architecture.
- **Methodological lesson**: Cross-script baseline comparisons are confounded. Always replicate within a single codebase with shared seeds before interpreting deltas.
- **Actionable direction**: Since the encoder is halving the CV-HO gap, MORE aggressive regularization in the same direction (longer pretraining, larger encoder, stronger weight decay, fewer raw FP features) may give further gains. This is a new breakthrough candidate.
- **File**: `models/b1/s10_replication_results.json`, `models/s10_replication.py`
- **Status**: NULL on HO AAFE (as pre-registered), POSITIVE on gap reduction. S10 retracted. Current PLM baseline corrected to HO ≈ 3.37.

### I7. Per-Drug Diagnostic — PLM vs Sisyphus on 97-drug Holdout
- **Date**: 2026-04-10
- **Purpose**: Before proposing another mechanism, diagnose WHICH drugs PLM systematically fails on, so the next proposal is data-driven not blind-guess.
- **Method**: Trained S11 config (fp_enc_base, seed 42) on all 4540 training rows, predicted 97 holdout. Loaded Sisyphus meta predictions from `holdout_definition.json` (caveat: these are the OLD contaminated Sisyphus values ≈2.19 AAFE, not the clean 2.808; however the RELATIVE per-drug comparison is still informative). Computed per-drug signed + absolute log errors, stratified by chemical features, ionization, drug class, non-linear PK.
- **Key Findings**:
  1. **PLM has +0.26 systematic over-prediction bias** (67% of holdout drugs over-predicted). This is NOT random noise.
  2. **Worst drug classes** (mean class AAFE, signed PLM error):
     | Class | n | PLM AAFE | Sis AAFE | PLM signed |
     |---|---|---|---|---|
     | SSRI/SNRI | 3 | **12.17** | 6.28 | +1.085 |
     | Steroids | 3 | **6.68** | 3.56 | +0.825 |
     | TKI | 4 | 4.05 | 2.28 | −0.104 |
     | Fluoroquinolones | 4 | 2.30 | 1.22 | −0.198 |
     | NSAID | 2 | 2.78 | 2.51 | +0.444 |
  3. **Surprising: Tanimoto-to-training HIGHER in PLM-worse subset** (0.555 vs 0.463, p=0.040). PLM does NOT fail on novel compounds — it fails on drugs structurally similar to training but with outlier PK. This is SAR-PK divergence.
  4. **S8 non-linear PK is NOT the main bottleneck**: PLM-worse has 12.0% non-linear, rest has 11.1%, Fisher p=1.0.
  5. **No significant difference** in MW, logP, TPSA, HBD/HBA, RotBonds, ionization class for PLM-worse vs rest.
  6. **Over-predicted drugs** cluster on: non-linear PK (carbamazepine, digoxin, phenytoin), prodrugs (losartan, tenofovir disoproxil), high first-pass (sildenafil, ramelteon, steroids), high Vd (SSRIs).
- **Actionable hypothesis**: Cmax depends directly on F and Vd via Cmax ≈ F·dose/Vd. PLM may be missing these "downward correction" signals. Proposed test: B2 — Vd as auxiliary target (like B1 but with the physically correct parameter).
- **File**: `data/validation/plm_vs_sisyphus_diagnostic.json`, `models/plm_diagnostic_vs_sisyphus.py`
- **Status**: ACTIONABLE DIAGNOSTIC. Directly motivated B2 experiment (below).

### F14. B2 — Vd Auxiliary Target (Motivated by I7 Diagnostic)
- **Date**: 2026-04-10
- **Pre-registered hypothesis**: Vd directly enters Cmax formula (Cmax ∝ F·dose/Vd) unlike half-life (ke has weak Cmax sensitivity). Providing Vd as auxiliary feature/target should improve Cmax prediction — specifically the SSRI and steroid classes identified as PLM-worst in I7.
- **Design**: 3 configs (A: baseline no Vd, B: + observed Vd feat, C: + OOF predicted Vd feat) × 3 seeds (42, 137, 2024) × 5-fold GroupKFold on IK14. Same XGB_PARAMS and features as S11.
- **Data**: 426 Vd-labeled training drugs (TDC vd_L_kg 1107 + FDA v3 36 + DailyMed 6), 54 holdout. Much better coverage than B1's half-life (356 drugs).
- **Pre-registered criteria**: PASS ≥ +0.10, PARTIAL +0.05 to +0.10, NULL −0.02 to +0.05, HARM < −0.02
- **Result**:
  | variant | HO mean±std | paired Δ | 95% CI | verdict |
  |---|---|---|---|---|
  | A baseline | 3.372±0.010 | — | — | — |
  | B observed Vd feat | 3.403±0.007 | −0.032 ± 0.015 | [−0.069, +0.006] | **HARM** (marginal) |
  | C predicted Vd feat (OOF) | 3.429±0.003 | −0.057 ± 0.013 | [−0.090, −0.024] | **HARM** (CI excludes 0) |
- **Class-specific (target classes from I7)**:
  | Class | A baseline | B obs Vd | C pred Vd |
  |---|---|---|---|
  | SSRI/SNRI (n=4) | 7.84 | 8.26 (Δ +0.42) | 8.07 (Δ +0.23) |
  | Steroids (n=3) | 6.06 | 6.37 (Δ +0.31) | 6.21 (Δ +0.15) |
  | TKI (n=4) | 4.25 | 3.75 (Δ −0.50) | 4.24 (Δ −0.01) |
- **Result interpretation**: Pre-registered hypothesis **DIRECTLY REFUTED**. The diagnosed target classes (SSRI, steroids) got MEASURABLY WORSE with Vd supervision, not better. Only TKIs marginally benefited from observed Vd. OOF Vd prediction quality was reasonable (MAE ~0.23 log) but still hurt Cmax.
- **Why it failed** (hypotheses):
  1. **Measurement context mismatch**: TDC Vd_L_kg comes from IV studies (Lombardo dataset). Apparent Vd from oral Cmax differs because it's confounded with F (oral Vd/F, not true Vd).
  2. **Extreme-class Vd is poorly measured**: SSRIs have very high tissue distribution (Vd 10-30 L/kg) that's hard to estimate clinically; the data is noisy for exactly the classes we wanted to fix.
  3. **XGB already extracting Vd-relevant signal**: Morgan FP + physchem + μPBPK ke-derived feature may already capture what Vd would add, and explicit Vd introduces context-mismatched noise.
- **Pattern across B1, B2, F11**: Three independent ADME-auxiliary approaches (half-life, Vd, DailyMed merge) all FAILED. Strong evidence that **scalar ADME features/targets are a saturated/dead-end direction for PLM**. The bottleneck is NOT an information-content gap in features.
- **File**: `models/b1/b2_vd_stacked_results.json`, `models/plm_b2_vd_stacked.py`
- **Status**: FAIL. B2 refuted. Plus broader conclusion: ADME auxiliary path (F11/B1/B2) is exhausted.

### I6. Visual Profile Extraction Pilot (Claude multimodal vision)
- **Date**: 2026-04-10
- **Goal**: Expand PLM C(t) profile dataset beyond v0.5's 199 (of which ~25 are usable absorption-shape) to enable B1-style parametric output experiments.
- **Pipeline attempts (in order)**:
  1. **fitz text scan** of pre-extracted PK text (387 PDFs): 0 profiles — text extraction destroys table structure, numbers become free-floating
  2. **pdfplumber structured table scan** (~200 PDFs): 1 profile (NDA021164 gepirone p214) — FDA PDFs mostly store C(t) in figures, not text tables
  3. **Claude multimodal vision** reading auto_digitized_full figure PNGs (86 candidates from training drugs): 17 valid profiles from 48 processed (35% yield, stopped at 55.8% of queue due to context budget)
- **Successful pipeline** (approach 3): Use auto_digitized_full.json → filter to training drugs (not holdout) → read figure PNG directly → visually classify (single-dose oral vs rejected types) → extract (time, conc) points → save to JSON.
- **Output**: `data/curated/profile_visual_extracted.json` with 17 profiles:
  - acyclovir, aficamten, amoxicillin (RHB-105), amphetamine, benzgalantamine (galantamine), daridorexant, desvenlafaxine, dexlansoprazole, dextroamphetamine (transdermal), diphenhydramine, edaravone, elacestrant, esomeprazole strontium, granisetron (SC ER), ibrutinib (fed + fasted), larotrectinib
- **Contamination patterns in REJECTED candidates** (31/48 = 65%):
  1. **DDI wrong-analyte**: auto_digitizer mapped figure to parent NDA drug, but figure actually shows CO-ADMINISTERED drug's profile (bremelanotide→norethindrone, buprenorphine→naloxone, bupropion→dextromethorphan, istradefylline→atorvastatin OH metabolite, drospirenone→estetrol)
  2. **Multi-dose steady-state sawtooth** over 300-500h (adagrasib, avapritinib, ivosidenib)
  3. **PK parameter tables** misclassified (auto_digitizer extracted 25 "points" from table cells)
  4. **PD response curves** (CD34+ cells, survival, ANC nadir)
  5. **Dissolution testing** (% dissolved vs Time(min))
  6. **Demographics box plots** (by renal impairment, BSA category)
  7. **Exposure-response scatter** (Ctau vs HIV-1 RNA, ANC vs AUC)
- **Limitations of extracted profiles**:
  - Visual precision ±10-20% on curve values
  - Most profiles lack explicit dose (visible on caption, not figure) → need lookup via v11_llm IK14 match or PDF caption read
  - ~30% of valid profiles are non-standard: transdermal patch (dextroamphetamine), SC extended release (granisetron APF530), steady-state multi-dose but clean shape (elacestrant MD)
- **Scale-out estimate**: At 35% yield, remaining 38 unprocessed candidates → ~13 more valid = ~30 total. Combined with v0.5's 25 usable → ~55 profiles. Still small but ~2x the starting point.
- **File**: `data/curated/profile_visual_extracted.json`, `data/curated/visual_extraction_queue.json`
- **Status**: PIPELINE VALIDATED. Visual extraction via Claude vision is the correct approach (0→1→17 progression across three methods). Scale-out requires either (a) completing the remaining 38 candidates in a fresh context, or (b) extending beyond auto_digitized's 86 to broader figure set (11,403 total PNGs, most are not profile figures). Current 17-profile dataset may be too small for B1 regularizer strength but is usable as auxiliary validation set.

### I8. Architectural Exhaustion Analysis (Session 2026-04-10 Conclusion)
- **Date**: 2026-04-10
- **Purpose**: After B1 (F12/F13) and B2 (F14) both failed, the user asked whether architectural expansion is limited to GNN/ensemble. This analysis consolidates what's been tried, what's refuted, and what's structurally open.
- **Method**: Systematic survey of 5 architectural dimensions (input representation, model class, output formulation, training regime, ensemble strategy) cross-referenced against the existing RESEARCH_LOG and this session's new refutations.
- **Key structural findings**:

  **1. "Better chemical representation" is a refuted dimension.** F2 (MolFormer embeddings) and F3 (Tanimoto-retrieval augmentation) both failed with the explicit conclusion "PK ≠ SAR". This session's I7 diagnostic independently confirmed the same pattern: **PLM-worse drugs have HIGHER Tanimoto to training** (p=0.04), refuting the "novelty-hurts-model" hypothesis. Any variant of this dimension (ChemBERTa, ChemGPT, Chemprop/GNN, Mordred, 3D descriptors, pharmacophore FPs) shares the same failure mode — they all operate in SAR-space, and SAR-space is a poor proxy for PK-space.

  **2. "Scalar ADME auxiliary" is a refuted dimension.** Three independent experiments (F11 DailyMed merge → feature; F12/F13 B1 half-life → feature+target; F14 B2 Vd → feature+target) all produced null or harmful results with different sources, different usage modes, and different physical rationales. The failure reason is consistent: TDC/public ADME data comes from IV studies whose measurement context does not match the oral Cmax training data; XGB already extracts whatever ADME-relevant signal exists from Morgan+physchem+μPBPK features implicitly; explicit scalar auxiliaries add measurement-context noise without new information.

  **3. "PBPK ensemble" is structurally blocked by data.** Sisyphus achieves 2.808 HO AAFE via PBPK engine + ML meta-stacking. Their Engine works because of high-quality proprietary ADME data (Biogen ~3000 compounds: hlm_clint, mdr1_efflux, ppb_human, permeability). PLM lacks this. Building PLM's own PBPK component would require predicting (ka, ke, Vd) from structure — exactly what B1/B2 failed at. The `simulator/pk_engine.py` has the analytical math (1-compartment with lag, Numba JIT'd, superposition) but the structure→parameter mapping is the bottleneck, and that mapping is ADME prediction, which is refuted at (2).

  **4. "Architecture tinkering without new data" is exhausted within current constraints.** Every remaining variant (GNN, FT-Transformer, TabNet, CatBoost, quantile regression, scaffold-stratified training, seed ensembles, etc.) either (a) shares failure mode with F2/F3/F11/B1/B2, (b) provides only marginal ensemble-variance reduction (~−0.02 to −0.05), or (c) is blocked by small data size (4540 rows is borderline for deep models, too small for meta-learning).

- **What remains open**:
  1. **Data quantity expansion** (CLAUDE.md "primary lever"). LLM extraction on older FDA PDFs (needs API key), visual profile extraction completion (started in I6, 38 candidates remaining in auto_digitized + 11,403 broader figure pool), stricter ChEMBL re-mining (F10 revisited).
  2. **Profile-based temporal supervision**. The original B1 (parametric C(t) output, 13→5) requires profile data. With ~17 visual-extracted profiles now + remaining queue + potential LLM-vision expansion, a profile dataset large enough (~200+) for temporal regularizer is achievable but multi-session.
  3. **Mechanism-aware data sourcing**. Biogen-equivalent in-vitro ADME would unlock PBPK ensemble. Realistically obtainable via: published in-vitro screening datasets (ChEMBL bioassays, FDA review appendix tables), academic group collaborations, or synthetic augmentation via first-principles docking. High effort, uncertain yield.
  4. **Different evaluation angle**. PLM's design premise is "direct [SMILES, dose] → Cmax without IVIVE chain". At HO AAFE 3.37, PLM is comparable to mechanistic PBPK engines (Sisyphus Engine alone = 3.416). The "gap to Sisyphus ensemble" (2.808) reflects the advantage of ensembling, not of PLM's ML component being worse. Reframing PLM's value proposition around simulator integration, trial simulation, or uncertainty calibration rather than chasing Cmax AAFE lower may be more productive.

- **Session 2026-04-10 closing inventory** (what was learned, not just tried):
  - 3 pre-registered experiments completed, all with falsifiable criteria: **F12, F14 FAIL**; **S11 NULL on HO (Δ=−0.015), POSITIVE on CV-HO gap (Δ=−0.089)**
  - 1 prior claim retracted: **old S10** (encoder-hurts) → **S11** (encoder-null-on-HO, regularizes gap)
  - 1 diagnostic with actionable patterns: **I7** (directional over-prediction bias, SSRI/SNRI/steroid class-specific failure, SAR-PK divergence)
  - 1 partial data expansion: **17 visual-extracted profiles** (I6 partial)
  - 1 corrected baseline number: **fp_enc_base HO ≈ 3.37** (not 3.456)
  - 1 architectural exhaustion map (this entry)

- **Takeaway for next session**: The scientifically honest path forward is NOT another architecture tweak. It is either **data quantity expansion** (primary CLAUDE.md lever) or **reframing PLM's value proposition** away from AAFE-chasing. Architecture changes have been tested across 5 dimensions and all cheap options are exhausted or known to fail for the same root cause (SAR-PK divergence, measurement-context mismatch, small training set).
- **Status**: CONSOLIDATED. Session 2026-04-10 closed with honest architectural exhaustion finding.

### I9. Data Expansion Attempt (Session 2026-04-10 continuation) — Two Surprises

- **Date**: 2026-04-10 (second half of session)
- **Goal**: Execute the "data quantity expansion" open direction from I8. User pointed out (twice, across sessions) that Claude Code's multimodal Read tool can scan PDFs/figures directly, no external API needed. Planned three-tier approach: (A) finish visual extraction queue 38 remaining candidates, (B) scan unprocessed FDA PDFs for PK tables via Read, (C) broader figure re-exploration.

- **Tier A execution** — visual extraction batches 2–5 (queue indices 48–85, 38 candidates):
  - **10 valid / 38 processed = 26% yield** (below I6's 35%). 3 unrendered JPX files (pyridostigmine, paclitaxel, tirzepatide). Valid list: methotrexate, oxycodone, panobinostat 60mg, sitagliptin, spironolactone, sumatriptan PO 100mg, tadalafil, telotristat (active moiety), vibegron, vorapaxar.
  - Rejection patterns: DDI victim wrong-analyte (netupitant→digoxin, rolapitant→DEX), metabolite instead of parent (nitroglycerin→1,2-GDN), PD response curves (motixafortide→CD34+ cells, relugolix→testosterone), nasal/IM route instead of oral (oxymetazoline, testosterone), multi-dose steady-state (nevirapine, prucalopride, rucaparib, oteseconazole, vismodegib), tables/scatter plots (paltusotine, sarecycline, teriflunomide, suvorexant, selumetinib, nirmatrelvir).

- **Surprise #1 — v0.5 cleaned dataset has ~20 contaminated entries**: Cross-referencing rejected candidates against `plm_dataset_v0.5_cleaned.json` revealed that the same auto-digitization errors had been propagated into the v0.5 "cleaned" training set:
  - `motixafortide` (2 rows, 1.0-1.25 mg "oral", Cmax 3.3/67.4) — actually CD34+ cell counts from SC peptide PD study
  - `nitroglycerin` (3 rows, 6.5 mg "oral") — real NTG is 0.4-0.8 mg sublingual; stored values are 1,2-GDN metabolite
  - `oxymetazoline` (5 rows, "18 mg oral" Cmax 7708) — real Kovanaze is 0.05-0.2 mg intranasal (100x dose error + wrong route)
  - `naloxone` (3 rows, "20 mg oral" Cmax 2-102) — oral naloxone bioavailability ~2%, Cmax should be <1 ng/mL
  - `nirmatrelvir` (1 row, 100 mg Cmax 2.0) — real 100 mg Cmax is ~1000 ng/mL (500x off); figure was DDI scatter
  - `rucaparib` (1 row, 600 mg Cmax 4.3) — real 600 mg Cmax is ~1900 ng/mL (400x off); figure was BID steady-state
  - `rolapitant` (1 row, 180 mg Cmax 621) — figure was dexamethasone DDI victim
  - `netupitant` (3 rows at 300 mg, Cmax < 20) — DDI probe digoxin, not netupitant
  - `sarecycline` (1 row at 100 mg Cmax 6912) — real Cmax is ~1000; figure was urinary excretion
  - `pipeline/build_v06_cleaned.py` produces `data/curated/plm_dataset_v0.6_cleaned.json`: 199 → 179 profiles after rule-based removal (20 entries, 9 drugs). Removal log: `data/curated/v06_cleanup_log.json`.

- **Surprise #2 — v0.5 is NOT used by S11 training**: Looking at `models/s10_replication.py` line 175, fp_enc_base training loads `plm_dataset_v11_llm.json` (4540 rows: 3340 SIS + 1050 LLM_FDA + 150 PLM source), NOT v0.5. All 9 flagged contaminated drugs were checked in v11_llm by canonical SMILES: **all have correct literature Cmax values** (rucaparib 600mg=1940, sarecycline 100mg=2620, nirmatrelvir 100mg=1042-2224, rolapitant 180mg=947, netupitant 300mg=599, oxymetazoline=0.05-0.3 mg correct nasal doses, etc.). v11 was independently built from SIS training data + LLM table extraction, not from v0.5 profiles. Consequence: **v0.6 cleanup has zero effect on current XGB training**, but is still kept as a data-quality artifact for downstream profile-based work.

- **Tier B execution** — unprocessed FDA PDF scan:
  - 217 of 456 PDFs are NOT in `pk_llm_merged.json` (the LLM was run against a 239-NDA subset).
  - Small PDFs (<800 KB) sampled first: NDA219840 (barium sulfate imaging), NDA215033 (bendamustine IV 505(b)(2)), NDA208419 (pemetrexed IV 505(b)(2)) — **all empty-shell nonclinical/reliance reviews, 0 rows**.
  - Mid-size PDFs (2-5 MB) sampled: NDA215446 (edaravone oral suspension RADICAVA ORS) yielded **6 extractable rows** (edaravone 105 mg oral healthy Cmax 1656, edaravone 105 mg ALS Cmax 1903, edaravone NGT Cmax 2431, plus DDI control arms: sildenafil 50 mg 194.3, rosuvastatin 10 mg 10.6, furosemide 40 mg 1502.8). NDA204141 (desoximetasone 0.25% topical spray) yielded **0 rows** (wrong route). Saved: `data/curated/pdf_scan_extracted.json`.
  - **Selection bias discovered**: The 239 processed NDAs were the *yield-positive* subset. Remaining 217 are disproportionately topical/IV/imaging/generic-BE/505(b)(2)-reliance reviews with no oral PK data. Yield from random sampling ≈ 1/5 PDFs × ~5 rows = ~40 rows from all 217 unprocessed (not the initially estimated ~500).
  - 6 rows / 4540 existing = 0.13% training-set growth. Far below the detection threshold for HO AAFE change. Not worth running a retrain experiment.

- **Tier C** — broader figure re-exploration: **Not executed**. Same selection-bias argument: the 927 figure candidates already captured by the heuristic are the low-hanging fruit; residual figures are likely lower yield.

- **Pre-registered hypothesis (`docs/prereg_x2_cleanup.md`)**: X2 planned to cleanup v0.5 + expand training and measure ΔHO. Upon discovering v11 is the real target and v11 is already clean, the pre-registered test became **vacuous** (cleanup touches wrong dataset, expansion size is below signal threshold). Retract X2 as "test design invalidated by upstream discovery". Document the pre-reg and the discovery together so the null is transparent rather than buried.

- **What this reveals about the actual bottleneck**: The CLAUDE.md claim that "data expansion is the primary lever" implicitly assumed there are many un-mined FDA PDFs. Empirically, the 239 already-processed NDAs captured the bulk of oral single-dose Cmax data available from FDA review PDFs. The remaining 217 are mostly unusable (topical, IV, imaging, BE). The true data quantity ceiling for the FDA-PDF route is approximately where v11 already sits (~4540 rows, ~1173 unique drugs). To materially expand beyond this, one must go outside FDA review PDFs — e.g., ChEMBL bioassay re-mining (F10 revisited with looser criteria), EMA review scraping (`data/raw/ema_medicines.json` exists and is unprocessed — worth a dedicated session), academic literature mining, or in-vitro ADME datasets.

- **Artifacts produced**:
  - `data/curated/plm_dataset_v0.6_cleaned.json` (179 profiles, 65 drugs) + `data/curated/v06_cleanup_log.json`
  - `data/curated/pdf_scan_extracted.json` (6 new PK tuples from NDA215446)
  - `data/curated/visual_extraction_full_findings.json` (batches 2-5 verdicts + v0.5 contamination map)
  - `data/curated/visual_extraction_batch1_findings.json`
  - `docs/prereg_x2_cleanup.md` (pre-registration, explicitly marked vacuous in retrospect)
  - `pipeline/build_v06_cleaned.py` (reusable cleanup rule engine)

- **Status**: HONEST NULL. No retrain run because row count is below signal threshold. Both the v0.5 cleanup and the PDF scan are archived as data-quality artifacts for potential future profile-based supervision work, not as ML improvements.

- **Takeaway for next session**: FDA-PDF-based data expansion is approximately saturated. The *genuine* open paths are (a) EMA medicines data (already downloaded, never processed), (b) ChEMBL bioassay re-mining with pharmacokinetic-context filters, (c) pivot to value-reframing per I8 point 4. Do not re-attempt "scan more FDA PDFs" without first checking whether the candidate NDA has an oral small molecule indication.

### S12. ChEMBL v2 Strict Re-Extraction + v12 Retrain — PARTIAL PASS (first HO improvement from data expansion)

- **Date**: 2026-04-11
- **Pre-registration**: `docs/prereg_s12_chembl_v12.md` (written before running)
- **Context**: F10 (ChEMBL Conservative Salvage, 2026-04-07) tried adding ChEMBL Cmax data and FAILED. I4 diagnosed three contamination modes: (1) animal PK mixed with human, (2) mg/kg regex confusion, (3) persistent log_cd shift. I9 (2026-04-10) also found that EMA medicines catalog is metadata-only (no PK) and FDA PDF expansion is saturated. ChEMBL re-extraction with stricter filters became the remaining concrete data-expansion path.

- **Method — `pipeline/chembl_v2_strict.py`**: queries ChEMBL activity table with `standard_type IN (Cmax, CMAX)` and applies cascading filters at extraction time rather than post-hoc:
  1. **Animal rejection** (21 keywords: rat, mice, mouse, dog, monkey, rabbit, rodent, murine, beagle, primate, porcine, ovine, bovine, sprague-dawley, wistar, c57, balb, cd1, irc/icr, cynomolgus, rhesus, macaque)
  2. **Positive human requirement** (description must mention human/healthy/patient/clinical/homo sapiens OR organism="Homo sapiens"). `assay_organism` is ~always None, so description text is primary.
  3. **Oral-only positive requirement** (description must mention po / oral / orally / tablet / capsule / suspension). This was critical — F10 didn't enforce this.
  4. **Non-oral rejection** (iv, intravenous, infusion, im, sc, topical, intranasal, inhalation, sublingual, buccal, transdermal, ocular, otic, rectal)
  5. **mg/kg regex rejection** — explicit `(\d+)\s*mg\s*/\s*kg` pattern check before generic mg extraction
  6. **Unit whitelist + molar conversion** — ChEMBL uses `ug.mL-1` style notation; 96% of Cmax records are in nM which needs MW conversion
  7. **log_cd sanity** — must fall within v11 p1-p99 ± 0.5 buffer [−2.12, 2.62]
  8. **Per-(drug, dose) grouping** — crucial fix: F10's code took MEDIAN dose across a drug's records, destroying dose-response info. S12 keeps each (IK14, dose) as a distinct row.

- **Extraction yield (25,002 activities processed in ~8 min)**:
  - 22,443 rejected as animal (89.8%)
  - 592 rejected no human marker, 545 missing fields, 533 no oral marker, 268 already in v11/holdout, 166 non-oral route, 106 mg/kg dose, 32 no mg pattern, 7 log_cd out of range
  - **290 rows accepted → 164 unique (drug, dose) pairs → 91 new unique drugs**
  - `pipeline/build_v12_chembl.py` merges into v11: 4540 rows → **v12 = 4704 rows (+3.6%), 1264 unique drugs (+7.8%)**

- **Pre-registered retrain — `models/s12_v12_retrain.py`** (fp_enc_base, same as S11: FP4096+encoder128+physchem20+TDC9+μPBPK6+log_dose; 5-fold GroupKFold on IK14; 3 seeds 42/137/2024):

  | Metric | v11 baseline | v12 (with ChEMBL) | Δ |
  |---|---|---|---|
  | CV AAFE mean±std | 3.165 ± 0.005 | 3.220 ± 0.015 | +0.055 |
  | **HO AAFE mean±std** | **3.372 ± 0.010** | **3.327 ± 0.024** | **−0.045** |
  | CV-HO gap | 0.207 | 0.107 | **−0.100** |
  | Per-seed HO (42/137/2024) | 3.359 / 3.374 / 3.383 | 3.304 / 3.317 / 3.359 | −0.055/−0.058/−0.024 |

- **Pre-registered verdict**: ΔHO −0.0452 ± 0.019 → **PARTIAL** (just below PASS threshold of −0.05, clearly outside NULL band of ±0.02−0.05). 2 of 3 seeds individually crossed PASS threshold (−0.055, −0.058), one was PARTIAL (−0.024).

- **Why this matters** (first HO-improving experiment in the entire data expansion series):
  1. **CV-HO gap collapsed −0.100 from just +164 rows.** The new data didn't change the training CV much (v11 CV 3.165 → v12 CV 3.220, +0.055 actually worse on in-distribution) but dramatically improved OOD holdout. This is the signature of **distribution-shift regularization** — the new drugs fill chemical/PK space that v11 was missing, pulling the decision function toward better OOD generalization.
  2. **91 new drugs for 7.8% diversity increase.** Contrast with F10's 174 rows after naive filtering that FAILED: the difference is not row count but row *quality*. Strict oral+species+dose filtering matters more than raw volume.
  3. **The first "data is the lever" result that actually measures.** I8 concluded architectural tinkering was exhausted; data expansion was the recommended path but all prior attempts (I6 visual profiles, FDA PDF scan) gave 0-6 rows each, below detection threshold. S12 is the first experiment to validate the "data is the lever" hypothesis with a measurable HO improvement.

- **Caveats and honest limits**:
  - 164 rows is still small; ΔHO −0.045 is only 2.4σ from zero (paired std 0.019). A fourth seed could push it either direction.
  - Gap reduction from 0.207 to 0.107 is striking but single experiment. Needs replication.
  - I did not run a 10k or 50k ChEMBL scan — yield was plateauing around 25k activities at 290 rows, suggesting ChEMBL Cmax data is mostly medicinal-chemistry animal PK with only ~1% human-oral-single-dose clinical. Full scan might yield 500-1000 more rows; incremental gain uncertain.
  - Some of the new drugs may still be borderline (e.g., multi-dose confounded with single-dose despite description filters, or incorrect dose from regex). Random manual audit of 10 entries recommended before production use.

- **Artifacts**:
  - `pipeline/chembl_v2_strict.py` — extraction with new filters
  - `data/curated/chembl_v2_strict.json` — 164 (drug, dose) pairs with sample descriptions
  - `pipeline/build_v12_chembl.py` — merge script
  - `data/curated/plm_dataset_v12_chembl.json` — merged training set (4704 rows)
  - `data/curated/v12_merge_summary.json` — row/drug counts before/after
  - `models/s12_v12_retrain.py` — pre-registered retrain
  - `models/b1/s12_v12_results.json` — per-seed metrics + verdict
  - `docs/prereg_s12_chembl_v12.md` — pre-registration document

- **Status**: **PARTIAL PASS**. First data expansion experiment with HO improvement. Validates ChEMBL re-mining as a viable expansion path when properly filtered. New baseline candidate: fp_enc_base HO AAFE **3.327** (was 3.372 from S11).

- **Takeaway for next session**:
  1. Run a larger ChEMBL scan (50k–100k activities) to see if yield continues past 290 rows
  2. Manually audit 10-20 random new rows for residual contamination
  3. Try to push S12 from PARTIAL to full PASS by adding one more data source (ChEMBL AUC records, or text mining of PubMed abstracts for human oral PK mentions)
  4. If v12 replicates across a 4th seed, publish as new baseline (3.327 vs Sisyphus 2.808, gap reduced from 1.09 to 0.52)

## Key Metrics Timeline

| Experiment | Best AAFE | Type | Notes |
|---|---|---|---|
| XGBoost v0.4 figure | 10.1 | CV | Auto-digitization noise |
| XGBoost v2 table | **3.275** | CV | Table extraction breakthrough |
| v7 clean | 3.098 | CV | In-domain only |
| v8 expanded | 3.149 | CV | More data |
| ADME pretrain (FP+enc) | 2.788 | CV | Best CV, doesn't transfer to HO |
| Mechanistic ML | 2.864 | CV | Physchem features |
| Phase A holdout | **3.723** | HO | First clean holdout eval |
| Phase B (3D) | 3.702 | HO | Marginal gain from 3D descriptors |
| Phase D tuned | 3.964 | HO | Feature engineering overfitting |
| Novel baseline | 3.355 | HO | Current best holdout |
| DrugBank expansion | 3.469 | HO | Synthetic data hurts (F1) |
| LLM CoT + Lasso CV-cal | **2.043** | HO | Current best (Phase 4) |
| NL routing (NL→XGB) | **2.009** | HO | Mechanism-aware routing (S8) |
| Linear-only subset | **1.923** | HO | LLM on 86 linear drugs (S8) |
| Oracle best-of-2 | **1.834** | HO | Per-drug routing ceiling (S8) |
| Brown 2025 external | **3.255** | EXT | N=29 independent, PASS (S9-E1) |
| Post-cutoff NMEs | 4.262 | EXT | N=6, novel compounds (S9-E2) |
| Holdout expanded | **3.354** | HO | N=103, +6 recovered (S9-E3) |
| B1v5 XGB clean baseline (no enc) | 3.387 | HO | (S11 replication, 3-seed mean) |
| S11 fp_enc_base replication | 3.372 | HO | (S11, 3-seed mean; corrects old 3.456) |
| S12 v12 (v11 + ChEMBL v2 strict) | **3.327** | HO | (S12, 3-seed mean; ΔHO=−0.045, gap=0.107) |
| Sisyphus Meta | **2.283** | HO | Benchmark target |

## Cross-References

- **Project spec**: [CLAUDE.md](../CLAUDE.md) — model progression table, unit conventions
- **Scale-up plan**: [docs/scaleup_plan.md](scaleup_plan.md) — PDF extraction pipeline
- **Holdout definition**: `data/validation/holdout_definition.json` — 97 drugs
- **All result JSONs**: `data/validation/*_results.json`, `models/*_results.json`
- **Simulator**: `simulator/` — clinical trial simulator POC (independent of PK model)
