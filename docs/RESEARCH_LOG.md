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
| Sisyphus Meta | **2.283** | HO | Benchmark target |

## Cross-References

- **Project spec**: [CLAUDE.md](../CLAUDE.md) — model progression table, unit conventions
- **Scale-up plan**: [docs/scaleup_plan.md](scaleup_plan.md) — PDF extraction pipeline
- **Holdout definition**: `data/validation/holdout_definition.json` — 97 drugs
- **All result JSONs**: `data/validation/*_results.json`, `models/*_results.json`
- **Simulator**: `simulator/` — clinical trial simulator POC (independent of PK model)
