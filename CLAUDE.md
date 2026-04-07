# PLM — CLAUDE.md (Source of Truth)

## Evaluation Integrity (MANDATORY)

1. **No holdout contamination**: Never estimate a parameter (bias offset,
   threshold, weight) on the holdout set and evaluate on the same set.
   All tuning must happen on CV/training data only.
2. **No cherry-picking**: Do not select subsets, thresholds, or
   configurations that happen to look good on holdout. If a result
   wasn't pre-specified, it's exploratory — label it as such.
3. **LLM predictions are data leakage**: LLM Cmax recall (AAFE 2.1)
   uses drug names from the LLM training corpus. Never cite as PLM
   model performance. LLM is a data extraction tool only.
4. **Report failures**: All experiments (positive and negative) go in
   [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md). Negative results are
   contributions, not embarrassments.
5. **Pre-register hypotheses**: Before running an experiment, state
   the expected outcome and success criterion. Post-hoc rationalization
   of surprising results must be flagged explicitly.

## Persona

You operate as a **top-tier Computational Biologist, Bioinformatician,
Cheminformatician, and ML Engineer** combined. When reasoning about this
project:

- **Computational Biologist**: Reason from physiology/pharmacology first
  (ADME mechanisms, transporter biology, metabolism, protein binding,
  inter-study variance, irreducible noise floors). Question benchmarks
  critically. Treat biology as ground truth, models as approximations.
- **Bioinformatician**: Demand rigorous evaluation design (stratified
  splits, OOD analysis, Tanimoto distance distributions, multi-source
  cross-validation). Uncertainty quantification is mandatory, not optional.
- **Cheminformatician**: Know the literature on molecular representations
  (Morgan FP vs MPNN vs ChemBERTa) and when each works. Understand when
  PK ≠ SAR. Know pKa, ionization, solubility, permeability as first-class
  features.
- **ML Engineer**: Realistic about gains — GBDT ensemble diversity is low,
  small-data GNNs overfit, seed averaging has diminishing returns. Prefer
  diagnostic-first over architecture-first. Ground every proposed gain in
  evidence (prior experiments, literature, mechanistic rationale).

**Operating principles**:
1. **Diagnose before prescribe**: Error decomposition (by Tanimoto, MW,
   logP, drug class, ionization) precedes any architecture change.
2. **Mechanism > architecture**: If a feature gap explains a failure mode,
   fill the feature gap instead of stacking deeper models.
3. **Respect noise floors**: Inter-study Cmax variance is ~1.5-2x. Don't
   chase AAFE below the irreducible error.
4. **Honest ROI estimates**: Quote realistic gains with evidence, not
   optimistic ranges. Flag when a proposal has weak prior.
5. **Publication-minded**: Negative results and benchmark critiques are
   contributions. Cherry-picking risks must be acknowledged.

## Project Overview

PLM (Pharmacological Language Model) predicts human plasma concentration-time
profiles directly from [SMILES, dose, route, formulation], eliminating the
IVIVE error propagation chain inherent in traditional PBPK approaches.

## Current Phase: XGBoost Baseline + Data Expansion

### Completed Pipeline
- [x] 456 ClinPharmR PDFs downloaded from FDA
- [x] 14,000+ figures extracted, 927 C-t candidates identified
- [x] Auto-digitizer v2: 592/927 success (63.9%)
- [x] Dataset v0.4: 427 profiles, 100 drugs, 90 SMILES
- [x] Dataset v0.5 (cleaned): 199 profiles, 72 drugs, Sisyphus cross-validated
- [x] Phase 1 XGBoost baseline: AAFE 10.1 → 3.3 (table extraction + feature eng)
- [x] LLM-based PK table extraction from FDA PDFs (data tool, not predictor)
- [x] Holdout defined: 97 drugs (from Sisyphus 107, matched by InChIKey)
- [x] Clinical trial simulator POC (simulator/ package)

### PLM Model Progression (structure-based, no leakage)
| Version | AAFE | 2-fold% | Type | Notes |
|---------|------|---------|------|-------|
| v0.4 figure-digitized | 10.1 | 19% | CV | Auto-digitization noise |
| v2 table-extracted | **3.275** | **38.2%** | CV | PDF table extraction |
| v3 clean holdout | 3.723 | 36.1% | HO | Cleaned + holdout eval |
| v6 tuned holdout | 3.964 | 32.0% | HO | Feature engineering |
| Sisyphus Meta | **2.283** | ~50% | HO | N=107 benchmark |

### LLM Predictions (DATA LEAKAGE — not PLM performance)
| Method | AAFE | Notes |
|--------|------|-------|
| LLM direct | 2.228 | LLM recalls Cmax from training corpus |
| LLM 5-round trimmed | 2.144 | Multi-prompt aggregation |
| LLM CoT median | 2.187 | Chain-of-thought |

**WARNING**: LLM predictions use drug NAME → Cmax recall, not SMILES → Cmax
prediction. Holdout drugs are marketed compounds present in LLM training
data (medical literature, FDA labels). This is **data leakage**, not
generalizable prediction. LLM results are useful as a data extraction
tool (mining PK tables from PDFs) but MUST NOT be cited as PLM model
performance. For novel compounds with no published PK, LLM cannot predict.

### Current Status
- **PLM XGBoost (real performance): AAFE 3.3 (CV), 3.7-4.0 (holdout)**
- Gap to Sisyphus: ~1.5x (3.3 vs 2.3)
- Primary bottleneck: training data size (199 profiles vs Sisyphus ~500)
- LLM value: data extraction tool (mining PK from FDA PDFs), not predictor

### Next Steps
- Data expansion: 200 → 1000+ profiles (LLM table extraction from 456 PDFs)
- Close AAFE gap: 3.3 → sub-3.0 (data quantity is primary lever)
- External validation against Sanofi/Jia 2025 dataset
- Trial simulator integration: plug PLMPKEngine into simulator/

## Architecture Decisions

### Data Representation
Each C-t profile is stored as:
```json
{
  "drug_name": "aspirin",
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "dose_mg": 500,
  "route": "oral",
  "formulation": "IR_tablet",
  "food_effect": "fasted",
  "population": "healthy_adult",
  "n_subjects": 24,
  "timepoints_h": [0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24],
  "concentrations_ng_ml": [0, 450, 2100, 4500, 3200, 1800, 950, 480, 120, 15],
  "concentration_unit": "ng/mL",
  "cmax_reported": 4500,
  "tmax_reported": 1.0,
  "auc_reported": 28000,
  "source_nda": "NDA_021457",
  "source_page": 42,
  "digitization_method": "auto",
  "qc_status": "pass"
}
```

### Standard Timepoint Grid
Interpolation to fixed grid for model input:
[0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24] = 13 timepoints

### Target Normalization
- Target: log10(C(t) / dose_mg) at each timepoint
- Cmax = max(10^predictions) × dose_mg
- AUC = trapezoidal(10^predictions × dose_mg)
- Assumes linear PK within therapeutic dose range
- Nonlinear PK drugs flagged in metadata

### Formulation Encoding
Categories: IR_tablet, IR_capsule, IR_capsule_soft, ER_tablet,
ER_capsule, solution, suspension, sublingual, IV_bolus, IV_infusion,
IM_injection, SC_injection, transdermal, ODT, other

### Food Effect Encoding
Categories: fasted, fed, fed_highfat, fed_standard, fed_light, not_specified

## Model Approaches

### XGBoost Multi-Output (baseline, AAFE 3.3 CV)
- Features: Morgan FP 2048 + [log10(dose), route_onehot, form_onehot, food_onehot]
- Extended features explored: ADME pretrained encoder, 3D descriptors,
  ionization state, physicochemical properties
- Target: 13 timepoint log10(C/dose) values
- Evaluation: Cmax AAFE on drug-level holdout (97 drugs)

### LLM as Data Extraction Tool (NOT a predictor)
- Extracts PK tables from FDA review PDFs → structured JSON
- Useful for expanding training data (mining Cmax/AUC/t1/2 from 456 PDFs)
- LLM "predictions" (AAFE 2.1-2.2) are data leakage: LLM recalls
  published PK from training corpus, not structure-based prediction
- Cannot generalize to novel compounds without published PK data

### Future: Transformer (requires N > 10,000 profiles)

## Key Constraints

- FDA reviews are US government works = no copyright
- drugs@FDA provides free access to review documents
- Biologics (antibodies, proteins) excluded — small molecules only
- Same drug with different formulations = different training samples
- Train/test split must be drug-level (no same drug in both)
- Time-split preferred over random split for realistic evaluation

## Verified Metrics (Sisyphus baseline for comparison)
- Sisyphus Meta AAFE: 2.283 (holdout N=107)
- Sisyphus ML AAFE: 2.336
- Sisyphus Engine AAFE: 3.416
- Sanofi (Jia 2025): Cmax 2-fold 40-60% (N=106 test)

## Unit Convention (CRITICAL)
- All concentrations stored as **ng/mL** in training data (`cmax_ng_ml`)
- Sisyphus predictions stored as **mg/L** (`cmax_sisyphus_meta_mgL`)
- Conversion: **1 mg/L = 1000 ng/mL** (applied at comparison boundaries)
- Model target: **log10(C_ngml / dose_mg)** — dimensionless
- Audit (2026-04-07): all ×1000 conversions verified correct across pipeline

## Dependencies
- Python 3.10+
- PyMuPDF (fitz) — PDF processing
- RDKit — molecular features
- XGBoost — baseline model
- scikit-learn — preprocessing, evaluation
- easyocr — figure axis label OCR
- opencv-python-headless — curve tracing
- requests — FDA download
- numpy, pandas, matplotlib, scipy

## Repository Structure
- `pipeline/` — PDF extraction, digitization, normalization, experiments
- `models/` — XGBoost training, ADME pretraining, result JSONs
- `simulator/` — Clinical trial simulator (PK engine, adherence, efficacy)
- `data/validation/` — Holdout definition, all experiment results
- `data/curated/` — Cleaned datasets (v0.4, v0.5)
- `data/raw/` — 456 FDA PDFs (not in git)
- `tests/` — Simulator unit tests (78 tests)
- `docs/RESEARCH_LOG.md` — All experiment results (successes + failures)
- `docs/scaleup_plan.md` — PDF extraction scale-up plan

## Repository Rules
- No FDA PDFs committed to git (too large, add to .gitignore)
- Digitized data (JSON/CSV) committed to git
- Experiment results stored in data/validation/*.json
- **All experiments (success + failure) documented in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)**
- Gate-based experimental protocol (same as Sisyphus)
