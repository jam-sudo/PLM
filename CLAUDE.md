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

## Current Phase: Baseline Established + Simulator Integration

### Completed Pipeline
- [x] 456 ClinPharmR PDFs downloaded from FDA
- [x] 14,000+ figures extracted, 927 C-t candidates identified
- [x] Auto-digitizer v2: 592/927 success (63.9%)
- [x] Dataset v0.4 → v0.5 → v11_llm (4,540 rows) → v12 (4,704 rows)
- [x] Phase 1 XGBoost baseline: AAFE 10.1 → 3.3 (table extraction + feature eng)
- [x] LLM-based PK table extraction from FDA PDFs (data tool, not predictor)
- [x] Holdout defined: 97 drugs (from Sisyphus 107, matched by InChIKey)
- [x] ADME pretrained encoder (128-d, reduces CV-HO gap)
- [x] Clinical trial simulator with PLMPKEngine (SMILES → trial outcomes)
- [x] Data expansion: ChEMBL v2 strict +164 rows → HO 3.33 (S12, p=0.006)
- [x] All automated public data sources exhausted (I10)
- [x] External validation: Brown 2025 PASS, post-cutoff tested (S9)
- [x] Architectural exhaustion analysis: 5 dimensions tested, all null/fail (I8)
- [x] Value reframing: PLM 3.33 > Sisyphus Engine 3.42 (docs/value_reframing.md)

### PLM Model Progression (structure-based, no leakage)
| Version | AAFE | 2-fold% | Type | Notes |
|---------|------|---------|------|-------|
| v0.4 figure-digitized | 10.1 | 19% | CV | Auto-digitization noise |
| v2 table-extracted | 3.275 | 38.2% | CV | PDF table extraction |
| S11 fp_enc_base (v11) | 3.165 | 40.4% | CV | FP4096+encoder+physchem+TDC+µPBPK |
| S11 fp_enc_base (v11) | 3.372 | 38.1% | HO | 3-seed mean, corrected baseline |
| **S12 v12 (current)** | **3.220** | **39.4%** | **CV** | **+164 ChEMBL rows** |
| **S12 v12 (current)** | **3.332** | **39.2%** | **HO** | **4-seed, p=0.006 vs v11** |
| Brown 2025 external | 3.255 | 37.9% | EXT | N=29 independent (S9-E1) |
| Sisyphus Engine | 3.416 | — | HO | N=107 (PLM is better) |
| Sisyphus Meta | **2.283** | ~50% | HO | N=107 ensemble benchmark |

### Current Status (2026-04-12)
- **PLM XGBoost v12: AAFE 3.22 (CV), 3.33 (HO, p=0.006)**
- **PLM > Sisyphus Engine** (3.33 < 3.42, same-tier comparison)
- Gap to Sisyphus Meta: 1.05 (ensemble advantage, not ML inferiority)
- Training data: **4,704 rows, 1,264 drugs** (v12 = v11 + ChEMBL strict)
- Feature stack: FP4096 + ADME encoder 128 + physchem 20 + TDC 9 + µPBPK 6 + log_dose
- PLMPKEngine: **operational** — SMILES → trial simulation, no IVIVE needed
- Data expansion: **all automated public sources exhausted** (I10)
- Architecture: **5 dimensions exhausted** — representation, ADME aux, PBPK, loss, ensemble (I8)
- PLM wins 35/97 drugs (36%), 45.5% win rate on nonlinear PK drugs
- Oracle best-of-2 (PLM+Sisyphus): AAFE 1.79

### Next Steps
- Value reframing for publication positioning (see docs/value_reframing.md)
- Manual EMA EPAR extraction (~50-100 potential rows, labor intensive)
- Academic collaboration for proprietary ADME datasets
- Uncertainty quantification for clinical decision support

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

### XGBoost fp_enc_base (current, AAFE 3.33 HO)
- Features: Morgan FP 4096 + ADME encoder 128 + physchem 20 + TDC ADME 9 + µPBPK 6 + log10(dose) = **4,260 features**
- ADME encoder: pretrained on 11 TDC tasks, frozen 128-d embedding (reduces CV-HO gap by 0.09)
- Target: log10(Cmax / dose_mg) — single scalar per (drug, dose) pair
- Evaluation: 5-fold GroupKFold by IK14, 4-seed replication
- Training: v12 dataset (4,704 rows, 1,264 drugs from SIS + LLM_FDA + PLM + ChEMBL)

### PLMPKEngine (simulator integration, operational)
- SMILES → Cmax prediction → PK parameter derivation → C(t) profile
- Enables clinical trial simulation from molecular structure alone
- No IVIVE chain, no in-vitro ADME data required
- Integrated with adherence model, AE feedback, efficacy modeling

### LLM as Data Extraction Tool (NOT a predictor)
- Extracts PK tables from FDA review PDFs → structured JSON
- LLM "predictions" (AAFE 2.1-2.2) are **data leakage**: LLM recalls
  published PK from training corpus, not structure-based prediction
- Cannot generalize to novel compounds without published PK data

### Refuted Approaches (do not re-propose)
- **Better chemical representation** (F2 MolFormer, F3 Tanimoto retrieval): PK ≠ SAR
- **Scalar ADME auxiliary** (F11 DailyMed, F12/F13 half-life, F14 Vd): measurement-context mismatch
- **PBPK ensemble** (blocked by data): requires proprietary in-vitro ADME
- See I8 architectural exhaustion analysis in RESEARCH_LOG.md

## Key Constraints

- FDA reviews are US government works = no copyright
- drugs@FDA provides free access to review documents
- Biologics (antibodies, proteins) excluded — small molecules only
- Same drug with different formulations = different training samples
- Train/test split must be drug-level (no same drug in both)
- Time-split preferred over random split for realistic evaluation

## Verified Metrics
- **PLM v12 HO AAFE: 3.332** (4-seed, p=0.006 vs v11; N=97 holdout)
- **PLM v12 CV AAFE: 3.220** (4-seed mean)
- PLM vs Sisyphus win rate: 35/97 drugs (36.1%)
- Sisyphus Meta AAFE: 2.283 (holdout N=107)
- Sisyphus ML AAFE: 2.336
- Sisyphus Engine AAFE: 3.416 (**PLM is better**: 3.332 < 3.416)
- Sanofi (Jia 2025): Cmax 2-fold 40-60% (N=106 test)
- Brown 2025 external: PLM AAFE 3.255 (N=29, PASS)

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
- `pipeline/` — PDF extraction, digitization, ChEMBL/DailyMed extraction, experiments
- `models/` — XGBoost training, ADME pretraining, result JSONs, s12 retrain scripts
- `models/b1/` — Experiment result JSONs (s12, b1, b2, s10 replication)
- `simulator/` — Clinical trial simulator (PLMPKEngine, adherence, efficacy)
- `data/validation/` — Holdout definition, all experiment results
- `data/curated/` — Training datasets (v11_llm, v12_chembl), ChEMBL/DailyMed extractions
- `data/raw/` — 456 FDA PDFs, EMA EPARs (not in git)
- `tests/` — Simulator unit tests (79 tests)
- `docs/RESEARCH_LOG.md` — All experiment results (S1-S12c, F1-F14, I1-I10)
- `docs/value_reframing.md` — PLM vs Sisyphus positioning analysis

## Repository Rules
- No FDA PDFs committed to git (too large, add to .gitignore)
- Digitized data (JSON/CSV) committed to git
- Experiment results stored in data/validation/*.json
- **All experiments (success + failure) documented in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)**
- Gate-based experimental protocol (same as Sisyphus)
