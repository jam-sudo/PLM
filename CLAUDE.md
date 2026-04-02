# PLM — CLAUDE.md (Source of Truth)

## Project Overview

PLM (Pharmacological Language Model) predicts human plasma concentration-time
profiles directly from [SMILES, dose, route, formulation], eliminating the
IVIVE error propagation chain inherent in traditional PBPK approaches.

## Current Phase: Pre-Feasibility

### Immediate Goal
Validate that FDA Clinical Pharmacology Reviews can be programmatically
accessed, figures extracted, and C-t data digitized at scale.

### Success Criteria for Feasibility
- [ ] 10 FDA PDFs downloaded and figures extracted
- [ ] C-t figures identified vs non-C-t figures (>80% accuracy)
- [ ] 5 C-t profiles digitized with <10% error vs manual
- [ ] Metadata (dose, route, formulation) extracted from captions

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
Categories: IR_tablet, IR_capsule, ER_tablet, ER_capsule,
solution, suspension, sublingual, IV_bolus, IV_infusion,
IM_injection, SC_injection, transdermal, other

### Food Effect Encoding
Categories: fasted, fed, not_specified

## Model Phases

### Phase 1: XGBoost Multi-Output (current target)
- Features: Morgan FP 2048 + [log10(dose), route_onehot, form_onehot, food_onehot]
- Target: 13 timepoint log10(C/dose) values
- One XGBoost model per timepoint, or sklearn MultiOutputRegressor
- Evaluation: Cmax AAFE on drug-level time-split holdout

### Phase 2: Transformer (future, N > 10,000)
### Phase 3: Sisyphus Ensemble (future)

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

## Dependencies
- Python 3.10+
- PyMuPDF (fitz) — PDF processing
- RDKit — molecular features
- XGBoost — Phase 1 model
- scikit-learn — preprocessing, evaluation
- anthropic — Claude API for caption parsing
- requests — FDA download
- numpy, pandas, matplotlib

## Repository Rules
- No FDA PDFs committed to git (too large, add to .gitignore)
- Digitized data (JSON/CSV) committed to git
- All experiments documented in this file
- Gate-based experimental protocol (same as Sisyphus)
