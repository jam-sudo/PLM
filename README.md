# PLM: Pharmacological Language Model

Predicting human plasma Cmax from molecular structure and dosing conditions. Two complementary paradigms: a structure-based XGBoost baseline and an LLM-augmented approach that leverages published pharmacological knowledge via Chain-of-Thought reasoning with training-derived calibration.

## Concept

**Traditional PBPK** chains 7+ sequential models, each with prediction error that propagates multiplicatively:

```
SMILES → CLint → fup → Peff → Kp → IVIVE → ODE → C(t) → Cmax
```

**PLM** collapses this into a single prediction:

```
[SMILES, dose, route, formulation] → Cmax
```

## Current Results

### Best model: LLM CoT + CV-calibrator (holdout AAFE 2.043)

| Model | AAFE | 2-fold% | Evaluation | N (drugs) |
|-------|------|---------|------------|-----------|
| **LLM CoT + Lasso CV-calibrator** | **2.043** | — | 97-drug holdout | 97 |
| LLM CoT 3-round geomean (raw) | 2.127 | — | 97-drug holdout | 97 |
| LLM single-shot | 2.228 | — | 97-drug holdout | 97 |
| Sisyphus Meta | 2.283 | ~50% | 107-drug holdout | 107 |
| Sisyphus ML | 2.336 | — | 107-drug holdout | 107 |
| Sisyphus Engine | 3.416 | — | 107-drug holdout | 107 |
| PLM XGBoost (holdout) | 3.355 | 37.1% | 97-drug holdout | 97 |
| PLM XGBoost (CV best) | 3.275 | 38.2% | 5-fold GroupKFold | 1,191 |

**Statistical caveat**: Wilcoxon signed-rank test p=0.49 (two-sided, paired N=97). PLM wins 52/97 drugs (53.6%). Numerical advantage is real but not statistically significant at alpha=0.05 due to high per-drug error heterogeneity.

### Transductive disclosure

The LLM (Claude) has been pre-trained on FDA labels, PubMed, and medical literature, so it has prior knowledge of specific drugs' PK properties. The calibrator is fitted on training data only (zero holdout leakage), but the LLM's predictions reflect knowledge of published PK — not purely from-scratch structure-based prediction. This is distinct from data leakage (no holdout labels used in fitting), but should be understood as a knowledge-leveraging approach that may not generalize to truly novel compounds with no published PK data.

Full experiment history (22 experiments, including failures): [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)

## Architecture

### LLM-Augmented Pipeline (current best, AAFE 2.043)

```
Query drug (SMILES + dose)
  │
  ├─ Round 1: Physiological reasoning (F%, Vd, CL derivation)
  ├─ Round 2: Analogical reasoning (similar drugs + dose-scaling)
  └─ Round 3: FDA label recall (scaled to query dose)
  │
  ▼
Per-drug: geomean(log_cd), std(log_cd)
  │
  ▼
CV-validated Lasso calibrator (α=0.01)
  8 features selected by L1: std, MW, HBD, RingCount, MinPC, Charge, log_dose, LogP
  Fitted on 797 training drugs only
  │
  ▼
Cmax = 10^(predicted_log_cd) × dose
```

### XGBoost Baseline (AAFE 3.355)

```
SMILES → Morgan FP 2048 + PhysChem + TDC ADME + log10(dose) + condition one-hots
       → XGBoost (GroupKFold CV)
       → log10(Cmax/dose)
```

## Experimental Evolution

10 feature/architecture experiments failed to close the gap to Sisyphus → Shannon information analysis (S7) revealed 73% of error is model capacity gap, not generalization gap → paradigm shift to LLM knowledge leverage.

| Phase | Approach | Result |
|-------|----------|--------|
| 1. XGBoost baseline | Morgan FP + ADME features + data expansion | HO AAFE 3.355 (stable best) |
| 2. Novel architectures | MolFormer, delta learning, ADME encoder | All negative (chemistry representation not bottleneck) |
| 3. LLM direct prediction | Claude CoT as zero-shot PK predictor | HO AAFE 2.127 (3-round geomean) |
| 4. Training-derived calibration | Lasso on LLM uncertainty + physchem features | **HO AAFE 2.043** (beats Sisyphus Meta) |

## Data Pipeline

456 FDA Clinical Pharmacology & Biopharmaceutics Reviews → structured PK data.

| Stage | Output | Count |
|-------|--------|-------|
| PDF download (drugs@FDA) | FDA review PDFs | 456 |
| Figure extraction (PyMuPDF) | Figure images | 14,000+ |
| Auto-digitization (EasyOCR + OpenCV) | C-t profiles | 592/927 (63.9%) |
| LLM table extraction (Claude) | PK tuples (Cmax, AUC, t1/2) | 1,333 from 226 drugs |
| LLM PK prediction (Claude CoT) | 3-round Cmax predictions | 799 training + 97 holdout drugs |
| Unit normalization | Standardized ng/mL | All data |
| Training set (v10 + Sisyphus) | Model-ready profiles | 3,490 (1,191 drugs) |
| Holdout set | Evaluation drugs (Sisyphus-aligned) | 97 drugs |

## Model Details

- **XGBoost features**: Morgan FP 2048-bit + log10(dose) + route/formulation/food one-hot + physicochemical descriptors + TDC ADME predictions
- **LLM calibrator features**: LLM prediction std + log_dose + 15 RDKit descriptors (17 total), L1-selected to 8
- **Target**: log10(Cmax_ngml / dose_mg) — dose-normalized, dimensionless
- **Evaluation**: Cmax AAFE on 97-drug holdout (no drug overlap with training)
- **Unit convention**: All concentrations in ng/mL. Sisyphus predictions in mg/L (1 mg/L = 1000 ng/mL, converted at comparison boundaries)

## Clinical Trial Simulator

Standalone PK-driven trial simulator in `simulator/`. Simulates virtual clinical trials with:

- 1-compartment PK engine with allometric scaling and absorption lag time
- Two-state Markov adherence model with dose-timing jitter
- Concentration-dependent AE model (Cmax-driven sigmoid)
- Emax efficacy model (Ctrough-driven)
- PK-AE feedback loop: adverse events reduce adherence, reducing exposure
- Multi-arm dose-finding support

```bash
python -m pytest tests/test_simulator.py -v    # 78 tests
python -m simulator.demo                        # 4-arm dose-finding demo
python -m simulator.real_drug_test              # Random real drug simulation
```

## Reproducibility

```bash
# Current best: LLM CoT + Lasso CV-calibrator (HO AAFE 2.043)
python3 pipeline/cv_feature_calibration.py

# XGBoost baseline (HO AAFE 3.355)
python3 pipeline/ho_diagnostic.py

# LLM 3-round CoT aggregation (HO AAFE 2.127)
python3 pipeline/cot_self_consistency_eval.py
```

Regenerating LLM predictions requires Claude API access. Pre-computed predictions are saved in `data/llm_extracted/` and `data/validation/`.

## Known Limitations

- **Not statistically significant**: Wilcoxon p=0.49 on 97 drugs — numerical win, no statistical significance
- **Transductive**: LLM has seen holdout drugs in pretraining (disclosed, unavoidable for marketed compounds)
- **Single LLM dependency**: Only Claude tested; not validated with other LLMs
- **LLM determinism**: Minor variance across runs
- **Holdout size**: N=97 (10 dropped from Sisyphus 107 due to InChIKey mismatch)
- **Non-linear PK**: Saturable absorption not explicitly modeled

## Project Structure

```
PLM/
├── CLAUDE.md                          # Project spec (source of truth)
├── SYSTEM.md                          # System architecture review
├── docs/
│   ├── RESEARCH_LOG.md                # All experiments: successes + failures
│   └── scaleup_plan.md                # PDF extraction scale-up plan
├── pipeline/                          # Data extraction & experiments (37 scripts)
│   ├── cv_feature_calibration.py      #   Current best model (Lasso CV, 2.043)
│   ├── cot_self_consistency_eval.py   #   LLM 3-round CoT aggregation
│   ├── train_std_calibration.py       #   Std-adaptive linear calibrator
│   ├── ho_diagnostic.py               #   XGBoost holdout evaluation
│   ├── novel_experiment.py            #   XGBoost ablation experiments
│   ├── llm_extractor.py               #   PDF text → PK table extraction (LLM)
│   ├── scraper.py                     #   FDA PDF download
│   ├── auto_digitizer.py              #   Figure → C-t data (OCR + curve tracing)
│   ├── normalizer.py                  #   Unit normalization (ng/mL standard)
│   └── ...                            #   + 28 more experiment/evaluation scripts
├── models/
│   ├── train_xgboost.py               # Phase 1 XGBoost trainer
│   ├── pretrain_adme_xgb.py           # ADME feature pretraining
│   ├── novel_phase{1,2,3}.pkl         # Trained model checkpoints
│   └── *_results.json                 # Experiment results
├── simulator/                         # Clinical trial simulator
│   ├── patient.py                     #   Virtual population generator
│   ├── pk_engine.py                   #   Analytical PK + PLM adapter stub
│   ├── adherence.py                   #   Markov adherence + jitter
│   ├── pharmacology.py                #   AE (sigmoid) + efficacy (Emax)
│   ├── trial.py                       #   Multi-arm trial engine
│   ├── visualize.py                   #   Publication-quality plots
│   ├── demo.py                        #   4-arm dose-finding demo
│   └── real_drug_test.py              #   Random real drug simulation
├── data/
│   ├── raw/                           # 456 FDA PDFs (not in git)
│   ├── curated/                       # Cleaned datasets (v0.1 → v11)
│   ├── digitized/                     # Auto-digitized C-t profiles
│   ├── figures/                       # Extracted figure images
│   ├── llm_extracted/                 # LLM-extracted PK tuples + predictions
│   ├── splits/                        # Train/test split definitions
│   ├── validation/                    # Holdout definition + 50 result JSONs
│   └── trial_sim_plots/               # Simulator output plots
├── tests/
│   └── test_simulator.py              # 78 unit tests
├── evaluation/
│   └── metrics.py                     # AAFE, fold-accuracy metrics
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Key dependencies: RDKit, XGBoost, scikit-learn, PyMuPDF.

For auto-digitization (optional): `pip install easyocr opencv-python-headless`

For LLM predictions (optional): Claude API access via `anthropic` package (included in requirements).

FDA PDFs are not included in the repository (too large). To reproduce from scratch, run `pipeline/scraper.py` with access to drugs@FDA.

## Related Work

- **Sisyphus PBPK Platform**: [github.com/jam-sudo/Sisyphus](https://github.com/jam-sudo/Sisyphus) — physics-based PK prediction (AAFE 2.283)
- Jia et al. (2025) J Med Chem — 800 digitized C-t profiles, PBPK hybrid
- Pillai et al. (2024) Clin Transl Sci — Sanofi ML framework (2-fold 40-60%)

## License

MIT

## Author

Jae Min Yoon — jaemin6013@gmail.com
