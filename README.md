# PLM: Pharmacological Language Model

Predicting human plasma Cmax directly from molecular structure and dosing conditions, bypassing the IVIVE error propagation chain inherent in traditional PBPK approaches.

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

| Model | AAFE | 2-fold% | Evaluation | N |
|-------|------|---------|------------|---|
| PLM XGBoost (CV best) | **3.275** | 38.2% | 5-fold GroupKFold | 3,490 |
| PLM XGBoost (holdout) | **3.355** | — | 97-drug holdout | 97 |
| Sisyphus Meta | 2.283 | ~50% | 107-drug holdout | 107 |
| Sisyphus ML | 2.336 | — | 107-drug holdout | 107 |
| Sisyphus Engine | 3.416 | — | 107-drug holdout | 107 |

Gap to Sisyphus Meta: ~1.5x. Primary bottleneck: training data size and chemical space coverage.

Full experiment history (39 experiments, including failures): [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)

## Data Pipeline

456 FDA Clinical Pharmacology & Biopharmaceutics Reviews → structured PK data.

| Stage | Output | Count |
|-------|--------|-------|
| PDF download (drugs@FDA) | FDA review PDFs | 456 |
| Figure extraction (PyMuPDF) | Figure images | 14,000+ |
| Auto-digitization (EasyOCR + OpenCV) | C-t profiles | 592/927 (63.9%) |
| LLM table extraction (Claude) | PK tuples (Cmax, AUC, t1/2) | 1,333 from 226 drugs |
| Unit normalization | Standardized ng/mL | All data |
| Training set (v10 + Sisyphus) | Model-ready profiles | 3,490 |
| Holdout set | Evaluation drugs (Sisyphus-aligned) | 97 drugs |

## Model

XGBoost with drug-level GroupKFold cross-validation.

- **Features**: Morgan FP 2048-bit + log10(dose) + route/formulation/food one-hot + physicochemical descriptors + TDC ADME predictions
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

## Project Structure

```
PLM/
├── CLAUDE.md                          # Project spec (source of truth)
├── docs/
│   ├── RESEARCH_LOG.md                # All experiments: successes + failures
│   └── scaleup_plan.md               # PDF extraction scale-up plan
├── pipeline/                          # Data extraction & experiments (33 scripts)
│   ├── scraper.py                     #   FDA PDF download
│   ├── figure_extractor.py            #   PDF → figure images
│   ├── auto_digitizer.py              #   Figure → C-t data (OCR + curve tracing)
│   ├── llm_extractor.py              #   PDF text → PK table extraction (LLM)
│   ├── normalizer.py                  #   Unit normalization (ng/mL standard)
│   ├── novel_experiment.py            #   Latest XGBoost experiment
│   ├── ho_diagnostic.py               #   Holdout error decomposition
│   └── ...                            #   + 26 more experiment/evaluation scripts
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
│   └── demo.py                        #   4-arm dose-finding demo
├── data/
│   ├── raw/                           # 456 FDA PDFs (not in git)
│   ├── curated/                       # Cleaned datasets (v0.4 → v10)
│   ├── llm_extracted/                 # LLM-extracted PK tuples
│   ├── validation/                    # Holdout definition + 39 result JSONs
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

Requires Python 3.10+. Key dependencies: RDKit, XGBoost, scikit-learn, PyMuPDF, EasyOCR.

FDA PDFs are not included in the repository (too large). To reproduce from scratch, run `pipeline/scraper.py` with access to drugs@FDA.

## Related Work

- **Sisyphus PBPK Platform**: [github.com/jam-sudo/Sisyphus](https://github.com/jam-sudo/Sisyphus) — physics-based PK prediction (AAFE 2.283)
- Jia et al. (2025) J Med Chem — 800 digitized C-t profiles, PBPK hybrid
- Pillai et al. (2024) Clin Transl Sci — Sanofi ML framework (2-fold 40-60%)

## License

MIT

## Author

Jae Min Yoon — jaemin6013@gmail.com
