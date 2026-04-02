# PLM: Pharmacological Language Model

Predicting human plasma concentration-time profiles directly from molecular structure, bypassing IVIVE error propagation chains.

## Concept

Traditional PBPK: `SMILES → CLint → fup → Peff → Kp → IVIVE → ODE → C(t) → Cmax`
- 7 sequential models, each with prediction error
- Errors propagate multiplicatively through the chain

PLM: `[SMILES, dose, route, formulation] → C(t) → Cmax, AUC, tmax, t1/2`
- Single model, no intermediate ADME parameters
- Error propagation chain eliminated

## Data Source

FDA Clinical Pharmacology & Biopharmaceutics Reviews from drugs@FDA.
~1,200 small molecule NDAs (2000-2025), each containing 3-5 concentration-time figures.
Target: 3,000-4,000 digitized C-t profiles with standardized metadata.

## Architecture (Planned)

**Phase 1 — XGBoost Multi-Output Baseline**
- Morgan FP + [dose, route, formulation, food_effect] → 13 timepoint log-concentrations
- Fast to train, works at N=3,000

**Phase 2 — Encoder-Decoder Transformer**
- SMILES tokenizer + dose/route embeddings → autoregressive C(t) generation
- Requires N > 10,000 (FDA + EMA + literature)

**Phase 3 — Sisyphus Ensemble**
- Blend physics-based C(t) (Sisyphus engine) with data-driven C(t) (PLM)
- Timepoint-level weighted averaging

## Project Structure

```
PLM/
├── README.md
├── CLAUDE.md                    # Source of truth for Claude Code
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/                     # Downloaded FDA PDFs
│   │   └── .gitkeep
│   ├── figures/                 # Extracted figure images
│   │   └── .gitkeep
│   ├── digitized/               # Digitized C-t profiles (JSON/CSV)
│   │   └── .gitkeep
│   ├── curated/                 # QC-passed, standardized profiles
│   │   └── .gitkeep
│   └── splits/                  # Train/val/test splits
│       └── .gitkeep
├── pipeline/
│   ├── __init__.py
│   ├── scraper.py               # FDA PDF download automation
│   ├── figure_extractor.py      # PDF → figure images (PyMuPDF)
│   ├── digitizer.py             # Figure → data points (ChartOCR/LLM)
│   ├── caption_parser.py        # Caption → metadata (Claude API)
│   ├── normalizer.py            # Unit harmonization, interpolation
│   └── quality_filter.py        # Automated QC checks
├── models/
│   ├── __init__.py
│   ├── xgb_multioutput.py       # Phase 1: XGBoost baseline
│   ├── transformer.py           # Phase 2: Encoder-decoder
│   └── ensemble.py              # Phase 3: Sisyphus + PLM blend
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               # AAFE, %2-fold, C-t RMSE
│   └── benchmark.py             # Holdout evaluation, comparison vs Sisyphus
├── notebooks/
│   ├── 01_feasibility_test.ipynb
│   └── .gitkeep
├── scripts/
│   └── .gitkeep
└── tests/
    └── .gitkeep
```

## Evaluation

Primary: Cmax AAFE on holdout set (drug-level time-split)
Secondary: AUC AAFE, tmax MAE, C(t) log-RMSE
Benchmark: vs Sisyphus (AAFE 2.283), vs Sanofi hierarchical ML (2-fold 40-60%)

## Related Work

- Sisyphus PBPK Platform: [github.com/jam-sudo/Sisyphus](https://github.com/jam-sudo/Sisyphus)
- Jia et al. (2025) J Med Chem — 800 digitized C-t profiles, PBPK hybrid
- Pillai et al. (2024) Clin Transl Sci — Sanofi ML framework for PK profiles

## Status

**Pre-feasibility** — Validating FDA PDF extraction pipeline.

## Author

Jae Min Yoon — jaemin6013@gmail.com
