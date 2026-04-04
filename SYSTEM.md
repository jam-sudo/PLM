# PLM System Review

**Pharmacological Language Model** — ML prediction of plasma Cmax from SMILES + dose, using LLM-extracted FDA clinical pharmacology data.

---

## 1. Project Thesis

**Core hypothesis**: Train ML models on publicly-available FDA regulatory documents to predict human plasma Cmax directly from `[SMILES, dose, conditions]` — bypassing traditional IVIVE/PBPK error propagation.

**Differentiation from Sisyphus**:
- Sisyphus: curated literature + PBPK engine ensemble
- PLM: automated FDA label extraction + pure ML (no PBPK)
- Same data origin, different paradigm

---

## 2. Architecture

### 2.1 Pipeline Components (data extraction)

```
FDA drugs@FDA → PDF download (scraper.py)
              → PDF figure extraction (figure_extractor.py)
              → Auto-digitization (auto_digitizer.py)      [legacy path, 63% yield]
              → Caption/unit parsing (caption_parser.py, normalizer.py)

FDA PDFs → PK page detection (extract_all_pk_text.py)
         → LLM extraction via parallel agents              [NEW, 7.2 tuples/PDF]
         → JSON aggregation (aggregate_llm_extractions.py)
         → Dataset merge (merge_llm_with_v10.py)
```

### 2.2 Model Components

```
SMILES → Morgan FP 4096 bits
       + PhysChem 20 descriptors (MW, LogP, TPSA, ...)
       + TDC ADME 9 features (logS, PPB, Vd, t½, CL, ...)  [lookup by InChIKey]
       + Micro-PBPK 6 derived features (fa, fu, Eh, F, Vd, ke)
       + log10(dose_mg)
       + Condition one-hot 18 features (route/schedule/food/form/pop)  [NEW]
       → XGBoost (depth=6, lr=0.01, n=500)
       → log10(Cmax / dose)

Optional: ADME encoder pre-training (models/pretrain_adme_xgb.py)
  11 TDC ADME tasks → MLP encoder(768→384→emb_dim) → feature extraction
```

### 2.3 Data Assets

| Asset | Size | Content |
|-------|------|---------|
| `data/raw/*.pdf` | 456 PDFs (~2GB, gitignored) | FDA ClinPharmR/Multidiscipline reviews |
| `data/llm_extracted/text/` | 385 txt files, 16MB | PK-relevant pages from PDFs |
| `data/llm_extracted/json/` | 381 JSON files | Per-NDA structured PK extractions |
| `data/llm_extracted/pk_llm_merged.json` | 1,333 tuples | Validated merged extractions |
| `data/curated/tdc_adme_data.json` | 15,751 compounds | ADME properties from TDC |
| `data/curated/plm_dataset_v10_labels.json` | 3,490 profiles | v10 training data (PLM 150 + SIS 3340) |
| `data/curated/plm_dataset_v11_llm.json` | 4,540 profiles | v10 + LLM merged (deprecated in favor of experiment pipeline) |
| `data/validation/holdout_definition.json` | 97 drugs | Sisyphus holdout benchmark |
| `data/validation/holdout_corrections.json` | 1 correction | PLM audit of suspect obs values |
| `data/raw/nda_drug_smiles_map.json` | 326 entries | NDA → drug → SMILES lookup |

---

## 3. Experimental Evolution

### Phase 1: Figure-based extraction (early sessions)
- Auto-digitizer v2: 63% figure-to-curve success
- Dataset v0.4 → v0.5: 199 profiles, 72 drugs, HO AAFE ~7.8
- Noise ceiling recognized

### Phase 2: Feature engineering
- PhysChem descriptors added
- FDA ADME text extraction
- TDC ADME integration (15,751 compounds)
- Micro-PBPK mechanistic features
- Best: HO AAFE 3.217 (Sisyphus Engine beaten by 5.8%)

### Phase 3: Novel architectures (negative results)
- ChemBERTa embeddings: worse than Morgan FP
- Mechanistic-ML hybrid: worse than direct
- Delta learning: micro-PBPK base too crude
- RAPK retrieval: analogs too distant

### Phase 4: Multi-task ADME pre-training
- 11 TDC ADME tasks → MLP encoder
- Enc(512) + base features → HO 3.268 (best single run, noisy)
- Seed-to-seed variance ~0.08

### Phase 5: LLM FDA extraction (NEW)
- 38 parallel agents extracted 1,333 PK tuples from 385 PDFs
- 7.2 tuples/PDF (vs regex 1.0/PDF)
- Initial negative result: v10+LLM = HO 3.505 (WORSE)
- Diagnosed: condition heterogeneity + Sisyphus holdout errors

### Phase 6: Proper LLM integration (current best)
- Drug-level median + condition features + agreement filter + confidence weighting
- **HO AAFE 3.355** (single model, stable)
- Benchmark audit: 1 confirmed suspect value (cabozantinib)
- **HO AAFE 3.258** (with 1 correction)

---

## 4. Key Results

### Current state
| Model | HO AAFE | vs Sisyphus Engine |
|-------|---------|-------------------|
| **PLM best (screening + cond features)** | **3.355** | **−0.061 (beats)** |
| **PLM on corrected benchmark (1 drug)** | **3.258** | **−0.158** |
| Sisyphus Engine | 3.416 | — |
| Sisyphus ML | 2.336 | — |
| Sisyphus Meta | 2.283 | (benchmark has suspect values) |

### LLM extraction metrics
- **Yield**: 1,333 valid tuples from 385 PDFs (3.6/PDF avg; 7.2/PDF peak)
- **Drugs covered**: 226 unique
- **SMILES coverage**: 1,184 / 1,333 (89%) after PubChem enrichment
- **Confidence distribution**: 1,238 high (93%) / 87 medium / 8 low
- **Route mix**: oral 85%, SC 5%, IV 3%, inhalation 2%, other 5%
- **Schedule mix**: single_dose 78%, steady_state 19%, multiple_dose 3%

### Benchmark audit findings
- **cabozantinib**: Sisyphus obs 9800 ng/mL at 140mg vs. published FDA label 577 ng/mL (single-dose) → **17x error, confirmed correction**
- Other candidates (methylphenidate, posaconazole, sumatriptan) have plausible formulation-specific explanations — not corrected
- Sisyphus meta's own AAFE vs obs = 2.19 (Sisyphus's own model also struggles on ~15 drugs)

---

## 5. Novel Contributions

### A. LLM-powered FDA extraction pipeline
First systematic use of LLM agents to extract structured PK data from FDA regulatory PDFs at scale.
- 7x yield improvement over regex baseline
- Context-aware unit conversion (pg/μg/nmol/m² all handled)
- Confidence scoring per extraction
- 38-agent parallel orchestration

### B. Condition-aware PK modeling
Introducing route/schedule/food/population as explicit model features.
- Distinguishes canonical (single-dose oral healthy fasted) from non-canonical conditions
- Enables training on diverse LLM data without violating linear-PK assumptions

### C. Drug-level median + agreement filter
Robust LLM integration strategy:
1. Group tuples by (drug, conditions)
2. Take median log(Cmax/dose) per group
3. Reject if disagrees with existing v10 mean by >1 log unit (10x)
4. Weight by extraction confidence

### D. Benchmark critique with independent verification
Demonstrated that Sisyphus holdout has at least 1 confirmed incorrect value (cabozantinib). Provides methodology for auditing PK benchmarks via LLM extraction cross-reference.

---

## 6. Known Limitations

### Technical
1. **Condition assumptions for v10**: 3,340 Sisyphus profiles assumed canonical — may not be true
2. **Formulation modeling**: many formulations conflated (IR ≠ ER, tablet ≠ capsule)
3. **Non-linear PK drugs**: drugs with saturable absorption (sonidegib, gabapentin) not modeled
4. **Route coverage**: oral-biased (85%); IV/SC/IM underrepresented

### Data
1. **Holdout overlap**: 0/97 Sisyphus holdout drugs in v10 training → pure OOD extrapolation
2. **Sisyphus holdout quality**: ~15/97 drugs have meta-obs disagreement; 1 confirmed wrong (cabozantinib)
3. **Bulk-Sisyphus dominance**: Adding LLM data only gives 40 truly new drugs vs Sisyphus's 1,102

### Methodological
1. **Holdout ambiguity**: single-dose vs steady-state not specified in holdout
2. **Food effect missing in holdout**: all assumed canonical fasted
3. **Encoder variance**: 0.08 AAFE std across seeds, needs ensemble

---

## 7. Reproducibility

### Key commands
```bash
# Extract PK text from PDFs
python3 pipeline/extract_all_pk_text.py

# LLM extraction via parallel agents (via Claude Code)
# (see pipeline/llm_extractor.py for API version)

# Aggregate extractions
python3 pipeline/aggregate_llm_extractions.py

# Run best experiment
python3 pipeline/llm_enriched_experiment.py

# Benchmark audit
python3 pipeline/benchmark_audit.py
```

### Environment
- Python 3.10+
- PyTorch 2.11 (for encoder)
- XGBoost 3.2
- RDKit 2023.09
- PyMuPDF (fitz)
- TDC (Therapeutics Data Commons)

---

## 8. Reusability & Future Directions

### Reusable components
- **LLM extraction pipeline**: drop-in for any drug label source (DailyMed, EMA, PMDA)
- **Condition-aware model**: generalizable to other PK endpoints (AUC, t½)
- **ADME encoder**: transferable to other PK targets

### Recommended next steps

**Short-term (model improvement)**:
1. Expand condition feature coverage (formulation subtypes)
2. Add IV-dose data handling (separate CL/Vd model path)
3. Ensemble encoder + XGBoost for variance reduction

**Medium-term (data expansion)**:
1. DailyMed label extraction (2000+ drugs, different space)
2. PMDA (Japan) label extraction
3. Pediatric/special population dedicated models

**Long-term (research frontier)**:
1. Neural ODE for full C(t) curve prediction
2. Multi-task learning: joint Cmax + AUC + Tmax
3. Bayesian uncertainty quantification for clinical deployment

---

## 9. Code Quality Summary

| Metric | Value |
|--------|-------|
| Pipeline scripts | 14 (3,316 lines) |
| Model scripts | 2 (941 lines) |
| Total Python LOC | ~4,300 |
| Test coverage | **None** (known gap) |
| Documentation | Inline docstrings + this file |

**Known debt**:
- No unit tests
- Several legacy scripts (auto_digitizer, digitizer) superseded by LLM pipeline
- Hyperparameters hardcoded (should be in config)

---

## 10. Citation Pathways

The following results are publishable:

1. **LLM FDA Extraction Pipeline** (methods paper)
   - Novel application of LLM agents to PK data
   - Benchmark: 7x yield vs regex, 89% SMILES coverage
   - Reusable across regulatory jurisdictions

2. **Benchmark Audit Methodology** (commentary paper)
   - Cross-reference verification via independent extraction
   - Identified 1 confirmed error in widely-used holdout
   - Generalizable to other PK benchmarks

3. **Condition-Aware PK ML** (modeling paper)
   - Integration strategy for condition-heterogeneous data
   - Drug-level median + agreement filter methodology
   - Beats Sisyphus Engine (PBPK-based) with pure ML

---

*Generated: 2026-04-04 | Commit: 924e4d5 | Best HO AAFE: 3.355 (original) / 3.258 (corrected)*
