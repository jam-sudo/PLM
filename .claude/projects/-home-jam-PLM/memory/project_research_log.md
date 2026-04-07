---
name: PLM Research Log Location
description: All PLM experiment results (successes and failures) are documented in docs/RESEARCH_LOG.md — check before proposing already-tried approaches
type: reference
---

PLM research log lives at `docs/RESEARCH_LOG.md`. Contains 6 successes (S1-S6), 6 failures (F1-F6), and 2 informational entries (I1-I2 — LLM data leakage).

**Key failures to avoid re-proposing:**
- F1: DrugBank synthetic C-t profiles (noisy synthetic data hurts model)
- F2: MolFormer embeddings (PK ≠ SAR, fancy embeddings don't help)
- F3: Retrieval-augmented delta (Tanimoto similarity ≠ PK similarity)
- F4: Asymmetric loss (not enough data to benefit)
- F5: Isotonic calibration (overfits CV, hurts HO)
- F6: PK-DB API (data endpoints return empty)

**Current best holdout AAFE: 3.355** (XGBoost, novel_results baseline)
**Sisyphus target: 2.283**

Cross-linked from: [CLAUDE.md](../../../CLAUDE.md), [docs/RESEARCH_LOG.md](../../../docs/RESEARCH_LOG.md)
