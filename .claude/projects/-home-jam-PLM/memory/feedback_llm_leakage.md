---
name: LLM Predictions Are Data Leakage
description: User explicitly flagged that LLM Cmax predictions (AAFE 2.1) are data leakage — never cite as PLM model performance
type: feedback
---

LLM "predictions" of Cmax from drug name+dose are memorized recall from training corpus, not structure-based prediction. User warned about this and corrected CLAUDE.md accordingly.

**Why:** Holdout drugs are all marketed compounds present in LLM training data (FDA labels, medical literature). For novel compounds with no published PK, LLM cannot predict.

**How to apply:** Always separate PLM model performance (AAFE 3.355, structure-based) from LLM recall (AAFE 2.1, name-based). LLM is a data extraction tool, not a predictor. Never present ensemble of XGBoost+LLM as PLM performance.
