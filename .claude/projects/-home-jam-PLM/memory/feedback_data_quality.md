---
name: Data Quality Over Quantity
description: Noisy synthetic data actively hurts the model — proven by DrugBank expansion experiment (AAFE 3.355 → 3.469)
type: feedback
---

Do not propose synthetic data expansion without high-quality PK parameters per drug. Generic assumptions (fixed ka, fixed dose, 1-compartment) produce noise that degrades model performance.

**Why:** DrugBank experiment (2026-04-07) showed 335 synthetic profiles made AAFE 0.11 worse. The model learns the noise pattern instead of true PK relationships.

**How to apply:** Before adding data, verify: (1) drug-specific PK params, not generic estimates, (2) realistic dose values, not reference doses, (3) correct compartmental model. If any are missing, the data will hurt more than help.
