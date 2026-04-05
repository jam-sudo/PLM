---
name: PLM Scale-Up Plan (50-100 NDA)
description: Comprehensive 7-phase plan to scale FDA C-t profile extraction from 11 to 50-100 NDAs, targeting 500-1500 usable profiles. Includes batch download, figure extraction, digitization, unit normalization, interpolation, dataset assembly, and Sisyphus overlap analysis.
type: project
---

Full plan document saved at: docs/scaleup_plan.md

**Why:** Feasibility test passed (11 PDFs, Cmax error 9.2%). Next step is scaling to build a training dataset for Phase 1 XGBoost model. Key risk is consistency (unit normalization), not accuracy.

**How to apply:** Follow phases sequentially (0→7). Commit after each phase. Use existing pipeline code as base. Critical phase is Phase 4 (unit normalization). Never skip unit conversion or MW lookup for molar units.
