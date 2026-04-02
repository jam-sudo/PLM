# PLM Scale-Up: 50-100 NDA C-t Profile Extraction

Read CLAUDE.md.

## 배경

> **네트워크 전제조건:** Claude Code allowed domains에 아래가 포함되어야 한다:
> `accessdata.fda.gov`, `api.fda.gov`, `pubchem.ncbi.nlm.nih.gov`
> Feasibility test에서 이미 확인됐으면 skip.
> 차단되어 있으면 사용자에게 설정 변경을 요청하라.

Feasibility test 통과:

- 11 PDFs, 1,058 figures, Cmax error 3.5%, 4/5 metadata
- 파이프라인 작동 확인

이번: 50-100 NDA로 스케일업. 목표 500-1,500 usable C-t profiles.

> **이 단계의 핵심 위험은 accuracy가 아니라 consistency다.**
> 약물마다 dose unit, concentration unit, formulation 표기가 다르다.
> Normalization 실패 하나가 전체 dataset을 오염시킨다.

---

## Phase 0: NDA Target List 구성

### 0A: 2010-2025 승인 oral small molecule NDA 목록

drugs@FDA API로 최근 승인 NDA 목록 추출:

```python
import requests

results = []
for skip in range(0, 500, 100):
    resp = requests.get(
        "https://api.fda.gov/drug/drugsfda.json",
        params={
            "search": 'submissions.submission_type:"ORIG"',
            "limit": 100,
            "skip": skip,
        },
        timeout=30,
    )
    if resp.ok:
        results.extend(resp.json().get("results", []))
```

필터 조건:

- Small molecule만 (biologics 제외)
- Oral route 우선 (IV도 포함하되 별도 표기)
- Clinical Pharmacology review PDF 존재
- 2010년 이후 승인 (PDF 품질 + digital native 확률 높음)

> API에서 route/molecule type 필터가 안 되면
> 전체 목록을 받아서 Python에서 필터.

**보고:**

- 전체 NDA 수
- 필터 통과 NDA 수
- 최종 target 100개 목록 (NDA number, drug name, approval year)

### 0B: 이전 feasibility test 약물과 중복 제거

이미 처리한 11개 NDA는 다운로드를 skip.
단, feasibility에서 digitize한 C-t profiles는 최종 dataset에 포함한다.
`data/digitized/feasibility_samples/`에서 가져와서
동일한 normalization pipeline (Phase 4)을 거쳐 통합.
Feasibility 데이터가 Phase 4의 unit/formulation normalization을
거치지 않은 상태면 재처리 필수.

---

## Phase 1: Batch PDF Download

target NDA 목록에서 Clinical Pharmacology review PDFs 다운로드.

```python
for nda in target_ndas:
    download_clinical_pharm_review(nda, outdir="data/raw/")
    time.sleep(2)  # Rate limiting
```

> 일부 NDA는 review PDF가 여러 개 (original + supplements).
> Original NDA의 Clinical Pharmacology review만 다운로드.
> Supplement (sNDA)의 추가 PK studies는 Phase 2에서 별도 처리.

> PDF 다운로드 실패 시 3회 retry 후 skip. 실패 목록 기록.

**보고:**

- 다운로드 성공 / 시도
- 총 PDF 크기 (GB)
- 실패한 NDA 목록과 이유

**Gate:**
50개 이상 다운로드 성공 → Phase 2 진행
미달 → NDA 목록 확장 또는 수동 보완

---

## Phase 2: Batch Figure Extraction + C-t Classification

### 2A: Figure 추출

모든 PDFs에서 figure 추출.

> Feasibility test에서 검증된 pipeline (`pipeline/figure_extractor.py`) 사용.
> 새로 구현하지 마라. 기존 코드를 batch wrapper로 감싸라.

> PDF당 figure 수가 매우 다를 수 있다 (10-200개).
> 총 figure 수 예상: 50 PDFs × ~100 figures = ~5,000 figures.

### 2B: C-t Profile 분류

Heuristic classifier (feasibility에서 검증됨)로 1차 분류.

**보고:**

- 총 figure 수
- C-t profile로 분류된 수
- NDA당 평균 C-t figure 수

> 같은 NDA에서 여러 PK study가 나온다:
>
> - Single dose escalation (100mg, 200mg, 400mg)
> - Multiple dose steady state
> - Food effect study (fasted vs fed)
> - Renal/hepatic impairment study
> - Drug-drug interaction study
>
> 각각 별도 C-t profile이다. study_type을 metadata에 기록:
> `"single_dose"`, `"multiple_dose"`, `"food_effect"`, `"organ_impairment"`, `"ddi"`, `"other"`

DDI study의 C-t는 제외하거나 별도 처리 (perpetrator drug이 PK를 바꿈).
Organ impairment study는 `population="renal_impairment"` 등으로 태깅.

---

## Phase 3: C-t Digitization (Batch)

C-t로 분류된 figures에서 data points 추출.

> Scale-up에서는 Claude Code가 1,000+ figures를 한 세션에 볼 수 없다.
> 하지만 C-t로 분류된 figures는 전체의 10-20% (feasibility 기준).
> 예상 C-t figures: 50 PDFs × ~100 figs × 15% = ~750개.

접근법: Claude Code가 배치로 처리.

- 한 번에 20개 figure를 view + digitize
- JSON으로 저장 후 다음 배치
- 여러 세션에 걸쳐 진행 가능
- 750개 / 20개 배치 = ~38 배치

C-t figures가 200개 이하면 Claude Code가 직접 처리 가능.
200개 초과면 programmatic fallback 필요:
OpenCV + pytesseract로 axis/data point detection.
이 fallback pipeline은 별도 구현 (`pipeline/auto_digitizer.py`).

> Claude API 사용 금지. Claude Code가 직접 처리한다.

### 3B: Digitization Quality Assurance

> **Multi-curve figures 주의:**
> 하나의 C-t figure에 여러 curve가 있을 수 있다:
>
> - 같은 약물의 다른 dose (100mg, 200mg, 400mg)
> - Parent drug + metabolite
> - Mean ± SD (mean만 추출, SD는 metadata)
> - Fasted vs fed
> - Different formulations
>
> 각 curve를 별도 profile로 저장.
> Metabolite curves는 label에 "metabolite"가 있으면 제외.
> Mean+SD figure에서는 mean curve만 추출.
> 어떤 curve인지 불명확하면 FLAG.

자동 추출된 모든 C-t profile에 대해:

**QC Check 1: Timepoint 수**

- `data_points < 4` → REJECT
- `data_points >= 4 and < 8` → FLAG (manual review)
- `data_points >= 8` → PASS

**QC Check 2: Concentration range**

- 모든 concentration == 0 → REJECT
- `max(concentration) / min(nonzero concentration) > 10,000` → FLAG
- negative concentration → REJECT

**QC Check 3: Time range**

- `max(time) < 1h` → FLAG (might be IV bolus initial phase only)
- `min(time) > 2h` → FLAG (might be missing absorption phase)

**QC Check 4: Monotonic elimination**

- Cmax 이후 concentration이 단조감소하지 않으면 → FLAG
- 다만 multiple peaks (enterohepatic cycling) 가능하므로 REJECT 아님

**QC Check 5: Duplicate detection**

- 같은 NDA에서 동일한 curve가 여러 figure에 나올 수 있다
- Pearson r > 0.99인 curves → 형태 중복으로 FLAG
- 추가: r > 0.99이지만 Cmax 비율이 2배 이상이면
  → 같은 curve인데 unit이 다른 것. Unit error FLAG (QC Check와 별도).

**보고:**

- PASS / FLAG / REJECT 비율
- FLAG된 profiles의 대표 사례 5개

---

## Phase 4: Unit Normalization ★ CRITICAL ★

이 Phase가 전체 pipeline에서 가장 중요하다.
Unit 오류 하나가 1,000배 concentration 차이를 만든다.

### 4A: Concentration Unit Harmonization

FDA reviews에서 사용되는 concentration units:
`ng/mL`, `μg/mL` (= `mg/L`), `mg/L`, `μg/L` (= `ng/mL`),
`nmol/L`, `μmol/L`, `mg/dL`, `ng/dL`, `μg/dL`, `pg/mL`

Target unit: `ng/mL` (가장 common)

Conversion table:

```python
UNIT_TO_NGML = {
    "ng/mL": 1.0,
    "ng/ml": 1.0,
    "μg/mL": 1000.0,
    "ug/mL": 1000.0,
    "µg/mL": 1000.0,
    "mcg/mL": 1000.0,
    "mg/L": 1000.0,
    "mg/l": 1000.0,
    "μg/L": 1.0,
    "ug/L": 1.0,
    "pg/mL": 0.001,
    "mg/dL": 10000.0,
    "μg/dL": 10.0,
    "ng/dL": 0.01,
    "nmol/L": None,   # MW 필요
    "μmol/L": None,   # MW 필요
    "umol/L": None,    # MW 필요
}
```

> `nmol/L`, `μmol/L`은 molecular weight가 필요.
> Drug name → MW lookup (PubChem API).
> conversion = nmol/L × MW / 1000 = ng/mL

> μ vs µ vs u: Unicode 차이. 전부 normalize.
> "mcg" = "μg" = "ug" = "µg"

> 일부 figure는 Y축에 unit이 없거나 읽기 어려울 수 있다.
> Caption에서 unit을 추출. Caption에도 없으면 FLAG.

### 4B: Dose Unit Harmonization

Target unit: `mg`

```python
DOSE_TO_MG = {
    "mg": 1.0,
    "g": 1000.0,
    "μg": 0.001,
    "mcg": 0.001,
    "mg/kg": None,     # body weight 필요 (70kg default)
    "μg/kg": None,     # body weight 필요
}
```

> `mg/kg` dosing인 경우:
> FDA healthy volunteer study는 대부분 70kg 가정.
> `dose_mg = dose_mg_per_kg × 70`
> 이 가정을 metadata에 기록.

> 일부 약물은 body surface area (`mg/m²`) dosing.
> BSA = 1.73 m² (standard adult) 가정.
> `dose_mg = dose_mg_per_m2 × 1.73`
> 이 약물은 주로 oncology → oral small molecule에서는 드묾.

### 4C: Formulation Normalization

FDA reviews에서 formulation 표기가 일관되지 않다:

```python
FORMULATION_MAP = {
    # Immediate release
    "tablet": "IR_tablet",
    "film-coated tablet": "IR_tablet",
    "capsule": "IR_capsule",
    "hard gelatin capsule": "IR_capsule",
    "soft gelatin capsule": "IR_capsule_soft",
    "oral solution": "solution",
    "oral suspension": "suspension",
    "syrup": "solution",
    "powder for oral solution": "solution",
    
    # Extended release
    "extended-release tablet": "ER_tablet",
    "extended release tablet": "ER_tablet",
    "ER tablet": "ER_tablet",
    "XR tablet": "ER_tablet",
    "XL tablet": "ER_tablet",
    "SR tablet": "ER_tablet",
    "controlled-release": "ER_tablet",
    "modified-release": "ER_tablet",
    "extended-release capsule": "ER_capsule",
    
    # Injectable
    "IV bolus": "IV_bolus",
    "IV infusion": "IV_infusion",
    "intravenous": "IV_infusion",
    "intramuscular": "IM_injection",
    "subcutaneous": "SC_injection",
    
    # Other
    "sublingual tablet": "sublingual",
    "transdermal patch": "transdermal",
    "oral disintegrating tablet": "ODT",
}
```

> "tablet"만 적혀 있고 IR/ER 구분이 없으면:
> Drug name으로 lookup — 해당 약물의 NDA가 IR인지 ER인지.
> 모호하면 `"tablet_unspecified"`로 태깅. REJECT 아님.

### 4D: Route Normalization

```python
ROUTE_MAP = {
    "oral": "oral",
    "po": "oral",
    "by mouth": "oral",
    "intravenous": "IV",
    "iv": "IV",
    "i.v.": "IV",
    "intramuscular": "IM",
    "im": "IM",
    "i.m.": "IM",
    "subcutaneous": "SC",
    "sc": "SC",
    "s.c.": "SC",
    "sublingual": "sublingual",
    "sl": "sublingual",
    "transdermal": "transdermal",
    "topical": "topical",
    "rectal": "rectal",
    "inhaled": "inhaled",
    "intranasal": "intranasal",
}
```

### 4E: Food Effect Normalization

```python
FOOD_MAP = {
    "fasted": "fasted",
    "fasting": "fasted",
    "empty stomach": "fasted",
    "fed": "fed",
    "with food": "fed",
    "high-fat meal": "fed_highfat",
    "high fat meal": "fed_highfat",
    "standard meal": "fed_standard",
    "light meal": "fed_light",
}
```

> Food effect가 명시되지 않은 경우: `"not_specified"`

### 4F: Cross-Validation of Units

Unit 오류를 잡는 sanity checks:

**Check 1: Cmax plausibility**

- Oral small molecule: 대부분 Cmax 1-100,000 ng/mL
- `Cmax < 0.01 ng/mL` → unit 의심 (pg/mL을 ng/mL로 잘못?)
- `Cmax > 1,000,000 ng/mL` → unit 의심 (μg/mL을 ng/mL로 잘못?)

**Check 2: Dose-normalized Cmax plausibility**

- `Cmax/dose`가 0.001-1000 ng/mL/mg 범위 밖이면 → FLAG
- 극단적으로 potent한 약물 (μg dose)도 있으므로 REJECT 아님

**Check 3: Known drug cross-reference**

- PK-DB 또는 DrugBank에서 known Cmax가 있는 약물과 비교
- 추출된 Cmax가 known value의 5배 이상 차이 → FLAG

**Check 4: Same drug, multiple figures consistency**

- 같은 NDA에서 같은 dose의 C-t가 여러 figure에 있으면
- Cmax가 2배 이상 차이 → 하나는 unit이 틀렸을 가능성

**보고:**

- Unit conversion이 적용된 profile 수
- MW lookup이 필요했던 profile 수 (nmol/L 등)
- Sanity check FLAG 수와 대표 사례
- 최종 PASS profile 수

---

## Phase 5: Dose-Normalized C-t Interpolation

모든 QC-passed profiles를 standard timepoint grid로 interpolation.

Standard grid: `[0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24]` hours

```python
from scipy.interpolate import interp1d
import numpy as np

def interpolate_ct(timepoints, concentrations, grid):
    # Remove C=0 points before log-linear interpolation
    nonzero_mask = concentrations > 0
    if np.sum(nonzero_mask) < 2:
        return np.full(len(grid), np.nan)
    
    t_nz = np.array(timepoints)[nonzero_mask]
    c_nz = np.array(concentrations)[nonzero_mask]
    log_conc = np.log10(c_nz)
    
    # Only interpolate within observed nonzero time range
    mask = (grid >= t_nz.min()) & (grid <= t_nz.max())
    
    f = interp1d(t_nz, log_conc, kind='linear',
                 bounds_error=False, fill_value=np.nan)
    
    interpolated = np.full(len(grid), np.nan)
    interpolated[mask] = 10 ** f(grid[mask])
    
    return interpolated
```

> Extrapolation 금지. Observed time range 밖은 NaN.
> `C(0) = 0` for oral dosing: `log10(0/dose)` = undefined.
> → `t=0`의 `log_c_over_dose = NaN` (not -inf).
> → Model training에서 NaN timepoints는 loss 계산에서 mask.
> 24h 이후 data가 있으면 24h까지만 사용 (grid 범위).

> IV bolus는 다른 grid 필요:
> `[0, 0.033, 0.083, 0.167, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 24]`
> 초기 distribution phase를 포착하기 위해 더 촘촘한 early timepoints.
> Route가 "IV"이면 IV grid 사용.

> Route 정보는 Phase 4 (metadata extraction)에서 결정된다.
> Interpolation (Phase 5)은 반드시 Phase 4 완료 후 실행.
> Route가 `"not_specified"`이면 oral grid를 default로 사용.

### Dose normalization

target = `log10(C(t) / dose_mg)` at each grid point.

> 같은 약물의 multiple doses가 있으면 각각 별도 training sample.
> Linear PK 가정: C/dose가 dose-independent.
> Nonlinear PK drugs는 이 가정이 깨진다 — metadata에 flag.

---

## Phase 6: Final Dataset Assembly

### 6A: Output Format

```json
{
  "version": "0.1.0",
  "n_profiles": "???",
  "n_unique_drugs": "???",
  "profiles": [
    {
      "id": "PLM_00001",
      "drug_name": "apixaban",
      "smiles": "COc1ccc(-n2nc(C(=O)N3CCN(C(=O)c4ccc(-n5cccn5)cc4)CC3)c3ccccc32)cc1",
      "mw": 459.5,
      "dose_mg": 5.0,
      "route": "oral",
      "formulation": "IR_tablet",
      "food_effect": "fasted",
      "population": "healthy",
      "study_type": "single_dose",
      "n_subjects": 24,
      "source_nda": "NDA_202155",
      "source_page": 42,
      "concentration_unit_original": "ng/mL",
      "concentration_unit_normalized": "ng/mL",
      "unit_conversion_factor": 1.0,
      "timepoints_h": [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24],
      "concentrations_ngml": [0, 12, 85, 145, 170, 162, 130, 98, 52, 28, 8, 2, 0.5],
      "log_c_over_dose": [null, -1.62, -0.77, -0.54, -0.47, -0.49, -0.59, -0.71, -0.98, -1.25, -1.80, -2.40, -3.00],
      "cmax_ngml": 170,
      "tmax_h": 1.5,
      "auc_ngml_h": 1200,
      "qc_status": "pass",
      "qc_flags": []
    }
  ]
}
```

저장: `data/curated/plm_dataset_v0.1.json`

### 6B: Dataset Statistics

**보고:**

- 총 profile 수 (QC pass)
- Unique drug 수
- Route 분포: oral / IV / other
- Formulation 분포
- Food effect 분포
- Dose range: min, median, max
- Cmax range: min, median, max (ng/mL)
- Year of approval 분포
- Mean timepoints per profile

### 6C: SMILES Validation

모든 drug의 SMILES를 RDKit으로 검증:

```python
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
assert mol is not None, f"Invalid SMILES for {drug_name}"
```

> PubChem에서 가져온 SMILES가 RDKit에서 parse 실패할 수 있다.
> 실패 시 PubChem canonical SMILES 대신 isomeric SMILES 시도.
> 그래도 실패하면 해당 profile은 SMILES 없이 저장 (model training 제외).

> **Salt forms 주의:** "Metformin HCl" vs "Metformin"은 다른 SMILES.
> PubChem에서 "." (dot)이 포함된 SMILES는 salt/mixture.
> → dot 앞뒤 중 더 큰 fragment (atom 수 기준)가 active moiety.
> → 또는 drug name에서 "HCl", "hydrochloride", "sodium", "potassium",
>   "mesylate", "fumarate", "tartrate", "maleate", "besylate" strip 후 재검색.
> → RDKit: `Chem.SaltRemover`로 자동 처리 가능.

---

## Phase 7: Sisyphus Overlap Analysis

PLM dataset과 Sisyphus MMPK dataset의 overlap 확인.

Sisyphus MMPK drugs의 SMILES와 PLM drugs의 SMILES를
InChIKey로 매칭.

> Sisyphus MMPK 데이터 위치:
> Sisyphus repo를 clone하거나, 이미 local에 있으면:
> `data/training/` 아래의 MMPK CSV/JSON에서 SMILES + Cmax 추출.
> 정확한 파일명은 Sisyphus CLAUDE.md를 참조.
> Sisyphus repo가 접근 불가하면 이 Phase를 skip.

**보고:**

- Overlap drugs 수
- PLM-only drugs 수 (new chemical entities)
- Sisyphus-only drugs 수
- Overlap drugs에서 Cmax 비교: PLM Cmax vs MMPK Cmax
  (같은 약물, 비슷한 dose에서 일치하는지 = data quality check)

> Overlap drugs에서 PLM Cmax와 MMPK Cmax가 2배 이상 차이나면
> unit conversion 오류 또는 formulation 차이.
> 원인을 drug-by-drug로 조사.

---

## 출력

| File | Description |
|------|-------------|
| `pipeline/scraper.py` | 업데이트 (batch download) |
| `pipeline/figure_extractor.py` | 업데이트 (batch processing) |
| `pipeline/digitizer.py` | 자동 digitization (신규/업데이트) |
| `pipeline/caption_parser.py` | metadata extraction (신규/업데이트) |
| `pipeline/normalizer.py` | unit/formulation/route normalization (신규) |
| `pipeline/quality_filter.py` | QC checks (신규) |
| `pipeline/interpolator.py` | C-t interpolation to standard grid (신규) |
| `data/curated/plm_dataset_v0.1.json` | 최종 dataset |
| `data/curated/dataset_statistics.json` | 통계 |
| `data/curated/sisyphus_overlap.json` | Sisyphus overlap 분석 |

Git commit and push after each Phase completion.

---

## 금지 사항

- FDA 서버에 요청 간 2초 미만 간격 금지
- Unit conversion 없이 raw concentration을 dataset에 넣지 마라
- nmol/L → ng/mL 변환 시 MW 없이 변환하지 마라
- Interpolation에서 observed time range 밖으로 extrapolation 금지
- 같은 drug의 IR과 ER profiles를 하나로 합치지 마라 (별도 sample)
- IV와 oral C-t profiles를 같은 timepoint grid로 강제하지 마라
- QC FLAG된 profiles를 자동 REJECT하지 마라 (FLAG만, 수동 review 대기)
- Cmax plausibility check에서 REJECT 기준을 너무 tight하게 잡지 마라
  (potent drugs는 Cmax < 1 ng/mL 가능)
- PubChem API에 초당 5회 이상 요청 금지 (rate limit)
- Biologics를 dataset에 포함하지 마라.
  필터: drugs@FDA API에서 product_type이 "BLA"이면 제외.
  추가로 MW > 1500 Da이면 제외 (peptide/protein).
  MW 1000-1500 Da는 허용 (일부 kinase inhibitor가 이 범위).
- PDF를 git에 commit하지 마라
- Figure images를 git에 commit하지 마라
- `log10(0)`을 계산하지 마라. `C(0)=0`이면 `log_c_over_dose[0] = null` (JSON) / `NaN` (Python)
