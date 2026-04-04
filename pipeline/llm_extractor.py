"""
PLM: LLM-Powered FDA Clinical Pharmacology Extraction

Pipeline:
  FDA ClinPharmR PDF → PK-relevant pages → Claude API → structured PK tuples

Schema (per drug):
  {
    "drug_name": str,
    "nda": str,
    "pk_tuples": [
      {
        "dose_mg": float, "route": str, "formulation": str, "food": str,
        "population": str, "n_subjects": int, "dose_schedule": str,
        "cmax_ng_ml": float, "tmax_h": float, "auc_inf_ng_h_ml": float,
        "t_half_h": float, "source_page": int, "confidence": str,
      }, ...
    ]
  }

Usage:
  python pipeline/llm_extractor.py --pdf NDA210793 --model sonnet
  python pipeline/llm_extractor.py --batch 10 --model sonnet
"""

import os
import re
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional

import fitz
import anthropic


# ─── Constants ────────────────────────────────────────────────

MODEL_IDS = {
    'sonnet': 'claude-sonnet-4-5',
    'opus':   'claude-opus-4-5',
    'haiku':  'claude-haiku-4-5',
}

PK_KEYWORDS = re.compile(
    r'\b(cmax|c_max|tmax|t_max|auc|area under|pharmacokinetic|'
    r'bioavailab|half-life|half life|t1/2|t½|'
    r'clearance|volume of distribution|absorption|elimination)\b',
    re.IGNORECASE
)

MAX_TEXT_CHARS = 80_000   # ~20K tokens input limit per PDF
MAX_PAGES = 30            # cap PK pages to prevent huge inputs


# ─── Prompt ───────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are extracting pharmacokinetic (PK) data from an FDA Clinical Pharmacology Review document.

Your task: identify all reported PK parameters with dose, population, and conditions.

## Output Schema
Return ONLY a JSON object (no prose, no markdown fences) in this exact format:
```
{
  "drug_name": "<primary drug name, generic if possible>",
  "pk_tuples": [
    {
      "dose_mg": <float, total mg dose; convert from mcg/g/mg/kg carefully>,
      "dose_original": "<verbatim dose string, e.g. '200 mg BID'>",
      "route": "<oral|IV|IM|SC|topical|sublingual|inhalation|ophthalmic|other>",
      "formulation": "<tablet|capsule|solution|suspension|IR|ER|DR|injection|unknown>",
      "food": "<fasted|fed|high_fat_meal|not_specified>",
      "population": "<healthy_adult|patient|hepatic_impaired|renal_impaired|pediatric|elderly|other>",
      "n_subjects": <int or null>,
      "dose_schedule": "<single_dose|multiple_dose|steady_state>",
      "cmax_ng_ml": <float in ng/mL, null if not reported>,
      "tmax_h": <float in hours, null if not reported>,
      "auc_inf_ng_h_ml": <float in ng·h/mL, null if not reported>,
      "auc_last_ng_h_ml": <float in ng·h/mL, null if not reported>,
      "t_half_h": <float in hours, null if not reported>,
      "cl_L_h": <float in L/h, null if not reported>,
      "vd_L": <float in L, null if not reported>,
      "source_page": <int, original page number from the document>,
      "confidence": "<high|medium|low>",
      "notes": "<brief context, e.g. 'CYP3A4 inhibitor co-admin'>"
    }
  ]
}
```

## Unit Conversion Rules
- mg/kg dose: multiply by 70 kg (average adult) if population is adult, else specify in notes
- μg/L, mcg/L → multiply by 1.0 to get ng/mL
- ng/L → divide by 1000 to get ng/mL
- μg/mL, mcg/mL → multiply by 1000 to get ng/mL
- mg/L → multiply by 1000 to get ng/mL
- AUC: h·ng/mL = ng·h/mL; convert μg·h/mL → ×1000 → ng·h/mL

## Rules
1. Extract ONLY values explicitly reported in the text (no inference, no calculation).
2. For tables with multiple dose levels, emit ONE tuple per row.
3. If Cmax is reported in a units you cannot confidently convert, set confidence="low".
4. If population/food is ambiguous, use "not_specified".
5. ONLY include tuples where at least one of (cmax_ng_ml, auc_inf_ng_h_ml, auc_last_ng_h_ml) is non-null.
6. Skip drug-drug interaction study tuples where the PK is for the interacting drug (not the primary NDA drug).
7. confidence="high" if values are in a clear PK parameter table with units; "medium" if from narrative text; "low" if ambiguous units or extensive interpretation needed.

## Input Document
Drug NDA: __NDA__

__DOCUMENT_TEXT__
"""


# ─── Functions ────────────────────────────────────────────────

def find_pk_pages(pdf_path: Path, min_hits: int = 2) -> List[Dict]:
    """Extract pages with PK-relevant content."""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        hits = len(PK_KEYWORDS.findall(text))
        if hits >= min_hits and len(text) > 200:
            pages.append({'page': i + 1, 'hits': hits, 'text': text, 'chars': len(text)})
    doc.close()
    # Sort by hits descending, take top MAX_PAGES
    pages.sort(key=lambda p: -p['hits'])
    pages = pages[:MAX_PAGES]
    # Re-sort by page number for document order
    pages.sort(key=lambda p: p['page'])
    return pages


def build_document_text(pages: List[Dict]) -> str:
    """Concatenate PK pages with page markers, truncate if needed."""
    chunks = []
    total = 0
    for p in pages:
        header = f"\n\n===== PAGE {p['page']} =====\n\n"
        needed = len(header) + len(p['text'])
        if total + needed > MAX_TEXT_CHARS:
            # Truncate the last chunk
            remaining = MAX_TEXT_CHARS - total - len(header)
            if remaining > 500:
                chunks.append(header + p['text'][:remaining])
            break
        chunks.append(header + p['text'])
        total += needed
    return "".join(chunks)


def call_claude(client, model: str, prompt: str, max_tokens: int = 4096) -> Optional[str]:
    """Call Claude API with retry logic."""
    for attempt in range(3):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract text from response
            text = "".join(block.text for block in msg.content if block.type == 'text')
            usage = {
                'input_tokens': msg.usage.input_tokens,
                'output_tokens': msg.usage.output_tokens,
            }
            return text, usage
        except anthropic.RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"  Rate limit, waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            time.sleep(5)
    return None, None


def parse_json_response(text: str) -> Optional[Dict]:
    """Parse JSON from Claude response, handling markdown fences."""
    if not text:
        return None
    # Strip markdown fences
    text = re.sub(r'^```json\s*\n?', '', text.strip())
    text = re.sub(r'\n?```\s*$', '', text)
    # Find JSON object
    start = text.find('{')
    end = text.rfind('}')
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None


def validate_tuple(t: Dict) -> bool:
    """Validate a PK tuple for physical plausibility."""
    # Must have dose
    dose = t.get('dose_mg')
    if not dose or dose <= 0 or dose > 20000:
        return False
    # Must have at least one concentration value
    if not any(t.get(k) for k in ['cmax_ng_ml', 'auc_inf_ng_h_ml', 'auc_last_ng_h_ml']):
        return False
    # Cmax sanity: 0.001 ng/mL to 1 g/mL
    if t.get('cmax_ng_ml'):
        if t['cmax_ng_ml'] <= 0 or t['cmax_ng_ml'] > 1_000_000_000:
            return False
    # Tmax sanity: 0.01 to 72 hours
    if t.get('tmax_h') and (t['tmax_h'] <= 0 or t['tmax_h'] > 72):
        return False
    # t½ sanity: 0.01 to 500 hours
    if t.get('t_half_h') and (t['t_half_h'] <= 0 or t['t_half_h'] > 500):
        return False
    return True


def extract_one_pdf(client, pdf_path: Path, model: str) -> Dict:
    """Extract PK data from one PDF."""
    nda = pdf_path.stem.replace('_ClinPharmR', '').replace('_MultidisciplineR', '')
    print(f"  [{nda}] Finding PK pages...", end=" ", flush=True)

    pages = find_pk_pages(pdf_path)
    if not pages:
        print("no PK pages found")
        return {'nda': nda, 'status': 'no_pk_pages', 'pk_tuples': []}

    doc_text = build_document_text(pages)
    print(f"{len(pages)} pages, {len(doc_text)} chars", end=" → ", flush=True)

    prompt = EXTRACTION_PROMPT.replace('__NDA__', nda).replace('__DOCUMENT_TEXT__', doc_text)
    response, usage = call_claude(client, model, prompt)

    if response is None:
        print("API failed")
        return {'nda': nda, 'status': 'api_failed', 'pk_tuples': []}

    data = parse_json_response(response)
    if data is None:
        print("JSON parse failed")
        return {'nda': nda, 'status': 'parse_failed', 'raw': response[:500], 'pk_tuples': []}

    # Validate tuples
    tuples = data.get('pk_tuples', [])
    valid_tuples = [t for t in tuples if validate_tuple(t)]
    invalid = len(tuples) - len(valid_tuples)

    print(f"{len(valid_tuples)}/{len(tuples)} tuples ({usage['input_tokens']}→{usage['output_tokens']} tokens)")

    return {
        'nda': nda,
        'status': 'ok',
        'drug_name': data.get('drug_name', ''),
        'pk_tuples': valid_tuples,
        'invalid_count': invalid,
        'usage': usage,
    }


# ─── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str, help='Single NDA to extract (e.g., NDA210793)')
    parser.add_argument('--batch', type=int, help='Batch size of PDFs to process')
    parser.add_argument('--model', default='sonnet', choices=list(MODEL_IDS.keys()))
    parser.add_argument('--skip', type=int, default=0, help='Skip first N PDFs')
    parser.add_argument('--out', default='data/llm_extracted/pk_extracted.json')
    parser.add_argument('--seed', type=int, default=42, help='For reproducible PDF selection')
    args = parser.parse_args()

    model = MODEL_IDS[args.model]
    print(f"Model: {model}")

    client = anthropic.Anthropic()

    # Select PDFs
    all_pdfs = sorted(Path('data/raw').glob('*.pdf'))
    if args.pdf:
        pdfs = [p for p in all_pdfs if args.pdf in p.name]
    elif args.batch:
        # Deterministic selection for PoC
        import random
        rng = random.Random(args.seed)
        pool = list(all_pdfs)
        rng.shuffle(pool)
        pdfs = pool[args.skip:args.skip + args.batch]
    else:
        print("Specify --pdf or --batch")
        return

    print(f"Processing {len(pdfs)} PDFs...\n")

    # Load existing results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = []

    processed_ndas = {r['nda'] for r in results}
    total_cost = 0
    total_tuples = 0

    for pdf in pdfs:
        nda = pdf.stem.replace('_ClinPharmR', '').replace('_MultidisciplineR', '')
        if nda in processed_ndas:
            print(f"  [{nda}] skipped (already processed)")
            continue

        result = extract_one_pdf(client, pdf, model)
        results.append(result)
        total_tuples += len(result.get('pk_tuples', []))

        # Cost estimation
        if result.get('usage'):
            if args.model == 'sonnet':
                cost = result['usage']['input_tokens'] * 3e-6 + result['usage']['output_tokens'] * 15e-6
            elif args.model == 'opus':
                cost = result['usage']['input_tokens'] * 15e-6 + result['usage']['output_tokens'] * 75e-6
            else:
                cost = result['usage']['input_tokens'] * 0.8e-6 + result['usage']['output_tokens'] * 4e-6
            total_cost += cost

        # Save incrementally
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total tuples: {total_tuples}")
    print(f"Total cost: ${total_cost:.3f}")
    print(f"Saved to: {out_path}")


if __name__ == '__main__':
    main()
