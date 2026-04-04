"""
Extract PK-relevant text from FDA PDFs into text files for LLM processing.
Output: data/llm_extracted/text/{nda}.txt
"""

import re
import json
import random
from pathlib import Path
import fitz


PK_KEYWORDS = re.compile(
    r'\b(cmax|c_max|tmax|t_max|auc|area under|pharmacokinetic|'
    r'bioavailab|half-life|half life|t1/2|t½|'
    r'clearance|volume of distribution|absorption|elimination)\b',
    re.IGNORECASE
)

MAX_TEXT_CHARS = 80_000
MAX_PAGES = 30


def find_pk_pages(pdf_path, min_hits=2):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        hits = len(PK_KEYWORDS.findall(text))
        if hits >= min_hits and len(text) > 200:
            pages.append({'page': i + 1, 'hits': hits, 'text': text, 'chars': len(text)})
    doc.close()
    pages.sort(key=lambda p: -p['hits'])
    pages = pages[:MAX_PAGES]
    pages.sort(key=lambda p: p['page'])
    return pages


def build_document_text(pages):
    chunks = []
    total = 0
    for p in pages:
        header = f"\n\n===== PAGE {p['page']} =====\n\n"
        needed = len(header) + len(p['text'])
        if total + needed > MAX_TEXT_CHARS:
            remaining = MAX_TEXT_CHARS - total - len(header)
            if remaining > 500:
                chunks.append(header + p['text'][:remaining])
            break
        chunks.append(header + p['text'])
        total += needed
    return "".join(chunks)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    all_pdfs = sorted(Path('data/raw').glob('*.pdf'))
    rng = random.Random(args.seed)

    # Only PDFs with text content
    pdfs_with_text = []
    for pdf in all_pdfs:
        try:
            doc = fitz.open(pdf)
            n = len(doc)
            sample_pages = [n//4, n//2, 3*n//4] if n > 4 else [0]
            total = sum(len(doc[i].get_text()) for i in sample_pages)
            doc.close()
            if total > 500:
                pdfs_with_text.append(pdf)
        except Exception:
            pass

    print(f"{len(pdfs_with_text)} text-extractable PDFs")
    rng.shuffle(pdfs_with_text)
    selected = pdfs_with_text[:args.n]

    out_dir = Path('data/llm_extracted/text')
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for pdf in selected:
        nda = pdf.stem.replace('_ClinPharmR', '').replace('_MultidisciplineR', '')
        pages = find_pk_pages(pdf)
        if not pages:
            print(f"  {nda}: no PK pages")
            continue
        text = build_document_text(pages)
        out_file = out_dir / f'{nda}.txt'
        out_file.write_text(text, encoding='utf-8')
        print(f"  {nda}: {len(pages)} pages, {len(text)} chars → {out_file.name}")
        manifest.append({
            'nda': nda,
            'pdf': str(pdf),
            'n_pages': len(pages),
            'chars': len(text),
            'text_file': str(out_file),
        })

    manifest_file = out_dir / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_file}")
    print(f"Processed: {len(manifest)} PDFs")


if __name__ == '__main__':
    main()
