"""
Extract PK-relevant text from ALL text-extractable FDA PDFs.
Skips PDFs already processed.
"""

import re
import json
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
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return []
    pages = []
    for i in range(len(doc)):
        try:
            text = doc[i].get_text()
        except Exception:
            continue
        hits = len(PK_KEYWORDS.findall(text))
        if hits >= min_hits and len(text) > 200:
            pages.append({'page': i + 1, 'hits': hits, 'text': text})
    doc.close()
    pages.sort(key=lambda p: -p['hits'])
    pages = pages[:MAX_PAGES]
    pages.sort(key=lambda p: p['page'])
    return pages


def build_document_text(pages):
    chunks, total = [], 0
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
    all_pdfs = sorted(Path('data/raw').glob('*.pdf'))
    out_dir = Path('data/llm_extracted/text')
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    skipped, no_pk, processed = 0, 0, 0

    for i, pdf in enumerate(all_pdfs):
        nda = pdf.stem.replace('_ClinPharmR', '').replace('_MultidisciplineR', '')
        out_file = out_dir / f'{nda}.txt'

        if out_file.exists():
            manifest.append({'nda': nda, 'text_file': str(out_file), 'chars': out_file.stat().st_size})
            skipped += 1
            continue

        pages = find_pk_pages(pdf)
        if not pages:
            no_pk += 1
            continue

        text = build_document_text(pages)
        out_file.write_text(text, encoding='utf-8')
        manifest.append({'nda': nda, 'text_file': str(out_file), 'chars': len(text)})
        processed += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_pdfs)}: +{processed} new, {skipped} skipped, {no_pk} no PK")

    manifest_file = Path('data/llm_extracted/text/manifest_all.json')
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total PDFs: {len(all_pdfs)}")
    print(f"Newly processed: {processed}")
    print(f"Already had text: {skipped}")
    print(f"No PK content: {no_pk}")
    print(f"Total text files: {len(manifest)}")
    print(f"Manifest: {manifest_file}")


if __name__ == '__main__':
    main()
