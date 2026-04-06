"""
RAG-PK Engine — Retrieval-Augmented Generation for PK Prediction.

Architecture:
  1. Build vector index of 801 training drugs (Morgan FP 2048-bit)
  2. For a query drug, retrieve Top-K most similar training drugs
  3. Format retrieved PK data as structured context
  4. Generate LLM prompt with retrieved context for Cmax prediction

This differs from prior similarity_calibration.py (post-hoc) by injecting
context AT PREDICTION TIME, letting the LLM reason about analogues.

Usage:
  python pipeline/rag_pk_engine.py --build-index    # Build the retrieval index
  python pipeline/rag_pk_engine.py --predict-ho      # Predict holdout with RAG
  python pipeline/rag_pk_engine.py --evaluate         # Evaluate RAG predictions
"""

import json
import math
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("Warning: RDKit not available, fingerprint similarity disabled")


# ─── Constants ───────────────────────────────────────────────────

TOP_K = 5                # Number of similar drugs to retrieve
FP_NBITS = 2048          # Morgan fingerprint bits
FP_RADIUS = 2            # Morgan fingerprint radius
MIN_SIMILARITY = 0.15    # Minimum Tanimoto to include as analogue


# ─── Fingerprint Index ───────────────────────────────────────────

class MorganFPIndex:
    """In-memory Morgan fingerprint index for fast Tanimoto similarity search."""

    def __init__(self):
        self.smiles_list = []
        self.fp_list = []
        self.metadata = []  # {name, dose_mg, cmax_ngml, ...}

    def add(self, smiles: str, metadata: dict):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
        self.smiles_list.append(smiles)
        self.fp_list.append(fp)
        self.metadata.append(metadata)
        return True

    def query(self, query_smiles: str, top_k: int = TOP_K) -> List[Tuple[float, dict]]:
        """Return top-k most similar drugs with Tanimoto scores."""
        mol = Chem.MolFromSmiles(query_smiles)
        if mol is None:
            return []
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)

        similarities = []
        for i, fp in enumerate(self.fp_list):
            sim = DataStructs.TanimotoSimilarity(query_fp, fp)
            if sim >= MIN_SIMILARITY:
                similarities.append((sim, i))

        similarities.sort(reverse=True)
        results = []
        for sim, idx in similarities[:top_k]:
            meta = self.metadata[idx].copy()
            meta['tanimoto'] = round(sim, 4)
            meta['smiles'] = self.smiles_list[idx]
            results.append((sim, meta))
        return results

    def __len__(self):
        return len(self.smiles_list)


# ─── Index Builder ───────────────────────────────────────────────

def build_training_index() -> Tuple[MorganFPIndex, dict]:
    """Build retrieval index from training data."""

    # Load training predictions (R2 analogical — best single round)
    with open('data/llm_extracted/llm_train_predictions.json') as f:
        train_preds = json.load(f)

    # Load combined dataset for actual Cmax values
    with open('data/curated/plm_sisyphus_combined.json') as f:
        combined = json.load(f)

    # Build SMILES → actual data lookup
    actual_by_smiles = defaultdict(list)
    for entry in combined:
        smi = entry.get('smiles', '')
        if smi:
            actual_by_smiles[smi].append(entry)

    # Load holdout InChIKeys to exclude
    with open('data/validation/holdout_definition.json') as f:
        ho_def = json.load(f)
    ho_inchikeys = set(ho_def.get('holdout_inchikeys', []))

    # Build InChIKey lookup for exclusion
    def get_inchikey14(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return Chem.inchi.InchiToInchiKey(Chem.inchi.MolToInchi(mol))[:14]
        except:
            return None

    index = MorganFPIndex()
    stats = {'total': 0, 'added': 0, 'skipped_ho': 0, 'skipped_invalid': 0}

    for smiles, pred_info in train_preds.items():
        stats['total'] += 1
        name = pred_info.get('name', 'unknown')
        pred_cmax = pred_info.get('predicted_cmax_ngml')

        # Check holdout exclusion
        ik14 = get_inchikey14(smiles)
        if ik14 and ik14 in ho_inchikeys:
            stats['skipped_ho'] += 1
            continue

        # Get actual data
        actuals = actual_by_smiles.get(smiles, [])
        actual_entries = []
        for a in actuals:
            actual_entries.append({
                'dose_mg': a.get('dose_mg'),
                'cmax_ngml': a.get('cmax_ngml'),
                'drug': a.get('drug', name),
                'source': a.get('source', ''),
            })

        # Compute molecular descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            stats['skipped_invalid'] += 1
            continue

        try:
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
        except:
            mw = logp = tpsa = hbd = hba = None

        metadata = {
            'name': name,
            'predicted_cmax_ngml': pred_cmax,
            'actual_entries': actual_entries,
            'inchikey14': ik14,
            'mw': mw,
            'logp': logp,
            'tpsa': tpsa,
            'hbd': hbd,
            'hba': hba,
        }

        if index.add(smiles, metadata):
            stats['added'] += 1
        else:
            stats['skipped_invalid'] += 1

    return index, stats


def format_retrieval_context(retrieved: List[Tuple[float, dict]], query_name: str,
                              query_dose: float) -> str:
    """Format retrieved analogues as structured LLM context."""
    if not retrieved:
        return "No structurally similar drugs found in training database."

    lines = [
        f"## Structurally Similar Drugs (Top {len(retrieved)} by Tanimoto similarity)",
        f"Query: {query_name} at {query_dose} mg\n",
    ]

    for i, (sim, meta) in enumerate(retrieved, 1):
        name = meta.get('name', 'unknown')
        tani = meta.get('tanimoto', 0)
        mw = meta.get('mw')
        logp = meta.get('logp')

        lines.append(f"### Analogue {i}: {name} (Tanimoto = {tani:.3f})")
        if mw is not None:
            lines.append(f"  MW: {mw:.1f}, LogP: {logp:.2f}, TPSA: {meta.get('tpsa', 0):.1f}")

        actuals = meta.get('actual_entries', [])
        if actuals:
            lines.append(f"  Known PK data:")
            for a in actuals[:5]:  # Cap at 5 entries per analogue
                dose = a.get('dose_mg', '?')
                cmax = a.get('cmax_ngml', '?')
                lines.append(f"    - Dose {dose} mg → Cmax {cmax} ng/mL")

        pred = meta.get('predicted_cmax_ngml')
        if pred:
            lines.append(f"  LLM predicted Cmax: {pred} ng/mL")
        lines.append("")

    return "\n".join(lines)


# ─── RAG Prediction Prompt ───────────────────────────────────────

RAG_PREDICTION_PROMPT = """You are a clinical pharmacology expert predicting human plasma Cmax.

{retrieval_context}

## Task
Predict the Cmax (peak plasma concentration in ng/mL) for:
- Drug: {drug_name}
- SMILES: {smiles}
- Dose: {dose_mg} mg (oral, single dose, fasted, healthy adult)

## Instructions
1. Analyze the structural similarity to the retrieved analogues above.
2. Consider how molecular properties (MW, LogP, TPSA) affect absorption and distribution.
3. Use the known PK data of analogues as anchoring points, adjusting for structural differences.
4. Account for dose-proportionality when scaling from analogue doses.

## Response Format
Return ONLY a JSON object:
{{"predicted_cmax_ngml": <float>, "reasoning": "<1-2 sentences>", "confidence": "<high|medium|low>", "key_analogue": "<name of most informative analogue>"}}
"""


def generate_rag_prompt(query_smiles: str, query_name: str, query_dose: float,
                        index: MorganFPIndex, top_k: int = TOP_K) -> Tuple[str, List]:
    """Generate a RAG-augmented prediction prompt."""
    retrieved = index.query(query_smiles, top_k=top_k)
    context = format_retrieval_context(retrieved, query_name, query_dose)

    prompt = RAG_PREDICTION_PROMPT.format(
        retrieval_context=context,
        drug_name=query_name,
        smiles=query_smiles,
        dose_mg=query_dose,
    )
    return prompt, retrieved


# ─── Evaluation ──────────────────────────────────────────────────

def evaluate_retrieval_quality(index: MorganFPIndex):
    """Analyze retrieval statistics for holdout drugs."""
    with open('data/validation/holdout_definition.json') as f:
        ho_def = json.load(f)

    drugs = ho_def.get('holdout_drugs', ho_def.get('drugs', []))
    if not drugs:
        print("No holdout drugs found in definition")
        return

    stats = {
        'total': len(drugs),
        'with_analogues': 0,
        'avg_top1_sim': [],
        'avg_top5_sim': [],
        'no_analogues': [],
    }

    results_per_drug = []

    for drug in drugs:
        smiles = drug.get('smiles', '')
        name = drug.get('name', '')
        dose = drug.get('dose_mg', 0)
        cmax_obs = drug.get('cmax_obs_ngml', 0)

        retrieved = index.query(smiles, top_k=TOP_K)

        if retrieved:
            stats['with_analogues'] += 1
            stats['avg_top1_sim'].append(retrieved[0][0])
            avg_sim = np.mean([r[0] for r in retrieved])
            stats['avg_top5_sim'].append(avg_sim)

            # Generate prompt for this drug
            prompt, _ = generate_rag_prompt(smiles, name, dose, index)

            results_per_drug.append({
                'name': name,
                'smiles': smiles,
                'dose_mg': dose,
                'cmax_obs_ngml': cmax_obs,
                'n_analogues': len(retrieved),
                'top1_sim': round(retrieved[0][0], 4),
                'top1_name': retrieved[0][1].get('name', ''),
                'avg_sim': round(avg_sim, 4),
                'prompt_length': len(prompt),
            })
        else:
            stats['no_analogues'].append(name)
            results_per_drug.append({
                'name': name,
                'smiles': smiles,
                'dose_mg': dose,
                'cmax_obs_ngml': cmax_obs,
                'n_analogues': 0,
                'top1_sim': 0,
                'top1_name': None,
                'avg_sim': 0,
                'prompt_length': 0,
            })

    # Save results
    out_path = Path('data/validation/rag_retrieval_analysis.json')
    output = {
        'stats': {
            'total_ho_drugs': stats['total'],
            'with_analogues': stats['with_analogues'],
            'without_analogues': len(stats['no_analogues']),
            'avg_top1_tanimoto': round(np.mean(stats['avg_top1_sim']), 4) if stats['avg_top1_sim'] else 0,
            'median_top1_tanimoto': round(np.median(stats['avg_top1_sim']), 4) if stats['avg_top1_sim'] else 0,
            'avg_top5_tanimoto': round(np.mean(stats['avg_top5_sim']), 4) if stats['avg_top5_sim'] else 0,
            'drugs_without_analogues': stats['no_analogues'],
        },
        'per_drug': results_per_drug,
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RAG Retrieval Analysis (Holdout, N={stats['total']})")
    print(f"{'='*60}")
    print(f"Drugs with analogues:     {stats['with_analogues']}/{stats['total']}")
    print(f"Avg Top-1 Tanimoto:       {np.mean(stats['avg_top1_sim']):.4f}" if stats['avg_top1_sim'] else "")
    print(f"Median Top-1 Tanimoto:    {np.median(stats['avg_top1_sim']):.4f}" if stats['avg_top1_sim'] else "")
    print(f"Avg Top-5 Tanimoto:       {np.mean(stats['avg_top5_sim']):.4f}" if stats['avg_top5_sim'] else "")

    if stats['no_analogues']:
        print(f"\nDrugs without analogues ({len(stats['no_analogues'])}):")
        for name in stats['no_analogues']:
            print(f"  - {name}")

    # Show sample prompt for first drug
    if drugs:
        sample_drug = drugs[0]
        sample_prompt, _ = generate_rag_prompt(
            sample_drug['smiles'], sample_drug['name'],
            sample_drug['dose_mg'], index
        )
        print(f"\n{'='*60}")
        print(f"Sample RAG Prompt ({sample_drug['name']}):")
        print(f"{'='*60}")
        print(sample_prompt[:2000])
        if len(sample_prompt) > 2000:
            print(f"... [{len(sample_prompt) - 2000} more chars]")

    print(f"\nOutput: {out_path}")
    return output


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RAG-PK Engine')
    parser.add_argument('--build-index', action='store_true', help='Build retrieval index')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate retrieval on holdout')
    parser.add_argument('--predict-ho', action='store_true', help='Generate RAG prompts for holdout')
    parser.add_argument('--top-k', type=int, default=TOP_K, help='Number of analogues')
    args = parser.parse_args()

    if not any([args.build_index, args.evaluate, args.predict_ho]):
        args.build_index = True
        args.evaluate = True

    if not HAS_RDKIT:
        print("Error: RDKit required for fingerprint computation")
        return

    # Build index
    print("Building training drug index...")
    index, stats = build_training_index()
    print(f"Index built: {stats['added']} drugs indexed "
          f"(skipped: {stats['skipped_ho']} HO, {stats['skipped_invalid']} invalid)")

    # Save index metadata
    index_meta = {
        'n_drugs': len(index),
        'fp_bits': FP_NBITS,
        'fp_radius': FP_RADIUS,
        'min_similarity': MIN_SIMILARITY,
        'top_k': args.top_k,
        'build_stats': stats,
    }
    Path('data/curated/rag_index_meta.json').parent.mkdir(parents=True, exist_ok=True)
    with open('data/curated/rag_index_meta.json', 'w') as f:
        json.dump(index_meta, f, indent=2)

    if args.evaluate or args.predict_ho:
        evaluate_retrieval_quality(index)

    if args.predict_ho:
        # Generate all RAG prompts for holdout
        with open('data/validation/holdout_definition.json') as f:
            ho_def = json.load(f)

        prompts = []
        for drug in ho_def.get('holdout_drugs', ho_def.get('drugs', [])):
            prompt, retrieved = generate_rag_prompt(
                drug['smiles'], drug['name'], drug['dose_mg'],
                index, top_k=args.top_k
            )
            prompts.append({
                'name': drug['name'],
                'smiles': drug['smiles'],
                'dose_mg': drug['dose_mg'],
                'cmax_obs_ngml': drug.get('cmax_obs_ngml'),
                'prompt': prompt,
                'n_retrieved': len(retrieved),
                'top1_sim': retrieved[0][0] if retrieved else 0,
            })

        out_path = Path('data/validation/rag_ho_prompts.json')
        with open(out_path, 'w') as f:
            json.dump(prompts, f, indent=2)
        print(f"\nGenerated {len(prompts)} RAG prompts → {out_path}")


if __name__ == '__main__':
    main()
