"""
Extract MoLFormer-XL embeddings for all unique SMILES in training + HO.
Saves to data/curated/molformer_embeddings.json (SMILES → 768-dim vector).
"""

import json, math
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return Chem.MolToSmiles(mol)

def collect_all_smiles():
    """Collect unique canonical SMILES from v10 + LLM + HO."""
    smiles_set = set()
    with open('data/curated/plm_dataset_v10_labels.json') as f: v10 = json.load(f)
    with open('data/llm_extracted/pk_llm_merged.json') as f: llm = json.load(f)
    with open('data/validation/holdout_definition.json') as f: ho_data = json.load(f)

    for p in v10:
        s = p.get('smiles')
        if s:
            can = canonicalize(s)
            if can: smiles_set.add(can)
    for t in llm:
        s = t.get('smiles')
        if s:
            can = canonicalize(s)
            if can: smiles_set.add(can)
    for d in ho_data['holdout_drugs']:
        s = d.get('smiles')
        if s:
            can = canonicalize(s)
            if can: smiles_set.add(can)
    return sorted(smiles_set)

def extract_embeddings(smiles_list, model, tokenizer):
    """Extract 768-dim embeddings via MoLFormer pooler_output."""
    embeddings = {}
    n = len(smiles_list)
    for i in range(0, n, BATCH_SIZE):
        batch = smiles_list[i:i+BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        # Use pooler_output (768-dim per SMILES)
        embs = outputs.pooler_output.cpu().numpy()
        for smi, emb in zip(batch, embs):
            embeddings[smi] = emb.astype(np.float32).tolist()
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Batch {i//BATCH_SIZE + 1}/{(n+BATCH_SIZE-1)//BATCH_SIZE}, {len(embeddings)}/{n} done")
    return embeddings

def main():
    print("Collecting unique SMILES...")
    smiles_list = collect_all_smiles()
    print(f"  {len(smiles_list)} unique canonical SMILES")

    print(f"Loading MoLFormer-XL on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model = AutoModel.from_pretrained('ibm/MoLFormer-XL-both-10pct', trust_remote_code=True)
    model.eval()
    model = model.to(DEVICE)

    print("Extracting embeddings...")
    embeddings = extract_embeddings(smiles_list, model, tokenizer)

    print(f"Saving {len(embeddings)} embeddings to data/curated/molformer_embeddings.json")
    with open('data/curated/molformer_embeddings.json', 'w') as f:
        json.dump(embeddings, f)
    print("Done.")

if __name__ == '__main__':
    main()
