"""
Merge LLM-extracted PK tuples with v10 dataset.
Check for holdout overlap.
Create expanded dataset for ML training.
"""

import json
import math
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smiles_to_ik(smiles):
    """Canonical InChIKey (14 char)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.InchiToInchiKey(Chem.MolToInchi(mol))[:14]
    except Exception:
        return None


def main():
    # Load LLM extractions
    with open('data/llm_extracted/pk_llm_merged.json') as f:
        llm_tuples = json.load(f)
    print(f"LLM tuples: {len(llm_tuples)}")

    # Load v10
    with open('data/curated/plm_dataset_v10_labels.json') as f:
        v10 = json.load(f)
    if isinstance(v10, dict):
        v10 = v10.get('profiles', [])
    print(f"v10 profiles: {len(v10)}")

    # Load holdout
    with open('data/validation/holdout_definition.json') as f:
        ho = json.load(f)
    holdout_iks = set(ho.get('holdout_inchikeys', []))
    holdout_iks_14 = set(ik[:14] for ik in holdout_iks)
    print(f"Holdout drugs: {len(holdout_iks_14)}")

    # Convert LLM tuples to v10 schema
    new_profiles = []
    skipped = {'no_cmax': 0, 'no_dose': 0, 'no_smiles': 0, 'bad_ik': 0,
               'holdout_leak': 0, 'invalid_cmax': 0}
    holdout_leak_drugs = []
    new_drugs = set()
    holdout_overlap_drugs = set()

    for t in llm_tuples:
        smi = t.get('smiles')
        cmax = t.get('cmax_ng_ml')
        dose = t.get('dose_mg')

        if not smi:
            skipped['no_smiles'] += 1
            continue
        if not cmax or cmax <= 0:
            skipped['no_cmax'] += 1
            continue
        if not dose or dose <= 0:
            skipped['no_dose'] += 1
            continue

        ik = smiles_to_ik(smi)
        if not ik:
            skipped['bad_ik'] += 1
            continue

        # Check holdout overlap
        if ik in holdout_iks_14:
            skipped['holdout_leak'] += 1
            holdout_leak_drugs.append(t.get('drug_name', ''))
            holdout_overlap_drugs.add(t.get('drug_name', ''))
            continue

        log_cd = math.log10(cmax / dose)
        if log_cd < -6 or log_cd > 3:
            skipped['invalid_cmax'] += 1
            continue

        new_profiles.append({
            'smiles': smi,
            'dose_mg': dose,
            'cmax_ngml': cmax,
            'ik': ik,
            'src': 'LLM_FDA',
            'log_cd': log_cd,
        })
        new_drugs.add(ik)

    print(f"\n=== Conversion ===")
    print(f"New profiles: {len(new_profiles)}")
    print(f"New drugs: {len(new_drugs)}")
    print(f"Skipped: {skipped}")
    if holdout_leak_drugs:
        print(f"\n⚠️  Holdout drugs in LLM extraction (excluded from training):")
        for d in sorted(set(holdout_leak_drugs)):
            print(f"  - {d}")
        print(f"  Total holdout-overlapping drugs: {len(holdout_overlap_drugs)}")

    # Check overlap with v10
    v10_iks = set(p.get('ik', '') for p in v10)
    overlap_v10 = new_drugs & v10_iks
    print(f"\nNew drugs overlapping with v10: {len(overlap_v10)}")
    print(f"Truly new drugs (vs v10 AND holdout): {len(new_drugs - v10_iks)}")

    # Merge
    merged = v10 + new_profiles
    print(f"\n=== Merged Dataset ===")
    print(f"Total profiles: {len(merged)}")
    print(f"v10 profiles: {len(v10)}")
    print(f"LLM profiles added: {len(new_profiles)}")

    all_iks = set(p.get('ik', '') for p in merged)
    print(f"Unique drugs: {len(all_iks)}")
    print(f"Holdout overlap: {len(all_iks & holdout_iks_14)}")  # should be 0

    # Save merged
    out = Path('data/curated/plm_dataset_v11_llm.json')
    with open(out, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\n→ Saved to {out}")


if __name__ == '__main__':
    main()
