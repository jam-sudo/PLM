"""
ChEMBL drug space expansion: find new drugs with Cmax data not in v10 or holdout.

Strategy:
1. Query ChEMBL for all drugs with Cmax data (human, clinical)
2. Filter out drugs already in v10 or holdout
3. Extract (drug, SMILES, dose, Cmax) tuples where available
4. Add to training dataset
"""

import json, math, time
from collections import defaultdict
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def smiles_to_ik(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try: return Chem.InchiToInchiKey(Chem.MolToInchi(mol))[:14]
    except: return None


def cmax_to_ngml(value, unit, mw):
    if value is None or mw is None or unit is None: return None
    u = unit.lower().strip()
    v = float(value)
    if u in ('ng/ml',): return v
    if u in ('ug/ml', 'mcg/ml', 'μg/ml'): return v * 1000
    if u in ('mg/l',): return v * 1000
    if u in ('ug/l', 'mcg/l', 'μg/l'): return v
    if u in ('ng/l',): return v / 1000
    if u in ('pg/ml',): return v / 1000
    if u == 'nm': return v * mw / 1000
    if u == 'um': return v * mw
    return None


def extract_dose_from_description(desc):
    """Extract mg dose from assay description."""
    if not desc: return None
    import re
    # Match patterns like "at 100 mg", "100mg", "100 mg QD"
    m = re.search(r'(\d+(?:\.\d+)?)\s*mg\b', desc, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def main():
    # Load existing data
    with open('data/curated/plm_dataset_v10_labels.json') as f:
        v10 = json.load(f)
    v10_iks = set(p.get('ik','') for p in v10 if p.get('ik'))

    with open('data/validation/holdout_definition.json') as f:
        ho = json.load(f)
    ho_iks = set(d['inchikey14'] for d in ho['holdout_drugs'])

    print(f"v10 drugs: {len(v10_iks)}, Holdout drugs: {len(ho_iks)}")
    print(f"Total to exclude: {len(v10_iks | ho_iks)}")

    # Query ChEMBL for all human Cmax bioactivities
    print("\nQuerying ChEMBL for human Cmax bioactivities...")
    activity = new_client.activity
    acts = activity.filter(
        standard_type__in=['CMAX','Cmax','C max'],
        assay_organism='Homo sapiens',
    ).only(['molecule_chembl_id', 'standard_value', 'standard_units',
            'assay_description', 'canonical_smiles'])

    # Group by chembl_id
    by_mol = defaultdict(list)
    count = 0
    for a in acts:
        count += 1
        if count % 1000 == 0: print(f"  processed {count} activities...")
        if count > 20000: break  # cap

        mol_id = a.get('molecule_chembl_id')
        smi = a.get('canonical_smiles')
        val = a.get('standard_value')
        unit = a.get('standard_units')
        desc = a.get('assay_description', '')

        if not all([mol_id, smi, val, unit]): continue

        ik = smiles_to_ik(smi)
        if not ik: continue
        if ik in v10_iks or ik in ho_iks: continue

        dose = extract_dose_from_description(desc)
        if not dose: continue

        mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) if smi else None
        if not mw: continue
        cmax = cmax_to_ngml(val, unit, mw)
        if cmax is None or cmax <= 0: continue

        by_mol[ik].append({
            'smi': smi, 'ik': ik, 'dose_mg': dose, 'cmax_ngml': cmax,
            'log_cd': math.log10(cmax/dose) if dose > 0 else None,
            'description': desc[:80], 'chembl_id': mol_id,
        })

    print(f"\nTotal activities processed: {count}")
    print(f"Unique NEW drugs (not in v10/holdout): {len(by_mol)}")

    # Aggregate per drug: median log_cd
    new_entries = []
    for ik, entries in by_mol.items():
        log_cds = [e['log_cd'] for e in entries if e['log_cd'] is not None]
        if not log_cds: continue
        if min(log_cds) < -3 or max(log_cds) > 3: continue  # sanity
        # Take median + use first SMILES
        med_log_cd = float(np.median(log_cds))
        med_dose = float(np.median([e['dose_mg'] for e in entries]))
        if med_dose <= 0: continue
        cmax = 10**med_log_cd * med_dose
        new_entries.append({
            'smiles': entries[0]['smi'],
            'ik': ik,
            'dose_mg': med_dose,
            'cmax_ngml': cmax,
            'log_cd': med_log_cd,
            'src': 'CHEMBL',
            'n_records': len(entries),
        })

    print(f"Usable new drug profiles: {len(new_entries)}")
    print(f"\nSample:")
    for e in new_entries[:10]:
        print(f"  {e['ik']:<17s} dose={e['dose_mg']:>8.1f} Cmax={e['cmax_ngml']:>9.1f} n={e['n_records']}")

    # Save
    with open('data/curated/chembl_pk_expansion.json', 'w') as f:
        json.dump(new_entries, f, indent=2)
    print(f"\n→ Saved {len(new_entries)} new profiles to data/curated/chembl_pk_expansion.json")


if __name__ == '__main__':
    main()
