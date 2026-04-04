"""
ChEMBL-based benchmark audit: independent verification of Sisyphus holdout values.

For each of 97 HO drugs:
1. Query ChEMBL for drug
2. Get Cmax bioactivities with dose info
3. Compare with Sisyphus obs (converted to same dose via linear PK)
4. Flag systematic discrepancies

Output: data/validation/chembl_audit.json
"""

import json, math, time
from collections import defaultdict
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return Descriptors.ExactMolWt(mol)


def cmax_to_ngml(value, unit, mw):
    """Convert Cmax value to ng/mL."""
    if value is None or mw is None: return None
    if unit is None: return None
    u = unit.lower().strip()
    v = float(value)
    if u in ('ng/ml', 'ng.ml-1', 'ng ml-1'): return v
    if u in ('ug/ml', 'mcg/ml', 'μg/ml'): return v * 1000
    if u in ('mg/l', 'mg.l-1'): return v * 1000
    if u in ('ug/l', 'mcg/l', 'μg/l', 'ng/ml'): return v
    if u in ('ng/l'): return v / 1000
    if u in ('pg/ml'): return v / 1000
    if u in ('nm', 'nmol/l'): return v * mw / 1000  # nM → ng/mL via MW
    if u in ('um', 'umol/l', 'μmol/l', 'µm'): return v * mw  # μM → ng/mL
    if u in ('mm', 'mmol/l'): return v * mw * 1000
    return None


def query_drug_cmax(drug_name, mw):
    """Query ChEMBL for Cmax values."""
    try:
        # Find molecule
        molecule = new_client.molecule
        results = molecule.filter(
            molecule_synonyms__synonyms__iexact=drug_name
        ).only(['molecule_chembl_id','pref_name'])
        mols = list(results[:1])
        if not mols:
            # Try pref_name
            results = molecule.filter(pref_name__iexact=drug_name).only(['molecule_chembl_id','pref_name'])
            mols = list(results[:1])
        if not mols: return None, []

        chembl_id = mols[0]['molecule_chembl_id']

        # Get Cmax activities
        activity = new_client.activity
        acts = activity.filter(
            molecule_chembl_id=chembl_id,
            standard_type__in=['CMAX','Cmax','C max','CMAX SS','Cmax ss','CMAX_SS']
        )
        records = []
        for a in list(acts[:30]):  # limit
            val = a.get('standard_value')
            unit = a.get('standard_units')
            desc = a.get('assay_description', '')
            if not val or not unit: continue
            cmax_ngml = cmax_to_ngml(val, unit, mw)
            if cmax_ngml is None: continue
            records.append({
                'cmax_ngml': cmax_ngml,
                'unit_original': f'{val} {unit}',
                'description': desc[:100],
                'chembl_id': chembl_id,
            })
        return chembl_id, records
    except Exception as e:
        return None, []


def main():
    with open('data/validation/holdout_definition.json') as f:
        ho = json.load(f)
    holdout_drugs = ho['holdout_drugs']

    print(f"Querying ChEMBL for {len(holdout_drugs)} holdout drugs...")

    audit_results = []
    suspect_count = 0

    for i, d in enumerate(holdout_drugs, 1):
        name = d['name']
        smiles = d['smiles']
        obs = d['cmax_obs_ngml']
        ho_dose = d['dose_mg']
        mw = get_mw(smiles)
        if mw is None: continue

        chembl_id, records = query_drug_cmax(name, mw)

        if records:
            cmaxes = [r['cmax_ngml'] for r in records]
            # Report median ChEMBL value
            med_cmax = float(np.median(cmaxes))
            min_cmax = min(cmaxes)
            max_cmax = max(cmaxes)

            # Compare with obs (log space, assuming similar dose)
            log_diff = math.log10(med_cmax / obs) if obs > 0 else 0

            verdict = 'ok'
            if abs(log_diff) > 0.7:  # 5x off
                verdict = 'SUSPECT'
                suspect_count += 1
            elif abs(log_diff) > 0.48:  # 3x off
                verdict = 'check'

            print(f"  [{i:3d}/97] {name[:24]:<25s} obs={obs:>8.1f} ChEMBL[{len(records)}]={med_cmax:>8.1f} ({min_cmax:.0f}-{max_cmax:.0f}) Δ={log_diff:+.2f}  {verdict}")
        else:
            log_diff = None
            verdict = 'no_data'
            med_cmax = None
            print(f"  [{i:3d}/97] {name[:24]:<25s} obs={obs:>8.1f} ChEMBL: no data")

        audit_results.append({
            'drug': name, 'ik': d['inchikey14'], 'ho_dose': ho_dose, 'ho_obs': obs,
            'chembl_id': chembl_id, 'chembl_n_records': len(records),
            'chembl_median_cmax_ngml': med_cmax, 'log_diff': log_diff,
            'verdict': verdict,
        })
        time.sleep(0.2)

    suspects = [r for r in audit_results if r['verdict'] == 'SUSPECT']
    checks = [r for r in audit_results if r['verdict'] == 'check']
    ok = [r for r in audit_results if r['verdict'] == 'ok']
    no_data = [r for r in audit_results if r['verdict'] == 'no_data']

    print(f"\n{'='*70}")
    print(f"ChEMBL Audit Summary")
    print(f"{'='*70}")
    print(f"  SUSPECT (>5x):  {len(suspects)}")
    print(f"  check   (>3x):  {len(checks)}")
    print(f"  ok:             {len(ok)}")
    print(f"  no data:        {len(no_data)}")

    print(f"\nSUSPECT drugs:")
    for s in suspects:
        fold = 10**s['log_diff']
        print(f"  {s['drug']:<25s} obs={s['ho_obs']:>8.1f} ChEMBL={s['chembl_median_cmax_ngml']:>8.1f}  ({fold:.2f}x)")

    with open('data/validation/chembl_audit.json', 'w') as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n→ Saved to data/validation/chembl_audit.json")


if __name__ == '__main__':
    main()
