"""
Simple script to generate test files for standalone testing
"""

from rdkit import Chem
import pickle

ibuprofen_smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
acetaminophen_smiles = "CC(=O)Nc1ccc(O)cc1"
asprin_smiles = "O=C(C)Oc1ccccc1C(=O)O"

smiles = [ibuprofen_smiles, acetaminophen_smiles , asprin_smiles]

mols = [Chem.MolFromSmiles(s) for s in smiles]
mols = [Chem.AddHs(m) for m in mols]
[Chem.SanitizeMol(m) for m in mols]

FILENAME = "example"
w = Chem.SDWriter(f'{FILENAME}.sdf')
for m in mols: w.write(m)

pickle.dump([m.ToBinary() for m in mols], 
            open(f'{FILENAME}.rdkit', 'wb'))


