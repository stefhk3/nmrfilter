"""
Gerneric pipeline to split and cleanup datasets 
of coupling 
"""

import pandas as pd
import numpy as np
from rdkit import Chem
import os
import pickle
from glob import glob
from tqdm import tqdm
from ruffus import * 

WORKING_DIR = "dataset.named"
td = lambda x: os.path.join(WORKING_DIR, x)

DATASET_DIR = "coupling.datasets"
tdd = lambda x: os.path.join(DATASET_DIR, x)


@files("../nmrabinitio/data/gissmo.mol.pickle", 
       td("gissmo.subsets.pickle"))
def create_subsets(infile, outfile):
    mol_df = pickle.load(open(infile, 'rb'))

    mol_df['subset20_i'] = np.random.permutation(len(mol_df)) % 20
    mol_df_subsets = mol_df[['subset20_i']]

    pickle.dump(mol_df_subsets, 
                open(outfile, 'wb'))


def element_params():
    for elt_set in [['H', 'C', 'O'], 
                    ['H', 'C', 'O', 'N', 'F'], 
                    ['H', 'C', 'O', 'N', 'F', 'S', 'P', 'Cl']]:
        infile = "../nmrabinitio/data/gissmo.mol.pickle"
        elt_set_str = "".join(elt_set)
        outfile = td(f"gissmo.mols.{elt_set_str}.pickle")
        yield infile, outfile, elt_set

@files(element_params)
def create_atom_subsets(infile, outfile, elt_set):
    
    mol_df = pickle.load(open(infile, 'rb'))

    def only_in(m):
        for a in m.GetAtoms():
            if a.GetSymbol() not in elt_set: 
                return False
        return True


    mol_df = mol_df[mol_df.rdmol.apply(only_in)].copy()
    print('writing', outfile)
    pickle.dump(mol_df, open(outfile, 'wb'))


CV_SETS = [np.arange(4) + i*4 for i in range(5)]


def dataset_params():
    for mol_filename in glob(td("gissmo.mols.*.pickle")):
        for MAX_N in [16, 32, 64]:
            for cv_i in range(5):
                shifts_filename = "../nmrabinitio/data/gissmo.shifts.feather"
                couplings_filename = "../nmrabinitio/data/gissmo.j.feather"
                subset_filename = td("gissmo.subsets.pickle")
                basename = os.path.basename(mol_filename)
                out_filename = basename.replace(".mols", f".moldict.{MAX_N}.{cv_i}")
                out_filename = tdd(out_filename)

                cv_mol_subset = CV_SETS[cv_i]
                yield ((mol_filename, shifts_filename, couplings_filename, subset_filename), 
                       out_filename, cv_i, cv_mol_subset, MAX_N)

@follows(create_subsets)
@mkdir(DATASET_DIR )    
@files(dataset_params)
def create_dataset(infiles, outfile, cv_i, cv_mol_subset, MAX_N):
    
    mol_filename, shifts_filename, couplings_filename, subset_filename = infiles

    mol_df = pickle.load(open(mol_filename, 'rb'))
    shifts_df = pd.read_feather(shifts_filename)
    couplings_df = pd.read_feather(couplings_filename)
    
    mol_subsets = pickle.load(open(subset_filename, 'rb'))

    # atom count which is useful later
    mol_df['atom_n'] = mol_df.rdmol.apply(lambda x: x.GetNumAtoms())
    mol_df = mol_df[mol_df.atom_n <= MAX_N].copy()

    def shift_to_dict(g):
        return [{a.atom_idx : a['value'] for ai, a in g.iterrows()}]

    def J_to_dict(g):
        return [{(int(a.a1), int(a.a2)) : a['J'] for ai, a in g.iterrows()}]

    mol_df = mol_df.reset_index()
    print(mol_df.dtypes)

    ## Compute and merge shifts
    shift_dict_vals = pd.DataFrame(shifts_df.groupby('mol_id').apply(shift_to_dict)).rename(columns={0: 'shifts'})
    data_df = mol_df.join(shift_dict_vals, on='mol_id')

    ## compute and merge couplings
    couplings_dict_vals = pd.DataFrame(couplings_df.groupby('mol_id').apply(J_to_dict)).rename(columns={0: 'couplings'})
    data_df = data_df.join(couplings_dict_vals, on='mol_id')

    # cleanup molecules

    for row_i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        mol = row.rdmol
        try:
            Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, 
                             catchErrors=True)
            mol.UpdatePropertyCache()
            Chem.SetAromaticity(mol)
            #if kekulize_prop == 'kekulize':
            #    Chem.rdmolops.Kekulize(mol)
        except ValueError:
            pass
            

    print(mol_subsets.dtypes)

    print("Dropping", len(data_df) - len(data_df.dropna()), "records for NaNs")
    data_df = data_df.dropna()

    ### Train/test split
    train_test_split = mol_subsets.subset20_i.isin(cv_mol_subset)
    train_mols = mol_subsets[~train_test_split].index.values
    test_mols = mol_subsets[train_test_split].index.values

    train_df = data_df[data_df.mol_id.isin(train_mols)]
    test_df = data_df[data_df.mol_id.isin(test_mols)]

    print(outfile)
    print("len(train_df)=", len(train_df))
    print("len(test_df)=", len(test_df))

    pickle.dump({'train_df' : train_df, 
                 'test_df' : test_df, 
                 'MAX_N' : MAX_N, 
                 #'spectra_config' : spectra_config, 
                 #'tgt_nucs' : tgt_nucs},
    }, 
                open(outfile, 'wb'), -1)

if __name__ == "__main__":
    pipeline_run([create_subsets, create_atom_subsets, create_dataset])
