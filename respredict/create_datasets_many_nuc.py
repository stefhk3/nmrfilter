import numpy as np
import pickle
import pandas as pd
from ruffus import * 
from tqdm import  tqdm
from rdkit import Chem
import pickle
import os

from glob import glob

import time
import util
import nets
import relnets

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors as rdMD

import torch
from torch import nn
from tensorboardX import SummaryWriter

from tqdm import  tqdm
from netdataio import * 
import netdataio
import itertools


DATASET_DIR = "graph_conv_many_nuc_pipeline.datasets"
td = lambda x : os.path.join(DATASET_DIR, x)

CV_SETS = [np.arange(4) + i*4 for i in range(5)]

default_spect_fname = 'dataset.named/spectra.nmrshiftdb_{}.feather'

solv_spect_fname = 'dataset.named/spectra.nmrshiftdb_{}_{}.feather'

NMRSHIFTDB_SUBSET_FILENAME =     'predict.atomic/molconf.nmrshiftdb_hconfspcl_nmrshiftdb.subsets.pickle'

spectra_sets = {#'13C_1H' : [('13C', default_spect_fname.format('13C')), 
    #            ('1H', default_spect_fname.format('1H'))], 
    '13C' : [('13C', default_spect_fname.format('13C'))], 
    '1H' : [('1H', default_spect_fname.format('1H'))], 
    
    '13C_cdcl3' : [('13C', solv_spect_fname.format('13C', 'cdcl3'))], 
    '13C_13C_cdcl3' : [ ('13Call', default_spect_fname.format('13C')), 
                        ('13C', solv_spect_fname.format('13C', 'cdcl3'))], 
    '1H_cdcl3' : [('1H', solv_spect_fname.format('1H', 'cdcl3'))], 
    
                # ### All Hs as bonded 
                # '13C_1HasBonded' : [('13C', default_spect_fname.format('13C')), 
                #                  ('1H', default_spect_fname.format('hconfspcl_1H_as_bonded'))],
                # ## Hs bonded to C, as H
                # '13C_1Hcbonded' : [('13C', default_spect_fname.format('13C')), 
                #                  ('1H', default_spect_fname.format('hconfspcl_1H_Cbonded'))],
                # ### only Hs bonded to C 
                # '13C_1HcbondedasBonded' : [('13C', default_spect_fname.format('13C')), 
                #                  ('1H', default_spect_fname.format('hconfspcl_1HCbonded_as_bonded'))]

}

def dataset_params():
    for CV_I in [0, 1, 2, 3, 4]: #  range(len(CV_SETS)):
        for kekulize_prop in ['aromatic']: # ['kekulize', 'aromatic']:
            for dataset_name in ['nmrshiftdb_hconfspcl_nmrshiftdb']:
                for spectra_set_name, spectra_config in spectra_sets.items():
                    for MAX_N in [64]: # [32, 64]:
                        
                        outfile = 'graph_conv_many_nuc_pipeline.data.{}.{}.{}.{}.{}.mol_dict.pickle'.format(spectra_set_name, dataset_name, 
                                                                                                            kekulize_prop, MAX_N, CV_I)
                        mol_filename = f'dataset.named/molconf.{dataset_name}.pickle'
                        yield ([sc[1] for sc in spectra_config] + [NMRSHIFTDB_SUBSET_FILENAME] + [mol_filename], 
                               td(outfile), CV_I, 
                               kekulize_prop, spectra_set_name, spectra_config, 
                               MAX_N)


                        
    qm9_spectra_sets = {'13C_1H' : [('13C', "dataset.named/spectra.qm9.cheshire_g09_01_nmr.13C.feather"), 
                                   ('1H', "dataset.named/spectra.qm9.cheshire_g09_01_nmr.1H.feather")], 
                        '13C' : [('13C', "dataset.named/spectra.qm9.cheshire_g09_01_nmr.13C.feather"),], 
                        '1H' : [('1H', "dataset.named/spectra.qm9.cheshire_g09_01_nmr.1H.feather")], 
                                    
}

    QM9_SUBSET_FILENAME =  'dataset.named/qm9.subsets.pickle'


    for CV_I in [0]: #  range(len(CV_SETS)):
        for kekulize_prop in ['kekulize', 'aromatic']:
            for dataset_name in ['qm9']:
                for spectra_set_name, spectra_config in qm9_spectra_sets.items():
                    for MAX_N in [32]:
                        
                        outfile = 'graph_conv_many_nuc_pipeline.data.{}.{}.{}.{}.{}.mol_dict.pickle'.format(spectra_set_name, dataset_name, 
                                                                                                            kekulize_prop, MAX_N, CV_I)
                        mol_filename = f'dataset.named/molconf.{dataset_name}.pickle'
                        yield ([sc[1] for sc in spectra_config] +  [QM9_SUBSET_FILENAME] + [mol_filename], 
                               td(outfile), CV_I, 
                               kekulize_prop, spectra_set_name, spectra_config, 
                               MAX_N)


@mkdir(DATASET_DIR)
@files(dataset_params)
def create_dataset(infiles, outfile, cv_i,  kekulize_prop, 
                   spectra_set_name, spectra_config, MAX_N):
    mol_filename = infiles[-1]
    mol_subset_filename = infiles[-2]

    cv_mol_subset = CV_SETS[cv_i]
    mol_subsets = pickle.load(open(mol_subset_filename, 'rb'))['splits_df']

    
    tgt_nucs = [sc[0] for sc in spectra_config]
    spectra_dfs = []
    for nuc, spectra_filename in spectra_config:
        df =  pd.read_feather(spectra_filename)
        df = df.rename(columns={'id' : 'peak_id'}) 
        df['nucleus'] = nuc
        spectra_dfs.append(df)
    spectra_df = pd.concat(spectra_dfs)
                                                                 
    molecules_df = pickle.load(open(mol_filename, 'rb'))['df']
    molecules_df['atom_n'] = molecules_df.rdmol.apply(lambda x: x.GetNumAtoms())
    molecules_df = molecules_df[molecules_df.atom_n <= MAX_N]
    

    def s_dict(r):
        return dict(zip(r.atom_idx, r.value))



    spect_dict_df = spectra_df.groupby(['molecule_id', 'spectrum_id', 'nucleus']).apply(s_dict )

    data_df = spect_dict_df.reset_index()\
                           .rename(columns={0 : 'value'})\
                           .join(molecules_df, on='molecule_id').dropna()

    for row_i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        mol = row.rdmol
        try:
            Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, 
                             catchErrors=True)
            mol.UpdatePropertyCache()
            Chem.SetAromaticity(mol)
            if kekulize_prop == 'kekulize':
                Chem.rdmolops.Kekulize(mol)
        except ValueError:
            pass
            
        

    ### Combine and create the cartesian product across different nuclei
    ### of various spectra
    mol_data = []
    for mol_id, mol_df in tqdm(data_df.groupby('molecule_id')):
        sp = {n : list([(-1, {})]) for n in tgt_nucs}
        for row_i, row in mol_df.iterrows():
            sp[row.nucleus].append((row.spectrum_id, row.value))


        for nuc_lists in itertools.product(*[sp[n] for n in tgt_nucs]):
            spectra_ids = [a[0] for a in nuc_lists]
            if (np.array(spectra_ids) == -1).all():
                continue
            values = [a[1] for a in nuc_lists]
            mol_data.append({'molecule_id' : mol_id, 
                            'rdmol' : row.rdmol, 
                             'spectra_ids' : spectra_ids, 
                            'value': values})
        #mol_data.append({'molecule_id' : mol_id, ''})
    data_df = pd.DataFrame(mol_data)

    ### Train/test split
    train_test_split = mol_subsets.subset20_i.isin(cv_mol_subset)
    train_mols = mol_subsets[~train_test_split].index.values
    test_mols = mol_subsets[train_test_split].index.values

    train_df = data_df[data_df.molecule_id.isin(train_mols)]
    test_df = data_df[data_df.molecule_id.isin(test_mols)]

    pickle.dump({'train_df' : train_df, 
                 'test_df' : test_df, 
                 'MAX_N' : MAX_N, 
                 'spectra_config' : spectra_config, 
                 'tgt_nucs' : tgt_nucs}, 
                open(outfile, 'wb'), -1)


if __name__ == "__main__":
    pipeline_run([create_dataset])
