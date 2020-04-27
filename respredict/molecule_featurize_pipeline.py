"""
Featurization pipeline for molecules where each atom
gets its own feature vector
"""

"""
take in a dataframe with a bunch of Mol objects and then
compute the featurization and return the resulting dataframe
with index, atom_idx, feature
"""
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

import atom_features
import pickle
import pandas as pd
from ruffus import * 
from tqdm import tqdm
import torch
from multiprocessing import Pool
import itertools
import atom_features

WORKING_DIR = "features.molecular"
td = lambda x : os.path.join(WORKING_DIR, x)



featurize_params = {
    'atom_adj_mats_props_32' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf2_nmrshiftdb.pickle"], 
                                'featurizer' : 'mol_adj_mat', 
                                'array_out' : True, 
                                'max_atoms_in_molecule' : 32, 
                                'args' : [{'MAX_ATOM_N' : 32}]},
    'atom_adj_mats_props_32_kekulize' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf2_nmrshiftdb.pickle"], 
                                         'featurizer' : 'mol_adj_mat', 
                                         'array_out' : True, 
                                         'max_atoms_in_molecule' : 32, 
                                         'args' : [{'MAX_ATOM_N' : 32, 'kekulize' : True}]}, 
    
    'atom_adj_mats_props_32_aromatic' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf2_nmrshiftdb.pickle"], 
                                         'featurizer' : 'mol_adj_mat', 
                                         'array_out' : True, 
                                         'max_atoms_in_molecule' : 32, 
                                         'args' : [{'MAX_ATOM_N' : 32, 'kekulize' : False}]}, 
    

}

def params():
    mol_field = 'rdmol' # default
    for exp_name, ec in featurize_params.items():
        array_out = ec.get('array_out', False)
        for infile in ec['infiles']:
            data_filebase = os.path.splitext(os.path.basename(infile))[0] 
            for featurizer_args_i, featurizer_args in enumerate(ec['args']):
                outfile = td(f"{data_filebase}.{exp_name}.{featurizer_args_i}.pickle")
                max_run = ec.get('max_run', -1)
                feature_fields = ec.get('feature_fields', 'feature')
                max_atoms_in_molecule = ec.get("max_atoms_in_molecule", None) 
                yield (infile, outfile, mol_field, max_run, feature_fields, 
                       ec['featurizer'], featurizer_args, array_out , max_atoms_in_molecule)



def featurize(x):
    (mol_id, mol_row, mol_field,  feature_fields, featurizer,
     featurizer_args) = x

    mol = mol_row[mol_field]

    if mol is None:
        return []

    confs = mol.GetConformers()

    mol_feats = []

    for conf_i, conf in enumerate(confs):
        try:
            features = eval(featurizer)(mol, conf_i, **featurizer_args)
            fout = {'mol_id' : mol_id, 
                     'conf_i' : conf_i}
            if isinstance(feature_fields, str):
                # single feature
                fout[feature_fields] = features
            else:
                for fi, ff in enumerate(feature_fields):
                    fout[ff] = features[fi]
            mol_feats.append(fout)
        except:
            print(f"Warning: Skipping conformer {conf_i} for {mol_id}")
    return mol_feats

def mol_adj_mat(mol, conformer_i, **kwargs):
    MAX_ATOM_N = kwargs.get('MAX_ATOM_N', 64)
    KEKULIZE = kwargs.get('kekulize', True)
    atomic_nos_pad, adj = atom_features.mol_to_nums_adj(mol, MAX_ATOM_N, KEKULIZE)
    return adj


@mkdir(WORKING_DIR)
@files(params)
def featurize_molecules_par(infile, outfile, mol_field, max_run, feature_fields, 
                        featurizer, featurizer_args, array_out, max_atoms_in_molecule= None):
    """
    featurizers that run once per conformer and reutnr one feature per conformer
    
    This is frustrating because it is hard ahead of time to figure out 
    how many features we will end up with and what their number is, especially
    since featurization for a given molecule can fail. WE will upper-bound by
    sum(atomnum * conf*num * mol num)

    """


    molecules_df = pickle.load(open(infile, 'rb'))['df']
    if max_run > 0:
        molecules_df = molecules_df.iloc[:max_run]
    
    if max_atoms_in_molecule is not None:
        molecules_df = molecules_df[molecules_df.rdmol.apply(lambda x : x.GetNumAtoms() <= max_atoms_in_molecule)]

    FEATURE_NUM = len(molecules_df)
    if array_out:
        # output specific npy arrays
        # upper-limit on num of rows

        if isinstance(feature_fields, str):
            feature_field_filenames = [feature_fields]
        else:
            feature_field_filenames = feature_fields
        row_max = 0
        for row_i, row in molecules_df.iterrows():
            mol = row[mol_field]
            confs = mol.GetConformers()

            row_max += mol.GetNumAtoms() * len(confs)
        print("There are a max of", row_max, "rows")

        # assume each feature field is constant-dimensioned and featurize the first molecule
        res = featurize((0, molecules_df.iloc[0], 
                         mol_field, feature_fields, featurizer, featurizer_args))
        array_npy_outs = {}
        array_npy_outs_filenames = {}
        for f in feature_field_filenames:
            first_row = res[0]
            out_filename = outfile.replace(".pickle", f".{f}.npy")
            feature = first_row[f]
            out_shape = [row_max] + list(feature.shape)
            print("the out shape should be", out_shape, out_filename, feature.dtype)
            array_npy_outs[f] = np.lib.format.open_memmap(out_filename, mode='w+', 
                                                          shape = tuple(out_shape), 
                                                          dtype=feature.dtype)
            array_npy_outs_filenames[f] = out_filename
        array_pos = 0

    # FIXME use pywren at some point

    THREAD_N = 16
    pool = Pool(THREAD_N)
    print("GOING TO MAP")
    allfeat = []
    for features_for_mol in tqdm(pool.imap_unordered(featurize, 
                                                     [(a, b, mol_field, feature_fields, 
                                                       featurizer, featurizer_args) for a, b in molecules_df.iterrows()]), 
                                 total=len(molecules_df)):
        if not array_out:
            allfeat.append(features_for_mol)
        else:
            for feature in features_for_mol:
                for f in feature_field_filenames:
                    array_npy_outs[f][array_pos] = feature[f]
                    del feature[f] # delete the item 
                feature['array_pos'] = array_pos
                array_pos += 1
            allfeat.append(features_for_mol)

    pool.close()
    pool.join()

    df = pd.DataFrame(list(itertools.chain.from_iterable(allfeat)))

    out_dict = {'df' : df, 
                'featurizer' : featurizer, 
                'featurizer_args' : featurizer_args}
    if array_out:
        out_dict['feature_npy_filenames'] = array_npy_outs_filenames

    pickle.dump(out_dict, open(outfile, 'wb'))


if __name__ == "__main__":
    pipeline_run([featurize_molecules_par])
        
