"""
Featurize pipeline to generate per-molecule features based on geometry. 

Input is (molecule ID, rdmol, list of conformers)
Many conformers in the search process are not relevant. 

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
import util
import itertools
from atom_features import atom_adj_mat, advanced_atom_props
import atom_features_old
import shutil

WORKING_DIR = "features.geom"
td = lambda x : os.path.join(WORKING_DIR, x)


DEFAULT_ANGULAR_ARGS = {'thetas' :  np.linspace(0, np.pi, 6), 
                        'rses': np.linspace(0.5, 3.5, 8), 
                        'neighbor_cutoff' : 5.0, 
                        'pairings'  : [(1, 6), (1, 7), (1, 8), (6, 6), (6, 7), (6, 8)]}


DEFAULT_BP_IMAGES_ARGS = {'atomic_num_rows' : [1, 6, 7, 8, 9], 
                          'pairings' : [(1, 1), (1, 6), (1, 7), (1, 8), (1, 9), (6, 6), (6, 7), (6, 8), (6, 9)], 
                          'theta_bin_edges' : np.linspace(0, np.pi*1.01, 17), 
                          'r_bin_edges' :  np.logspace(np.log10(0.75), np.log10(4.5), 17)
}
DENSE_BP_IMAGES_ARGS = {'atomic_num_rows' : [1, 6, 7, 8, 9], 
                          'pairings' : [(1, 1), (1, 6), (1, 7), (1, 8), (1, 9), (6, 6), (6, 7), (6, 8), (6, 9)], 
                          'theta_bin_edges' : np.linspace(0, np.pi*1.01, 17), 
                          'r_bin_edges' :  np.logspace(np.log10(0.75), np.log10(12.0), 64)
}

HCONFSPCL_RADIAL = dict(ATOMIC_NUM_TO_POS = {6: 0, 1: 1, 7: 2, 8: 3, 9:4, 15: 5, 16: 6, 17: 7})
NEAR_RADIAL = dict(ATOMIC_NUM_TO_POS = 
                   {6: 0, 1: 1, 7: 2, 8: 3, 9:4, 15: 5, 16: 6, 17: 7}, 
                   r_centers = 
                   np.logspace(np.log10(0.95), np.log10(5.0), 30),
                   bin_widths = 
                   np.logspace(np.log10(0.95), np.log10(5.0), 30)/ 512, 
)



ALL_ATOMIC_NOS =  [1, 6, 7, 8, 9, 15, 16, 17]
all_pairs = []
for ai in range(len(ALL_ATOMIC_NOS)):
    for bi in range(ai, len(ALL_ATOMIC_NOS)):
        all_pairs.append((ALL_ATOMIC_NOS[ai], ALL_ATOMIC_NOS[bi]))
        

HCONFSPCL_ANGULAR_ARGS = {'thetas' :  np.linspace(0, np.pi, 6), 
                          'rses': np.linspace(0.5, 3.5, 8), 
                          'neighbor_cutoff' : 5.0, 
                          'pairings'  : all_pairs}


featurize_params = {'default_bp_radial' : {'infiles' : ["../nmrdata/confs/rdkit_ff.mmff94_64_opt2.conf.pickle"],
                                           'featurizer' : 'bp_radial',
                                           'args' : [{}], }, 
                    # 'hconsfspcl_bp_radial' : {'infiles' : ["dataset.named/molconf.nmrshiftdb_hconfspcl_nmrshiftdb.pickle"], 
                    #                           'featurizer' : 'bp_radial',
                    #                           'args' : [HCONFSPCL_RADIAL],}, 
                    # 'default_bp_radial_F' : {'infiles' : ["dataset.named/molconf.nmrshiftdb_hconf_nmrshiftdb.pickle"], 
                    #                        'featurizer' : 'bp_radial',
                    #                        'args' : [{'ATOMIC_NUM_TO_POS' : {6: 0, 1: 1, 7: 2, 8: 3, 9: 4}}]
                    #              }, 
                    # 'default_bp_radial_F_cutoffs' : {'infiles' : ["nmrshiftdb_data/nmrshiftdb.mol3d_etkdg.pickle"], 
                    #                                 'featurizer' : 'bp_radial',
                    #                                 'args' : [{'ATOMIC_NUM_TO_POS' : {6: 0, 1: 1, 7: 2, 8: 3, 9: 4, }, 'neighbor_cutoff' : 6.0}, 
                    #                                           {'ATOMIC_NUM_TO_POS' : {6: 0, 1: 1, 7: 2, 8: 3, 9: 4, }, 'neighbor_cutoff' : 10.0}, 
                    #                                           {'ATOMIC_NUM_TO_POS' : {6: 0, 1: 1, 7: 2, 8: 3, 9: 4, }, 'neighbor_cutoff' : 100.0}, 
                    #                                           ]},


                    # 'default_bp_radial_fine' : {'infiles' : ["nmrshiftdb_data/nmrshiftdb.mol3d_etkdg.pickle"], 
                    #                        'featurizer' : 'bp_radial',
                    #                        'args' : [{'ATOMIC_NUM_TO_POS' : {6: 0, 1: 1, 7: 2, 8: 3, 9: 4}, 
                    #                                   'r_centers' : np.linspace(0.85, 1.6, 100)**2, 
                    #                                   'bin_widths' : np.linspace(0.001, 0.02, 100)/30}]
                    #              }, 
                    # 'default_bp_radial_vfine' : {'infiles' : ["nmrshiftdb_data/nmrshiftdb.mol3d_etkdg.pickle"], 
                    #                        'featurizer' : 'bp_radial',
                    #                        'args' : [{'ATOMIC_NUM_TO_POS' : {6: 0, 1: 1, 7: 2, 8: 3, 9: 4}, 
                    #                                   'r_centers' : np.linspace(0.85, 2, 150)**3, 
                    #                                   'bin_widths' : np.linspace(0.001, 0.02, 150)/25}]
                    #              }, 
                    # 'default_angular' : {'infiles' : ["dataset.named/molconf.nmrshiftdb_hconfspcl_nmrshiftdb.pickle"], 
                    #                      'featurizer' : 'bp_angular',
                    #                      'args' : [DEFAULT_ANGULAR_ARGS], 
                                    
                    # },

                    # 'hconsfspcl_angular' : {'infiles' : ["dataset.named/molconf.nmrshiftdb_hconfspcl_nmrshiftdb.pickle"], 
                    #                      'featurizer' : 'bp_angular',
                    #                      'args' : [HCONFSPCL_ANGULAR_ARGS], 
                                    
                    # },

                    # 'default_bp_images2' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf_nmrshiftdb.pickle"], 
                    #                         'featurizer' : 'bp_images' ,
                    #                         'feature_fields' : ['feature_radial', 'feature_angular'], 
                    #                         'args' : [DEFAULT_BP_IMAGES_ARGS], 
                    # }, 
                    # 'dense_bp_images' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf_nmrshiftdb.pickle"], 
                    #                              'featurizer' : 'bp_images' ,
                    #                              'feature_fields' : ['feature_radial', 'feature_angular'], 
                    #                              'array_out' : True, 
                    #                              'args' : [DENSE_BP_IMAGES_ARGS], 
                    # }, 

                    # 'default_bp_images' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf_nmrshiftdb.pickle"], 
                    #                        'featurizer' : 'bp_images' ,
                    #                        'feature_fields' : ['feature_radial', 'feature_angular'], 
                    #                        'array_out' : True, 
                    #                        'args' : [DEFAULT_BP_IMAGES_ARGS], 
                    # }, 
                    # 'atom_neighborhood_vects' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf_nmrshiftdb.pickle"], 
                    #                              'featurizer' : 'atomic_neighborhood_vectors', 
                    #                              #'feature_fields' : ['feature_radial', 'feature_angular'], 
                    #                              #'array_out' : True, 
                    #                              'args' : [{}]
                    # }, 
    
                    # 'atom_adj_mats' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf_nmrshiftdb.pickle"], 
                    #                    'featurizer' : 'atom_adj_mat', 
                    #                    'array_out' : True, 
                    #                    #'max_run' : 10, 
                    #                    'max_atoms_in_molecule' : 64, 
                    #                    'args' : [{'MAX_ATOM_N' : 64}]
                    # }, 
                    # 'atom_adj_mats_32' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf2_nmrshiftdb.pickle"], 
                    #                    'featurizer' : 'atom_adj_mat', 
                    #                    'array_out' : True, 
                    #                    #'max_run' : 10, 
                    #                    'max_atoms_in_molecule' : 32, 
                    #                    'args' : [{'MAX_ATOM_N' : 32}]
                    # }, 

                    # 'atom_adj_mats_props_32' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf2_nmrshiftdb.pickle"], 
                    #                             'featurizer' : 'adj_mat_props', 
                    #                             'array_out' : True, 
                    #                             'feature_fields' : ['adj', 'props'], 
                    #                             'max_atoms_in_molecule' : 32, 
                    #                             'args' : [{'MAX_ATOM_N' : 32}]
                    # }, 
                    # 'atom_props' : {'infiles' :["dataset.named/molconf.nmrshiftdb_hconf2_nmrshiftdb.pickle"], 
                    #                             'featurizer' : 'advanced_atom_props', 
                    #                             'array_out' : True, 
                    #                             'args' : [{}]
                    # }, 
                    

#                     'default_torch_adj' : {'infiles' : ["nmrshiftdb_data/nmrshiftdb.mol3d_etkdg.pickle"], 
#                                            'featurizer' : 'adjm_neighbors_torch', 
#                                            'args' : [{"MAX_ADJ_POWER" : 2}, 
#                                                      {"MAX_ADJ_POWER" : 3},
#                                                      {"MAX_ADJ_POWER": 4}, 
#                                                      {"MAX_ADJ_POWER": 5}, 
# ], 
#                                          #'max_run' : 1000, 
#                     }, 
}

# def params():
#     mol_field = 'rdmol' # default
#     for exp_name, ec in featurize_params.items():
#         array_out = ec.get('array_out', False)
#         for infile in ec['infiles']:
#             data_filebase = os.path.splitext(os.path.basename(infile))[0] 
#             for featurizer_args_i, featurizer_args in enumerate(ec['args']):
#                 outfile = td(f"{data_filebase}.{exp_name}.{featurizer_args_i}.pickle")
#                 max_run = ec.get('max_run', -1)
#                 feature_fields = ec.get('feature_fields', 'feature')
#                 max_atoms_in_molecule = ec.get("max_atoms_in_molecule", None) 
#                 yield (infile, outfile, mol_field, max_run, feature_fields, 
#                        ec['featurizer'], featurizer_args, array_out , max_atoms_in_molecule)


def bp_radial(mol, conformer_i, **kwargs):
    atomic_nos, coords = atom_features_old.get_nos_coords(mol, conformer_i)
    return atom_features_old.custom_bp_radial(atomic_nos, coords, **kwargs)

# def bp_angular(mol, conformer_i, **kwargs):
#     atomic_nos, coords = atom_features.get_nos_coords(mol, conformer_i)
#     return atom_features.custom_bp_angular(atomic_nos, coords, **kwargs)

# def adjm_neighbors_torch(mol, conformer_i, **kwargs):
#     MAX_ATOM_N = kwargs.get('MAX_ATOM_N', 200)
#     MAX_ADJ_POWER = kwargs.get('MAX_ADJ_POWER', 2)
#     UNIQUE_ATOMIC_NOS = [1, 6, 7, 8, 9, 15, 16, 17]
#     atomic_nums, adj = atom_features.mol_to_nums_adj(mol, MAX_ATOM_N)
#     torch_features = atom_features.torch_featurize_mat_pow(torch.Tensor(atomic_nums), 
#                                                        torch.Tensor(adj), MAX_ATOM_N, 
#                                                        MAX_ADJ_POWER, 
#                                                        UNIQUE_ATOMIC_NOS)
#     return torch_features.numpy()

# def atomic_neighborhood_vectors(mol, conformer_i, **kwargs):
#     atomic_nos, coords = atom_features.get_nos_coords(mol, conformer_i)
#     ATOM_N = len(atomic_nos)
#     features = np.zeros((ATOM_N, ATOM_N-1), 
#                         dtype=[('atomicno', np.uint8), ('pos', np.float32, (3,))])
    
#     res = []
#     for atom_i in range(ATOM_N):
#         vects = coords - coords[atom_i]
#         vm = np.ones(len(vects), np.bool)
#         vm[atom_i] = 0
#         features[atom_i]['atomicno'] = atomic_nos[vm]
#         features[atom_i]['pos'] = vects[vm]
#     return features

# def bp_images(mol, conformer_i, **kwargs):
#     atomic_nos, coords = atom_features.get_nos_coords(mol, conformer_i)

#     rad_f = atom_features.bp_radial_image(atomic_nos, coords, 
#                                           r_bin_edges = kwargs['r_bin_edges'], 
#                                           atomic_num_rows = kwargs['atomic_num_rows'])
#     ang_f = atom_features.bp_angular_images(atomic_nos, coords, 
#                                             r_bin_edges = kwargs['r_bin_edges'], 
#                                             pairings = kwargs['pairings'], 
#                                             theta_bin_edges = kwargs['theta_bin_edges'])

#     return list(zip(rad_f, ang_f))

# def adj_mat_props(mol, conformer_i, **kwargs):
#     a = atom_features.atom_adj_mat(mol, conformer_i, **kwargs)
#     b = atom_features.advanced_atom_props(mol, conformer_i, **kwargs)
#     return list(zip(a, b))


def featurizer_create(name):
    if name == 'bp_radial':
        return bp_radial
    
    raise ValueError(name)

# @mkdir(WORKING_DIR)
# @files(params)
# def featurize_atoms_par(infile, outfile, mol_field, max_run, feature_fields, 
#                         featurizer, featurizer_args, array_out, max_atoms_in_molecule= None):
#     """
#     featurizers that run once per conformer and reutnr one feature per atom
    
#     This is frustrating because it is hard ahead of time to figure out 
#     how many features we will end up with and what their number is, especially
#     since featurization for a given molecule can fail. WE will upper-bound by
#     sum(atomnum * conf*num * mol num)

#     """


#     molecules_df = pickle.load(open(infile, 'rb'))['df']
#     if max_run > 0:
#         molecules_df = molecules_df.iloc[:max_run]
    
#     if max_atoms_in_molecule is not None:
#         molecules_df = molecules_df[molecules_df.rdmol.apply(lambda x : x.GetNumAtoms() <= max_atoms_in_molecule)]

#     FEATURE_NUM = len(molecules_df)
#     if array_out:
#         # output specific npy arrays
#         # upper-limit on num of rows

#         if isinstance(feature_fields, str):
#             feature_field_filenames = [feature_fields]
#         else:
#             feature_field_filenames = feature_fields
#         row_max = 0
#         for row_i, row in molecules_df.iterrows():
#             mol = row[mol_field]
#             confs = mol.GetConformers()

#             row_max += mol.GetNumAtoms() * len(confs)
#         print("There are a max of", row_max, "rows")

#         # assume each feature field is constant-dimensioned and featurize the first molecule
#         res = featurize((0, molecules_df.iloc[0], 
#                          mol_field, feature_fields, featurizer, featurizer_args))
#         array_npy_outs = {}
#         array_npy_outs_filenames = {}
#         for f in feature_field_filenames:
#             first_row = res[0]
#             out_filename = outfile.replace(".pickle", f".{f}.npy")
#             feature = first_row[f]
#             out_shape = [row_max] + list(feature.shape)
#             print("the out shape should be", out_shape, out_filename, feature.dtype)
#             array_npy_outs[f] = np.lib.format.open_memmap(out_filename, mode='w+', 
#                                                           shape = tuple(out_shape), 
#                                                           dtype=feature.dtype)
#             array_npy_outs_filenames[f] = out_filename
#         array_pos = 0

#     # FIXME use pywren at some point

#     THREAD_N = 16
#     pool = Pool(THREAD_N)
#     print("GOING TO MAP")
#     allfeat = []
#     for features_for_mol in tqdm(pool.imap_unordered(featurize, 
#                                                      [(a, b, mol_field, feature_fields, 
#                                                        featurizer, featurizer_args) for a, b in molecules_df.iterrows()]), 
#                                  total=len(molecules_df)):
#         if not array_out:
#             allfeat.append(features_for_mol)
#         else:
#             for feature in features_for_mol:
#                 for f in feature_field_filenames:
#                     array_npy_outs[f][array_pos] = feature[f]
#                     del feature[f] # delete the item 
#                 feature['array_pos'] = array_pos
#                 array_pos += 1
#             allfeat.append(features_for_mol)

#     pool.close()
#     pool.join()

#     df = pd.DataFrame(list(itertools.chain.from_iterable(allfeat)))

#     out_dict = {'df' : df, 
#                 'featurizer' : featurizer, 
#                 'featurizer_args' : featurizer_args}
#     if array_out:
#         out_dict['feature_npy_filenames'] = array_npy_outs_filenames

#     pickle.dump(out_dict, open(outfile, 'wb'))

# MERGE_FEATURE_CONFIGS = {'merged_bp_radial_F_angular' : {'files' : [td("molconf.nmrshiftdb_hconf_nmrshiftdb.default_bp_radial_F.0.pickle"), 
#                                                                     td("molconf.nmrshiftdb_hconf_nmrshiftdb.default_angular.0.pickle")], }
# }


# def merge_params():
#     for mf, mc in MERGE_FEATURE_CONFIGS.items():
#         infiles = mc['files']

#         outfile = td(f"merge.{mf}.pickle")
#         yield infiles, outfile

# @follows(featurize_atoms_par)
# @files(merge_params)
# def merge_features(infiles, outfile):

#     merge_feats = [pickle.load(open(f, 'rb'))['df'] for f in infiles]
#     tgt_meta_cols = ['mol_id', 'conf_i', 'atom_idx' ]
#     a = [f.set_index(tgt_meta_cols).rename(columns={"feature" : i})[i] for i,f in enumerate(merge_feats)]
#     b = pd.concat(a, axis=1)
#     res = []
#     for row_i, row in tqdm(b.iterrows(), total=len(b)):
#         a= np.concatenate([y.flatten() for y in row])
#         res.append(list(row_i) + [a])
#     out_df = pd.DataFrame(res, columns=tgt_meta_cols + ['feature'])
#     pickle.dump({'df' : out_df, 
#                  'featurizer' : 'merge', 
#                  'featurizer_args' : infiles}, 
#                 open(outfile, 'wb'))

DATASET_PRE = {# 'mmff94_64_opt4_p1' : {'filename': "../nmrdata/confs/rdkit_ff.mmff94_64_opt4.conf.rdmol.pickle", 
               #                     "min_p" : 0.1, 
               #                     "top_n" : -1}, 
               
               # 'mmff94_64_opt4_p01' : {'filename': "../nmrdata/confs/rdkit_ff.mmff94_64_opt4.conf.rdmol.pickle", 
               #                     "min_p" : 0.01, 
               #                     "top_n" : -1}, 

               'mmff94_64_opt1000_4_p1' : {'filename': "../nmrdata/confs/rdkit_ff.mmff94_64_opt1000_4.conf.rdmol.pickle", 
                                   "min_p" : 0.1, 
                                   "top_n" : -1}, 
               
               'mmff94_64_opt1000_4_p01' : {'filename': "../nmrdata/confs/rdkit_ff.mmff94_64_opt1000_4.conf.rdmol.pickle", 
                                   "min_p" : 0.01, 
                                   "top_n" : -1}, 

               # 'mmff94_64_opt2_top1' : {'filename': "../nmrdata/confs/rdkit_ff.mmff94_64_opt2.conf.pickle", 
               #                         "top_n" : 1}, 
               }



               

def params():
    for exp_name, ec in DATASET_PRE.items():
        infile_moldf = ec['filename']
        infile_p = ec['filename'].replace(".rdmol.pickle", ".p.feather")
        outfile = td(f"{exp_name}.conf")
        yield (infile_moldf, infile_p), outfile, exp_name, ec

@mkdir(WORKING_DIR)
@files(params) # "../nmrdata/confs/rdkit_ff.mmff94_64_opt2.conf.pickle", td("test.conf"))
def preprocess_data(infiles, outfile, exp_name, ec):
    """
    Generate a table of mol_id, rdmol, tgt_conf_idx
    """
    (infile_moldf, infile_p) = infiles
    mol_df = pickle.load(open(infile_moldf, 'rb'))
    all_conf_e_df = pd.read_feather(infile_p)
    MIN_P = ec.get('min_p', 0.00)
    

    res = []
    for molecule_id, g in all_conf_e_df[all_conf_e_df.p >= MIN_P].groupby('molecule_id'):
        tgt_confs = g.conf_id.astype(int).values
        tgt_probs = g.p.values
        res.append({'molecule_id' : molecule_id, 
                    'conf_idx' : tgt_confs, 'p' : tgt_probs})
    tgt_conf_df = pd.DataFrame(res)

    a = tgt_conf_df.join(mol_df.set_index('molecule_id'), on='molecule_id')
    print(float(len(a)) / len(mol_df) * 100, "%")
    pickle.dump({'df' : a, 
                 'infile_moldf' : infile_moldf, 
                 'infile_p' : infile_p}, 
                 open(outfile, 'wb'))

EXP_PARAMETERS = {#'default_bp_radial' : {'infile' : td("test.conf"),
                  #                       'featurizer' : 'bp_radial',
                  #                       'chunk_size' : 128, 
                  ##                       'mol_limit' : 50000, 
    #'args' : {}, }, 
    
                  'mmff94_64_opt1000_4_p1-default_bp_radial' : {'infile' : td("mmff94_64_opt1000_4_p1.conf"),
                                         'featurizer' : 'bp_radial',
                                         'chunk_size' : 128, 
                                         'mol_limit' : 50000, 
                                         'args' : HCONFSPCL_RADIAL, }, 
                  'mmff94_64_opt1000_4_p01-default_bp_radial' : {'infile' : td("mmff94_64_opt1000_4_p01.conf"),
                                         'featurizer' : 'bp_radial',
                                         'chunk_size' : 128, 
                                         'mol_limit' : 50000, 
                                         'args' : HCONFSPCL_RADIAL,
                  },


                  'mmff94_64_opt1000_4_p01-near_bp_radial' : {'infile' : td("mmff94_64_opt1000_4_p01.conf"),
                                         'featurizer' : 'bp_radial',
                                         'chunk_size' : 128, 
                                         'mol_limit' : 50000, 
                                         'args' : NEAR_RADIAL }, 
}


def featurize_mol(mol_id, rdmol, conf_indices, 
                  featurizer_name,
                  featurizer_args):
    mol_feats = []

    for conf_i in conf_indices:

        features =  featurizer_create(featurizer_name)(rdmol, conf_i, **featurizer_args)
        mol_feats.append(features)
    return {'mol_id' : mol_id, 
            'conf_indices' : conf_indices, 
            'mol_f' : np.stack(mol_feats)}

def featurize_multiple_mols(list_of_params):
    (mol_ids, mols, conf_indices, 
     featurizer_name,
     featurizer_args) = list_of_params

    return list(map(lambda x : featurize_mol(x[0], x[1], x[2], 
                                             featurizer_name,
                                             featurizer_args),
                    zip(mol_ids, mols, conf_indices,)))
    

def params():
    for exp_name, exp_config in EXP_PARAMETERS.items():
        outfile = td(f"features.{exp_name}.wait")
        yield exp_config['infile'], outfile, exp_config, exp_name

import pywren
import pywren.wrenconfig as wrenconfig
from pywren.executor import Executor
import pywren.invokers as invokers

@follows(preprocess_data)
@files(params)
def featurize_start(infile, outfile, exp_config, exp_name, ):

    config = wrenconfig.default()

    invoker = invokers.AWSBatchInvoker()
    job_max_runtime = 3000
    wrenexec= Executor(invoker, config, job_max_runtime)


    CHUNK_SIZE = exp_config['chunk_size']
    MOL_LIMIT = exp_config['mol_limit']
    #feature_func, feature_arg = []
    df = pickle.load(open(infile, 'rb'))['df']
    print(df.head())
    # create the list
    
    arg_list = []
    for df_chunk_i, df_chunk in enumerate(util.split_df(df.iloc[:MOL_LIMIT], CHUNK_SIZE)):
        arg_list.append((
            df_chunk.molecule_id.values, 
            df_chunk.rdmol, 
            df_chunk.conf_idx.values, 
            exp_config['featurizer'], 
            exp_config['args']))
    

    print("There are", len(arg_list), "chunks")
    fs = wrenexec.map(featurize_multiple_mols, arg_list)
    #[f.result() for f in fs]

    #res = list(map(featurize_multiple_mols, arg_list))

    pickle.dump({'futures' : fs, 
                 'exp_config' : exp_config, 
                 'infile' : infile}, 
                open(outfile, 'wb'))


from pywren.wait import wait, ALL_COMPLETED, ANY_COMPLETED, ALWAYS # pylint: disable=unused-import
import psutil
import os
import gc
import plyvel
@transform(featurize_start, 
           suffix(".wait"), 
           ".done")
def featurize_get(infile, outfile):
    d = pickle.load(open(infile, 'rb'))
    futures = d['futures']
    data_infile = d['infile']
    data_df = pickle.load(open(data_infile, 'rb'))['df']
    data_weights_lut = dict(zip(data_df.molecule_id, data_df.p))

    del d['futures']

    process = psutil.Process(os.getpid())
    CHUNK_SIZE = 30
    out_filenames = []

    dir_name = outfile.replace(".done", ".dir")
    shutil.rmtree(dir_name, ignore_errors=True,)
    os.makedirs(dir_name)

    mol_filename_map =[]


    for chunk_i in tqdm(range(int(np.ceil(len(futures)/CHUNK_SIZE))), 
                        desc="chunks of futures"):

        to_get, later = futures[:CHUNK_SIZE], futures[CHUNK_SIZE:]
        fut_done, fut_notdone = wait(to_get, 
                                     return_when=ALL_COMPLETED)
        print(len(fut_done), len(fut_notdone), print("{:3.1f}GB".format(process.memory_info().rss/1e9)))
        futures = later
        del to_get
        gc.collect()

        for f in tqdm(fut_done, desc=f'futures chunk {chunk_i}'):
            
            for single_mol in f.result():
                conf_indices = single_mol['conf_indices']
                conf_n = len(conf_indices)
                mol_f = single_mol['mol_f']
                mol_id = single_mol['mol_id']

                p = data_weights_lut[mol_id]
                p = p / np.sum(p)
                mol_feat = np.average(mol_f, axis=0, weights=p) 
                mol_filename = f"{dir_name}/{mol_id}.npy"
                np.save(mol_filename, mol_feat)

                mol_filename_map.append({'molecule_id' : mol_id, 
                                         'filename' : os.path.relpath(mol_filename)})
                # mol_f.shape

                # # do the averaging 
                
                # with db.write_batch() as wb:
                #     for i, conf_idx in enumerate(conf_indices):
                #         bytes_str = util.np_to_bytes(mol_f[i])
                #         id_str = "{:08d}.{:08d}".format(single_mol['mol_id'], 
                #                                         conf_idx)
                #         wb.put(id_str.encode('ascii'), bytes_str)
        
        # chunk_filename = f"{outfile}.{i:08d}"
        # pickle.dump({'i' : i, 
        #              'results' : [f.result() for f in fut_done]},
        #             open(chunk_filename, 'wb'))
        # out_filenames.append(chunk_filename)
    pickle.dump({'infile' : infile, 
                 'dir_name' : dir_name, 
                 'mol_filename_df' : pd.DataFrame(mol_filename_map), 
    }, 
                open(outfile, 'wb'))
    #res = pywren.get_all_results(futures)
    #pickle.dump(res, open(outfile, 'wb'))



if __name__ == "__main__":
    pipeline_run([preprocess_data, featurize_start, featurize_get]) # , merge_features])
        
