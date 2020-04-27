import pandas as pd

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
from rdkit import Chem

from glob import glob

import sklearn.ensemble
import sklearn.linear_model
import binascii
#import xgboost
import models
import time
#import nets
import util

WORKING_DIR = "predict.atomic"
td = lambda x: os.path.join(WORKING_DIR, x)

FEATURES_DIR = "features.atomic"
tfeat = lambda x : os.path.join(FEATURES_DIR, x)

@mkdir(WORKING_DIR)
@transform("dataset.named/molconf.*.pickle", formatter(), 
           td("{basename[0]}.subsets.pickle"))
def create_mol_subsets(infile, outfile):
    """
    Create dedicated cross-validation subsets

    Split on canonical smiles string -- this way we can make sure we either train or test
    on all stereoisomers of a given molecule. Since corresponding nuceli in stereoisomers
    will have very similar chemical shifts, this would provide a misleading gain to our 
    performance. 
    
    """
    molecules_df = pickle.load(open(infile, 'rb'))['df']

    def clean_smiles(x):
        try:
            return Chem.MolToSmiles(Chem.RemoveHs(x)) # sometimes we get valence errors
        except:
            return Chem.MolToSmiles(x)

    molecules_df['smiles_canonical'] = molecules_df.rdmol.apply(clean_smiles)

    unique_smiles = np.random.permutation(molecules_df.smiles_canonical.unique())

    unique_smiles_df = pd.DataFrame(unique_smiles, columns=['smiles'])
    unique_smiles_df['subset10_i'] = np.random.permutation(np.arange(len(unique_smiles_df)) % 10)
    unique_smiles_df['subset20_i'] = np.random.permutation(np.arange(len(unique_smiles_df)) % 20)
    unique_smiles_df = unique_smiles_df.set_index('smiles')

    splits_df = molecules_df.join(unique_smiles_df, on='smiles_canonical')
    del splits_df['rdmol']

    pickle.dump({'splits_df' : splits_df, 
                 'infile' : infile},
                open(outfile, 'wb'))


def create_model(model_config):
    
    name = model_config['name'] 
    model_args = model_config.get('model_args', {})
    feature_column = model_config.get('feature_column', 'feature')
    predict_column = model_config.get('predict_column', 'value')

    class_lookup = {'random_forest' : sklearn.ensemble.RandomForestRegressor, 
                    'ridge_regression' : sklearn.linear_model.Ridge, 
                    'rbf_ridge' :models.RBFRidge,  
                    'ff_ridge' : models.FFRidge, 
                    #'xgboost'  :  xgboost.XGBRegressor,
    }
    
    if name in class_lookup:
        c = class_lookup[name]
                    
        m = util.SKLearnAdaptor( c, feature_column, predict_column, model_args)
    elif name == 'nn_ResNetRegression':
        m = util.SKLearnNNAdaptor(nets.ResNetRegression, 
                                  feature_column, predict_column, model_args)
    else:
        raise NotImplementedError(f"{name} is not implemented")

    return m


CANONICAL_FOLD_SETS_20_4 = util.generate_canonical_fold_sets(20, 4)

DATA_CONFIGS = {'hconf_13c_bp' : {'peaks' : "dataset.named/spectra.nmrshiftdb_13c.feather", 
                                  'features' : "features.atomic/molconf.nmrshiftdb_hconf_nmrshiftdb.default_bp_radial.0.pickle", 
                                  'subsets' :  td("molconf.nmrshiftdb_hconf_nmrshiftdb.subsets.pickle")}, 
                'hconf_1H_bp' : {'peaks' : "dataset.named/spectra.nmrshiftdb_1H.feather", 
                                 'features' : "features.atomic/molconf.nmrshiftdb_hconf_nmrshiftdb.default_bp_radial.0.pickle", 
                                 'subsets' :  td("molconf.nmrshiftdb_hconf_nmrshiftdb.subsets.pickle")}, 
                'hconf_13C_bp_F' : {'peaks' : "dataset.named/spectra.nmrshiftdb_13c.feather", 
                                  'features' : "features.atomic/molconf.nmrshiftdb_hconf_nmrshiftdb.default_bp_radial_F.0.pickle", 
                                  'subsets' :  td("molconf.nmrshiftdb_hconf_nmrshiftdb.subsets.pickle")}, 
                'hconf_1H_bp_F' : {'peaks' : "dataset.named/spectra.nmrshiftdb_1H.feather", 
                                 'features' : "features.atomic/molconf.nmrshiftdb_hconf_nmrshiftdb.default_bp_radial_F.0.pickle", 
                                 'subsets' :  td("molconf.nmrshiftdb_hconf_nmrshiftdb.subsets.pickle")}, 
                'hconf_13C_bp_merged' : {'peaks' : "dataset.named/spectra.nmrshiftdb_13c.feather", 
                                         'features' : "features.atomic/merge.merged_bp_radial_F_angular.pickle", 
                                         'subsets' :  td("molconf.nmrshiftdb_hconf_nmrshiftdb.subsets.pickle")}, 
                'hconf_1H_bp_merged' : {'peaks' : "dataset.named/spectra.nmrshiftdb_1H.feather", 
                                 'features' : "features.atomic/merge.merged_bp_radial_F_angular.pickle", 
                                 'subsets' :  td("molconf.nmrshiftdb_hconf_nmrshiftdb.subsets.pickle")}, 
#                 'hconf_13c_bp_radial_F' : {'peaks' : "nmrshiftdb_data/nmrshiftdb.peaks.HCONF.13C.feather",
#                                   'features' : "features.atomic/nmrshiftdb.mol3d_etkdg.default_bp_radial_F.0.pickle", 
#                                   'subsets' :  td("nmrshiftdb.subsets.pickle")}, 
#                 'hconf_13c_manybp' : {'peaks' : "nmrshiftdb_data/nmrshiftdb.peaks.HCONF.13C.feather",
#                                   'features' : "features.atomic/nmrshiftdb.mol3d_etkdg.default_bp_radia*.pickle", 
#                                   'subsets' :  td("nmrshiftdb.subsets.pickle")}, 
#                 'hconf_13c_bp_angular' : {'peaks' : "nmrshiftdb_data/nmrshiftdb.peaks.HCONF.13C.feather",
#                                           'features' : "features.atomic/nmrshiftdb.mol3d_etkdg.default_angular.0.pickle", 
#                                           'subsets' :  td("nmrshiftdb.subsets.pickle")}, 
#                 'hconf_13c_bp_merged' : {'peaks' : "nmrshiftdb_data/nmrshiftdb.peaks.HCONF.13C.feather",
#                                           'features' : "features.atomic/merge.merged_bp_radial_F_angular.pickle", 
#                                           'subsets' :  td("nmrshiftdb.subsets.pickle")}, 
#                 'hconf_13c_torch_adj' : {'peaks' : "nmrshiftdb_data/nmrshiftdb.peaks.HCONF.13C.feather",
#                                           'features' : "features.atomic/nmrshiftdb.mol3d_etkdg.default_torch_adj.*.pickle", 
#                                           'subsets' :  td("nmrshiftdb.subsets.pickle")}, 
#                 'hconf_13c_bp_merged_cutoffs' : {'peaks' : "nmrshiftdb_data/nmrshiftdb.peaks.HCONF.13C.feather",
#                                                  'features' : "features.atomic/merge.merged_bp_radial_F_?_angular.pickle", 
#                                                  'subsets' :  td("nmrshiftdb.subsets.pickle")}, 


            
}

EXPERIMENTS = {'rf_basic' : {'data_config' : 'hconf_13c_bp', 
                             'model_config' :  {'name' : 'random_forest', 
                                                'feature_column' : 'feature', 
                                                'predict_column' : 'value', 
                                                'model_args' : [{'n_jobs' : 20,'n_estimators' : 20}, 
                                                                {'n_jobs' : 5,'n_estimators' : 5}, 
                                                                {'n_jobs' : 20,'n_estimators' : 100}], 
                             }, 
                             'cv_sets' : list(range(5))}, 
               'rf_basic_1H' : {'data_config' : 'hconf_1H_bp', 
                                'model_config' :  {'name' : 'random_forest', 
                                                   'feature_column' : 'feature', 
                                                   'predict_column' : 'value', 
                                                   'model_args' : [{'n_jobs' : 20,'n_estimators' : 20}, 
                                                                   {'n_jobs' : 5,'n_estimators' : 5}, 
                                                                   {'n_jobs' : 20,'n_estimators' : 100}], 
                                }, 
                                'cv_sets' : list(range(5))}, 


               'rf_basic_13C' : {'data_config' : 'hconf_13C_bp_F', 
                                 'model_config' :  {'name' : 'random_forest', 
                                                    'feature_column' : 'feature', 
                                                    'predict_column' : 'value', 
                                                    'model_args' : [{'n_jobs' : 20,'n_estimators' : 20}, 
                                                                    #{'n_jobs' : 5,'n_estimators' : 5}, 
                                                                    #{'n_jobs' : 20,'n_estimators' : 100}, 
                                                    ], 
                                 }, 
                                 'cv_sets' : list(range(5))}, 
               'rf_basic_1H' : {'data_config' : 'hconf_1H_bp_F', 
                                'model_config' :  {'name' : 'random_forest', 
                                                   'feature_column' : 'feature', 
                                                   'predict_column' : 'value', 
                                                   'model_args' : [{'n_jobs' : 20,'n_estimators' : 20}, 
                                                                   #{'n_jobs' : 5,'n_estimators' : 5}, 
                                                                   #{'n_jobs' : 20,'n_estimators' : 100},
                                                   ], 
                                }, 
                                'cv_sets' : list(range(5))}, 


               'rf_basic_13C_merged2' : {'data_config' : 'hconf_13C_bp_merged', 
                                 'model_config' :  {'name' : 'random_forest', 
                                                    'feature_column' : 'feature', 
                                                    'predict_column' : 'value', 
                                                    'model_args' : [{'n_jobs' : 20,'n_estimators' : 20}, 
                                                                    {'n_jobs' : 5,'n_estimators' : 5}, 
                                                                    {'n_jobs' : 20,'n_estimators' : 100}, 
                                                                    {'n_jobs' : 20,'n_estimators' : 250}, 
                                                    ], 
                                 }, 
                                 'cv_sets' : list(range(5))}, 
               'rf_basic_1H_merged2' : {'data_config' : 'hconf_1H_bp_merged', 
                                'model_config' :  {'name' : 'random_forest', 
                                                   'feature_column' : 'feature', 
                                                   'predict_column' : 'value', 
                                                   'model_args' : [{'n_jobs' : 20,'n_estimators' : 20}, 
                                                                   {'n_jobs' : 5,'n_estimators' : 5}, 
                                                                   {'n_jobs' : 20,'n_estimators' : 100},
                                                                   {'n_jobs' : 20,'n_estimators' : 250},
                                                   ], 
                                }, 
                                'cv_sets' : list(range(5))}, 

               
#                'rf_fast' : {'data_config' : 'hconf_13c_manybp', 
#                             'model_config' :  {'name' : 'random_forest', 
#                                                'feature_column' : 'feature', 
#                                                'predict_column' : 'value', 
#                                                'model_args' : [{'n_jobs' : 5,'n_estimators' : 5}]
                                                               
#                                                }, 
#                              'cv_sets' : list(range(5))}, 

#                'rf_32' : {'data_config' : 'hconf_13c_manybp', 
#                             'model_config' :  {'name' : 'random_forest', 
#                                                'feature_column' : 'feature', 
#                                                'predict_column' : 'value', 
#                                                'model_args' : [{'n_jobs' : 32,'n_estimators' : 32}]
                                                               
#                                                }, 
#                              'cv_sets' : list(range(5))}, 

#                'ridge1' : {'data_config' : 'hconf_13c_bp', 
#                           'model_config' :  {'name' : 'ridge_regression', 
#                                                'feature_column' : 'feature', 
#                                                'predict_column' : 'value', 
#                                                'model_args' : [{'alpha' : x} for x in np.logspace(-1, 3, 5)]
#                                                }, 
#                              'cv_sets' : list(range(5))}, 
               'rbf_ridge_merge_1' : {'data_config' : 'hconf_13C_bp_merged', 
                                      'model_config' :  {'name' : 'rbf_ridge', 
                                                         'feature_column' : 'feature', 
                                                         'predict_column' : 'value', 
                                                         'model_args' : [
                                                             {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.0002}, 
                                                             {'n_components' : 4096*4, 'alpha' : 0.00002, 'gamma' : 0.0002}, 
                                                             {'n_components' : 4096*4, 'alpha' : 0.00004, 'gamma' : 0.0002}, 
                                                             {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.0001}, 
                                                             {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.0004}, 
                                                             {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.0004, 'normalize' : False}, 
                                                             {'n_components' : 1024*16, 'alpha' : 0.0001, 'gamma' : 0.00001, 'normalize' : False}, 
                                                             {'n_components' : 1024*16, 'alpha' : 0.00001, 'gamma' : 0.0002, 'normalize' : False}, 
                                                             {'n_components' : 1024*16, 'alpha' : 0.000005, 'gamma' : 0.0002, 'normalize' : False}, 
                                                             
                                                         ], 
                                                         
                                      }, 
                                      'cv_sets' : list(range(1))}, 
               'rbf_ridge_merge_normalize_1' : {'data_config' : 'hconf_13C_bp_merged', 
                                                'model_config' :  {'name' : 'rbf_ridge', 
                                                                   'feature_column' : 'feature', 
                                                                   'predict_column' : 'value', 
                                                                   'model_args' : [
                                                                       {'n_components' : 1024, 'alpha' : 0.00001, 'gamma' : 0.0002}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.00001, 'gamma' : 0.0001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.00001, 'gamma' : 0.0004}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.0001, 'gamma' : 0.0001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.000001, 'gamma' : 0.0001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.000001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.000001, 'gamma' : 0.000001},  
                                                                       {'n_components' : 1024, 'alpha' : 0.00001,  'gamma' : 0.000001},  
                                                                       {'n_components' : 1024, 'alpha' : 0.0001,   'gamma' : 0.000001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.00001,  'gamma' : 0.000001}, 
                                                                       {'n_components' : 1024, 'alpha' : 1e-7,     'gamma' : 0.000001},
                                                                       {'n_components' : 1024, 'alpha' : 1e-7,     'gamma' : 1e-7}, 
                                                                       {'n_components' : 1024, 'alpha' : 1e-6,     'gamma' : 1e-7}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.000001, 'gamma' : 1e-7}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.00001,  'gamma' : 1e-7}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.0001,   'gamma' : 1e-7}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.001,    'gamma' : 1e-7}, 
                                                                       {'n_components' : 1024, 'alpha' : 1e-8,     'gamma' : 0.000001}, 
                                                                       {'n_components' : 1024, 'alpha' : 1e-8,     'gamma' : 1e-7}, 
                                                                       {'n_components' : 1024, 'alpha' : 1e-8,     'gamma' : 0.1}, # 20
                                                                       {'n_components' : 1024, 'alpha' : 1e-8,     'gamma' : 0.1}, 
                                                                       {'n_components' : 1024, 'alpha' : 1e-4,     'gamma' : 0.1}, 
                                                                       {'n_components' : 1024, 'alpha' : 1e-4,     'gamma' : 0.1}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.00000001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.0000001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.000001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.0001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024, 'alpha' : 0.001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 4096, 'alpha' : 0.00000001, 'gamma' : 0.00001}, # 30 
                                                                       {'n_components' : 4096, 'alpha' : 0.0000001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 4096, 'alpha' : 0.000001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 4096, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 4096, 'alpha' : 0.001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 4096, 'alpha' : 0.01, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 8192, 'alpha' : 0.000001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 8192, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024*16, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
                                                                       {'n_components' : 1024*16, 'alpha' : 0.0001, 'gamma' : 0.00001}, # 40
                                                                       {'n_components' : 1024*16, 'alpha' : 0.001, 'gamma' : 0.00001}, 
                                                                       # {'n_components' : 1024*16, 'alpha' : 0.000001, 'gamma' : 0.00001}, 
                                                                       # {'n_components' : 1024*16, 'alpha' : 0.00001, 'gamma' : 0.00005}, 
                                                                       # {'n_components' : 1024*16, 'alpha' : 0.00001, 'gamma' : 0.0001}, 

 

                                                                       
                                                                 ], 
                                                                
                                      }, 
                                            'cv_sets' : list(range(1))}, 
               

#                'rbf_ridge_angular_1' : {'data_config' : 'hconf_13c_bp_angular', 
#                                         'model_config' :  {'name' : 'rbf_ridge', 
#                                                            'feature_column' : 'feature', 
#                                                            'predict_column' : 'value', 
#                                                            'model_args' : [{'n_components' : 2048, 'alpha' : 0.001, 'gamma' : 0.002}, 
#                                                                            {'n_components' : 1024*4, 'alpha' : 0.001, 'gamma' : 0.002}, 
#                                                                            {'n_components' : 1024*16, 'alpha' : 0.001, 'gamma' : 0.002}, 
#                                                                            {'n_components' : 1024*16, 'alpha' : 0.0005, 'gamma' : 0.002}, 
#                                                                            {'n_components' : 1024*16, 'alpha' : 0.002, 'gamma' : 0.002}, 
# ], 
#                                         }, 
#                                         'cv_sets' : list(range(1))}, 

#                'rbf_ridge_merged' : {'data_config' : 'hconf_13c_bp_merged', 
#                                         'model_config' :  {'name' : 'rbf_ridge', 
#                                                            'feature_column' : 'feature', 
#                                                            'predict_column' : 'value', 
#                                                            'model_args' : [{'n_components' : 4096, 'alpha' : 0.01, 'gamma' : 0.002}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.001, 'gamma' : 0.002}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.002}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.01, 'gamma' : 0.001}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.001, 'gamma' : 0.001}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.001}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.01, 'gamma' : 0.0005}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.001, 'gamma' : 0.0005}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.0005},  
#                                                                            {'n_components' : 4096, 'alpha' : 0.01, 'gamma' : 0.00001}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.001, 'gamma' : 0.00001}, 
#                                                                            {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.00001}, 
                                                                       
# ], 
#                                         }, 
#                                         'cv_sets' : list(range(1))}, 
#                'rbf_ridge_merged1' : {'data_config' : 'hconf_13c_bp_merged', 
#                                         'model_config' :  {'name' : 'rbf_ridge', 
#                                                            'feature_column' : 'feature', 
#                                                            'predict_column' : 'value', 
#                                                            'model_args' : [
#                                                                {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.00001}, 
#                                                                {'n_components' : 4096, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
#                                                                {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.000001}, 
#                                                                {'n_components' : 4096, 'alpha' : 0.00001, 'gamma' : 0.000001}, 
#                                                                {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
#                                                                {'n_components' : 4096*4, 'alpha' : 0.0001, 'gamma' : 0.00001}, 
#                                                                {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.0001}, 
#                                                                {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
#                                                                {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.0002}, 
#                                                                {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.00005}, 

# ], 
#                                         }, 
#                                         'cv_sets' : list(range(5))}, 
#                'xgb_basic' : {'data_config' : 'hconf_13c_bp', 
#                               'model_config' :  {'name' : 'xgboost', 
#                                                  'feature_column' : 'feature', 
#                                                  'predict_column' : 'value', 
#                                                  'model_args' : [{}, 
#                                                                  {'booster' : 'gblinear', 
#                                                                   'silent' : False}, 
#                                                                  {'max_depth' : 5}, 
#                                                                  {'max_depth' : 7}, 
#                                                                  {'max_depth' : 9}, 
#                                                                  {'max_depth' : 11}, 
#                                                                  {'max_depth' : 13}, 
# ]
#                               }, 
#                             'cv_sets' : list(range(5))}, 
#                'xgb_merged' : {'data_config' : 'hconf_13c_bp_merged', 
#                               'model_config' :  {'name' : 'xgboost', 
#                                                  'feature_column' : 'feature', 
#                                                  'predict_column' : 'value', 
#                                                  'model_args' : [
#                                                                  {'max_depth' : 13},]
#                               }, 
#                             'cv_sets' : list(range(5))}, 
#                'rf_32_merged' : {'data_config' : 'hconf_13c_bp_merged', 
#                                  'model_config' :  {'name' : 'random_forest', 
#                                                     'feature_column' : 'feature', 
#                                                     'predict_column' : 'value', 
#                                                     'model_args' : [{'n_jobs' : 32,'n_estimators' : 32}]
                                                    
#                                  }, 
#                                  'cv_sets' : list(range(5))}, 

#                'rf_110_merged' : {'data_config' : 'hconf_13c_bp_merged', 
#                                  'model_config' :  {'name' : 'random_forest', 
#                                                     'feature_column' : 'feature', 
#                                                     'predict_column' : 'value', 
#                                                     'model_args' : [{'n_jobs' : 32,'n_estimators' : 110}]
                                                    
#                                  }, 
#                                  'cv_sets' : list(range(5))}, 
#                'nn_debug' : {'data_config' : 'hconf_13c_bp_radial_F', 
#                              'model_config' :  {'name' : 'nn_ResNetRegression', 
#                                                 'model_args' : [{'batch_size' : 256, 
#                                                                  'module__D' : 150, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 6, 
#                                                                  'criterion' : 'SmoothL1Loss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 10}, 
#                                                                 {'batch_size' : 1024, 
#                                                                  'module__D' : 150, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 6, 
#                                                                  'criterion' : 'SmoothL1Loss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 100}]
#                                  }, 
#                                  'cv_sets' : list(range(1))}, 
#                'nn_merged_debug' : {'data_config' : 'hconf_13c_bp_merged', 
#                              'model_config' :  {'name' : 'nn_ResNetRegression', 
#                                                 'model_args' : [{'batch_size' : 256,   # 0
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 6, 
#                                                                  'criterion' : 'SmoothL1Loss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 10}, 
#                                                                 {'batch_size' : 256,  # 1
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 6, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 10}, 
#                                                                 {'batch_size' : 256,  # 2
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 6, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256,  # 3
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 256, 
#                                                                  'module__int_layer_n' : 6, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256,  # 4
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 8, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256,  # 5
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 8, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256,  # 6 
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 256, 
#                                                                  'module__int_layer_n' : 8, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256,  #7
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 256, 
#                                                                  'module__int_layer_n' : 8, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-2, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256,  # 8 
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 256, 
#                                                                  'module__int_layer_n' : 4, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256, # 9
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 4, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256,  # 10
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 10, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 200}, 
#                                                                 {'batch_size' : 256, # 11
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 10, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 100}, 
#                                                                 {'batch_size' : 256,  # 12
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 10, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 150}, 
#                                                                 {'batch_size' : 256, #13 
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 12, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 150}, 
#                                                                 {'batch_size' : 256, # 14
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 10, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 200}, 
#                                                                 {'batch_size' : 256, # 15
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 12, 
#                                                                  'criterion' : 'SmoothL1Loss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 150}, 
#                                                                 {'batch_size' : 256,  # 16
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 9, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 200}, 
#                                                                 {'batch_size' : 256,  # 17
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 11, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-4, 
#                                                                  'max_epochs' : 200}, 
#                                                                 {'batch_size' : 256,  # 18
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 9, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 200}, 
#                                                                 {'batch_size' : 256,  # 19
#                                                                  'module__D' : 726, 
#                                                                  'module__INT_D' : 128, 
#                                                                  'module__int_layer_n' : 11, 
#                                                                  'criterion' : 'MSELoss', 
#                                                                  'optimizer' : 'Adam', 
#                                                                  'lr': 1e-3, 
#                                                                  'max_epochs' : 200}, 

#                                                 ]
#                                  }, 
#                                  'cv_sets' : list(range(1))}, 

#                'rf_32_torch_adj' : {'data_config' : 'hconf_13c_torch_adj', 
#                                     'model_config' :  {'name' : 'random_forest', 
#                                                        'feature_column' : 'feature', 
#                                                        'predict_column' : 'value', 
#                                                        'model_args' : [{'n_jobs' : 32,'n_estimators' : 32}, 
#                                                                     {'n_jobs' : 32,'n_estimators' : 100}, 
#                                                                     {'n_jobs' : 32,'n_estimators' : 200}, 
#                                                        ]
                                                       
#                                     }, 
#                                     'cv_sets' : list(range(5))}, 
#                'rbf_ridge_torch_adj' : {'data_config' : 'hconf_13c_torch_adj', 
#                                         'model_config' :  {'name' : 'rbf_ridge', 
#                                                            'feature_column' : 'feature', 
#                                                            'predict_column' : 'value', 
#                                                            'model_args' : [
#                                                                {'n_components' : 2048, 'alpha' : 0.0001, 'gamma' : 0.00001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.0001, 'gamma' : 0.0001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.0001, 'gamma' : 0.001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.0001, 'gamma' : 0.01}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.0001, 'gamma' : 0.1}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.001, 'gamma' : 0.00001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.001, 'gamma' : 0.0001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.001, 'gamma' : 0.001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.001, 'gamma' : 0.01}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.001, 'gamma' : 0.1}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.01, 'gamma' : 0.00001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.01, 'gamma' : 0.0001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.01, 'gamma' : 0.001}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.01, 'gamma' : 0.01}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.01, 'gamma' : 0.1}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.1, 'gamma' : 0.01}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.1, 'gamma' : 0.02}, 
#                                                                {'n_components' : 2048, 'alpha' : 0.1, 'gamma' : 0.005}, 
#                                                                {'n_components' : 2048, 'alpha' : 1.0, 'gamma' : 0.01}, 
#                                                                {'n_components' : 2048, 'alpha' : 1.0, 'gamma' : 0.02}, 
#                                                                {'n_components' : 2048, 'alpha' : 1.0, 'gamma' : 0.005}, 
#                                                                ]
#                                         }, 
#                                         'cv_sets' : list(range(5))}, 

#                'xgb_torch_adj' : {'data_config' : 'hconf_13c_torch_adj', 
#                               'model_config' :  {'name' : 'xgboost', 
#                                                  'feature_column' : 'feature', 
#                                                  'predict_column' : 'value', 
#                                                  'model_args' : [{}, 
#                                                                  {'max_depth' : 9}, 
#                                                                  {'max_depth' : 11}, 
#                                                                  {'max_depth' : 13}, 
#                                                  ]
#                               }, 
#                             'cv_sets' : list(range(5))}, 

#                'rbf_ridge_merged_cutoffs' : {'data_config' : 'hconf_13c_bp_merged_cutoffs', 
#                                              'model_config' :  {'name' : 'rbf_ridge', 
#                                                                 'feature_column' : 'feature', 
#                                                                 'predict_column' : 'value', 
#                                                                 'model_args' : [
#                                                                     {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.00001}, 
#                                                                     {'n_components' : 4096, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
#                                                                     {'n_components' : 4096, 'alpha' : 0.0001, 'gamma' : 0.000001}, 
#                                                                     {'n_components' : 4096, 'alpha' : 0.00001, 'gamma' : 0.000001}, 
#                                                                     {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
#                                                                     {'n_components' : 4096*4, 'alpha' : 0.0001, 'gamma' : 0.00001}, 
#                                                                     {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.0001}, 
#                                                                     {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.00001}, 
#                                                                     {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.0002}, 
#                                                                     {'n_components' : 4096*4, 'alpha' : 0.00001, 'gamma' : 0.00005}, 
                                                                    
#                                                                 ]
#                                              },
#                                              'cv_sets' : list(range(5))}, 



#   }  

}


# import binascii

def exp_params():
    for exp_name, ec in EXPERIMENTS.items():
        model_configs = ec['model_config']
        data_config = DATA_CONFIGS[ec['data_config']]
        subsets = data_config['subsets']
        exp_num = 0
        for peaks in glob(data_config['peaks']):
            for features in glob(data_config['features']):
                infiles = (peaks, features, subsets)
                
                
                crc = binascii.crc32((peaks + ":" + features + ":").encode('utf-8'))
                input_hash_str = "{:08x}".format(crc)
                
                for model_config_i, model_config in enumerate(util.dict_product(model_configs)):
                
    # should we hash input files to get output or something
                    outfile = td(f"{exp_name}.{input_hash_str}.{model_config_i}.pickle")

                    yield infiles, outfile, model_config, CANONICAL_FOLD_SETS_20_4[ec['cv_sets']]
                exp_num += 1
        if exp_num == 0:
            raise ValueError("exp configuration did not yield any experiments")


@follows(create_mol_subsets)
@files(exp_params)
def train_exps(infiles, outfile, model_config, cv_sets):
    """
    We are going to make the assumption that all of our methods are 
    parallel / capable of using all the cores on our machine and thus 
    we can do cv serially
    """

    outfile_base = os.path.splitext(outfile)[0]
    spectra_filename, features_filename, mol_subset_filename = infiles
    
    
    spectra_df = pd.read_feather(spectra_filename).rename(columns={'id' : 'peak_id'})
    features_df = pickle.load(open(features_filename, 'rb'))['df']

    # FIXME drop conformations except first
    features_df = features_df[features_df.conf_i == 0].set_index(['mol_id', 'atom_idx'])

    mol_subsets = pickle.load(open(mol_subset_filename, 'rb'))['splits_df']


    #spectra_df = spectra_df[['molecule', 'spectrum_id', 'atom_idx', 'value']]
    #with_features_df = spectra_df.join(features_df, on =['molecule', 'atom_idx']).dropna()

    res = pd.merge(features_df, spectra_df, left_on=('mol_id', 'atom_idx'), right_on=('molecule_id', 'atom_idx'))
    with_features_df = res[['molecule_id', 'conf_i', 'spectrum_id', 'atom_id', 'atom_idx', 'value', 'feature' ]]

    res = []
    for cv_i, cv_mol_subset in enumerate(tqdm(cv_sets)):
        t1 = time.time()
        train_test_split = mol_subsets.subset20_i.isin(cv_mol_subset)
        train_mols = mol_subsets[~train_test_split].index.values
        test_mols = mol_subsets[train_test_split].index.values

        
        train_df = with_features_df[with_features_df.molecule_id.isin(train_mols)]
        test_df = with_features_df[with_features_df.molecule_id.isin(test_mols)]
        #print(len(train_df), len(test_df))
        m = create_model(model_config)
        m.fit(train_df)
        train_est = m.predict(train_df)
        test_est = m.predict(test_df)
        
        model_filename = f"{outfile_base}.{cv_i}.model"
        est_filename = f"{outfile_base}.{cv_i}.est"

        pickle.dump({'m' : m, 
                     'train_df' : train_df, 
                     'test_df' : test_df}, 
                    open(model_filename, 'wb'), -1)
        pickle.dump({'train_est' : train_est, 
                     'test_est' : test_est}, 
                    open(est_filename, 'wb'), -1)
        t2 = time.time()
        res.append({'cv_i' : cv_i, 
                    'cv_mol_subset' : cv_mol_subset, 
                    'model_filename' :  os.path.abspath(model_filename), 
                    'runtime' : t2-t1,
                    'est_filename' :  os.path.abspath(est_filename)})
    pickle.dump({'res' : pd.DataFrame(res),
                 'spectra_filename' : os.path.abspath(spectra_filename), 
                 'features_filename' : os.path.abspath(features_filename), 
                 'model_config' : model_config, 
                 'mol_subset_filename' : os.path.abspath(mol_subset_filename)}, 
                open(outfile, 'wb'), -1)

@transform(train_exps, suffix(".pickle"), 
           '.summary.feather')
def summarize(infile, outfile):
    """
    Aggregate everything across folds for a given configuration
    into a single dataframe, split into ['train', 'test'] phases
    """

    a = pickle.load(open(infile, 'rb'))

    estimates = []
    for row_i, row in tqdm(a['res'].iterrows(), total=len(a['res'])):
        model = pickle.load(open(row.model_filename, 'rb'))
        est = pickle.load(open(row.est_filename, 'rb'))

        res = []

        for phase in ['train', 'test']:
            est_df = est[f'{phase}_est']
            data_df = model[f'{phase}_df']

            if 'feature' in data_df:
                del data_df['feature']
            data_df['est'] = est_df.est
            data_df['phase'] = phase
            data_df['cv_i'] = row.cv_i
            estimates.append(data_df)
        
    est_df = pd.concat(estimates).reset_index()
    est_df.to_feather(outfile)    

if __name__ == "__main__":
    print("Running pipeline") 
    pipeline_run([create_mol_subsets]) # , train_exps, summarize])# , train_exps, summarize])


