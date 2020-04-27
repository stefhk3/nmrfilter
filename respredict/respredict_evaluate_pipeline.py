"""
Pipeline for generating predictions on code

"""



import numpy as np
import pandas as pd
import seaborn as sns
import torch
import time
import pickle
from netdataio import * 
import os
import metrics

from tqdm import tqdm
import netutil
from ruffus import * 
from glob import glob
import copy

PRED_DIR = "respred.preds"

td = lambda x : os.path.join(PRED_DIR, x)

EXPERIMENTS = {
}    

# good_13C.41783978.00000160.state

# EXPERIMENTS['custom_13c_debug_many'] = {'model' : "checkpoints/13C_orig_13C.59586726",
#                                         'checkpoints' : [400, 600], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_13C.shifts.dataset.pickle"}
# EXPERIMENTS['13c_orig_batchnorm'] = {'model' : "checkpoints/13C_orig_batchnorm_13C.59590207", 
#                                         'checkpoints' : [400, 600], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_13C.shifts.dataset.pickle"}

# EXPERIMENTS['small_lr1e4_111_13C'] = {'model' : "checkpoints/small_lr1e4_111_13C.60193546", 
#                                         'checkpoints' : [500, 600, 700], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_13C.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H'] = {'model' : "checkpoints/default_std_1H.61522528", 
#                                         'checkpoints' : [100, 200, 300, 400, 450], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_nostd'] = {'model' : "checkpoints/bs256_128d_layer8_1H.61570634", 
#                                         'checkpoints' : [100, 200, 300, 400, 500, 600, 700, 800], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_newloss1'] = {'model' : "checkpoints/debug_newloss_backtopower_1H.61646559", 
#                                         'checkpoints' : [100, 200, 300], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_newloss2'] = {'model' : "checkpoints/smaller_scale_log_init1e3_1H.61669468", 
#                                         'checkpoints' : [300, 400, 500, 600], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_newloss3'] = {'model' : "checkpoints/smaller_scale_1H.61661306", 
#                                         'checkpoints' : [600, 800, 900, 1000, 1100, 1200], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_4'] = {'model' : "checkpoints/init01_1H.61695833", 
#                                         'checkpoints' : [500, 600, 700], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_5'] = {'model' : "checkpoints/try_again_1H.61732878", 
#                                         'checkpoints' : [500, 600, 700, 800, 900, 1000, 1200], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}
# EXPERIMENTS['debug_1H_fast1'] = {'model' : "checkpoints/bs32_layer8_128d_1H.61858181", 
#                                         'checkpoints' : [50], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_fast2'] = {'model' : "checkpoints/bs512_layer8_128d_lr4_1H.61859022", 
#                                         'checkpoints' : [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_fast5'] = {'model' : "checkpoints/bs512_layer8_128d_lr3_1H.61859195", 
#                                         'checkpoints' : [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}
# EXPERIMENTS['debug_1H_fast5_later'] = {'model' : "checkpoints/bs512_layer8_128d_lr3_1H.61859195", 
#                                         'checkpoints' : [1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_fast7'] = {'model' : "checkpoints/bs512_layer8_128d_lr4_resnet1x1x128_1H.61871356", 
#                                         'checkpoints' : [2000, 3000, 4000], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}
# EXPERIMENTS['debug_1H_fast9'] = {'model' : "checkpoints/bs512_layer8_128d_lr4_again2_1H.61893260",
#                                         'checkpoints' : [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1500, 2000, 2500, 3000, 3500], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['debug_1H_fast10_bn'] = {'model' : "checkpoints/bs512_layer8_128d_lr3_batchnorm_1H.61895293", 
#                                         'checkpoints' : [1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 
#                                                          2700, 2800, 3000, 3200, 3400, 3600, 3700, 3800], 
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# EXPERIMENTS['bootstrap_default'] = {'model' : "checkpoints/debug_data_subset_1H.62020207", 
#                                         'checkpoints' : [3000, 3400],
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}


# for f in ['bootstrap_1H.62214227', 'debug_bootstrap_1H.256d.62450118', 'debug_bootstrap_1H.512d.62502902', 'bootstrap_1H.newdefault.63912376']:
#     EXPERIMENTS[f"{f}"] = {'model' : f"checkpoints/{f}", 
#                                         'checkpoints' : [2000, 3000, 3500, 3950],
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

# for f in ['bootstrap_13C.62214252', 'debug_bootstrap_13C.256d.62502980', 'bootstrap_13C.newdefault.63912385']: # , 'predstd_13C.62252222']:
#     EXPERIMENTS[f"{f}"] = {'model' : f"checkpoints/{f}", 
#                                         'checkpoints' : [2000, 2500, 3000, 3500, 3950],
#                                         'cv_sets' : [(0,)], 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_13C.shifts.dataset.pickle"}

# EXPERIMENTS["bootstrap_13C_big.64762723"] = {'model' : f"checkpoints/bootstrap_13C_big.64762723", 
#                                              'checkpoints' : [1400],
#                                              'cv_sets' : [(0,1,2,3,4)], 
#                                              'batch_size' : 8, 
#                                              'dataset' : "../nmrdata/processed_data/nmrshiftdb_128_128_HCONFSPCl_13C.shifts.dataset.pickle"}


    

# EXPERIMENTS["bootstrap_13C.baseline.69927537"] = {'model' : f"checkpoints/bootstrap_13C.baseline.69927537", 
#                                              'checkpoints' : [1500, 2000, 2500, 3000],
#                                              'cv_sets' : [(0,)], 
#                                              'batch_size' : 8, 
#                                              'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_13C.shifts.dataset.pickle"}

# EXPERIMENTS["bootstrap_1H.baseline.69927777"] = {'model' : f"checkpoints/bootstrap_1H.baseline.69927777", 
#                                                      'checkpoints' : [1500, 2000, 2500, 3000, 3500, 3950],
#                                                      'cv_sets' : [(0,)], 
#                                                      'batch_size' : 8, 
#                                                      'dataset' : "../nmrdata/processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}
    




# EXPERIMENTS["bootstrap_13C_big.baseline.69928521"] = {'model' : f"checkpoints/bootstrap_13C_big.baseline.69928521",
#                                                       'checkpoints' : [1500, 2000, 2500, 3000, 3950],
#                                                       'cv_sets' : [(0,)], 
#                                                       'batch_size' : 8, 
#                                                       'dataset' : "../nmrdata/processed_data/nmrshiftdb_128_128_HCONFSPCl_13C.shifts.dataset.pickle"}

# EXPERIMENTS["bootstrap_1H_big.baseline.69928569"] = {'model' : f"checkpoints/bootstrap_1H_big.baseline.69928569",
#                                                      'checkpoints' : [1500, 2000, 2500, 3000, 3500, 3950],
#                                                      'cv_sets' : [(0,)], 
#                                                      'batch_size' : 8, 
#                                                      'dataset' : "../nmrdata/processed_data/nmrshiftdb_128_128_HCONFSPCl_1H.shifts.dataset.pickle"}
    
# EXPERIMENTS["bootstrap_1H_debug.a"] = {'model' : f"checkpoints/bootstrap_1H_debug.80222009",
#                                                      'checkpoints' : [2600],
#                                                      'cv_sets' : [(0,)], 
#                                                      'batch_size' : 8, 
#                                                      'dataset' : "../nmrdata/processed_data/nmrshiftdb_128_128_HCONFSPCl_1H.shifts.dataset.pickle"}
    




# EXPERIMENTS["bootstrap_13C_debug_d"] = {'model' : f"checkpoints/bootstrap_13C_debug.config.79962635", 
#                                         'checkpoints' : [2000],
#                                         'cv_sets' : [(0,)], 
#                                         'batch_size' : 8, 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_128_128_HCONFSPCl_13C.shifts.dataset.pickle"}


# EXPERIMENTS["bootstrap_13C_debug_e"] = {'model' : f"checkpoints/bootstrap_13C_debug.layer8_256d_dropout1e2_lr1e3_sched10.80070285",
#                                         'checkpoints' : [800, 1000],
#                                         'cv_sets' : [(0,)], 
#                                         'batch_size' : 8, 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_128_128_HCONFSPCl_13C.shifts.dataset.pickle"}

# EXPERIMENTS["bootstrap_13C_debug_g"] = {'model' : f"checkpoints/bootstrap_13C_debug.bootstrap_13C_debug.layer8_256d_dropout1e2_extraterms.80129056", 
#                                         'checkpoints' : [1000, 1300],
#                                         'cv_sets' : [(0,)], 
#                                         'batch_size' : 8, 
#                                         'dataset' : "../nmrdata/processed_data/nmrshiftdb_128_128_HCONFSPCl_13C.shifts.dataset.pickle"}

EXPERIMENTS["DEBUG_varpred_13C_c"] = {'model' : f"checkpoints/uncertainty_13C_debug.BIG128.81114083",
                                      'checkpoints' : [150, 200],
                                      'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
                                      'batch_size' : 64, 
                                      'dataset' : "processed_data/nmrshiftdb_128_128_HCONFSPCl_13C.shifts.dataset.pickle"}


EXPERIMENTS["DEBUG_varpred_13C_d"] = {'model' : f"checkpoints/uncertainty_13C_debug.big128_baseline.81219345",
                                      'checkpoints' : [500],
                                      'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
                                      'batch_size' : 64, 
                                      'dataset' : "processed_data/nmrshiftdb_128_128_HCONFSPCl_13C.shifts.dataset.pickle"}


EXPERIMENTS["debug_bootstrap_1H_bs32"] = {'model' : f"checkpoints/bootstrap_1H_debug.baseline_bs32.82218381",
                                          'checkpoints' : [2000],
                                          'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
                                          'batch_size' : 64, 
                                          'dataset' : "processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}

EXPERIMENTS["debug_bootstrap_newdata_1H_bs32_b"] = {'model' : f"checkpoints/bootstrap_1H_debug.newdata_baseline_bs32.82678587",
                                          'checkpoints' : [800, 1200],
                                          'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
                                          'batch_size' : 64, 
                                          'dataset' : "processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}


EXPERIMENTS['new_baseline']  = {'model' : f"checkpoints/bootstrap_1H_debug.lr1e4_sched10_95.82890979",
                                'checkpoints' : [1000, 1500, 2000],
                                'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
                                'batch_size' : 64, 
                                'dataset' : "processed_data/nmrshiftdb_64_64_HCONFSPCl_1H.shifts.dataset.pickle"}



EXPERIMENTS["bootstrap_1H_eval2_cv01"] = {'model' : f"checkpoints/bootstrap_1H.cv_0,1.eval2.82988407",
                                          'checkpoints' : [750],
                                          'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (0, 1)}], 
                                          'batch_size' : 64, 
                                          'dataset' : "processed_data/nmrshiftdb_128_128_HCONFSPCl_1H.shifts.dataset.pickle"}

EXPERIMENTS["bootstrap_1H_eval2_cv23"] = {'model' : f"checkpoints/bootstrap_1H.cv_2,3.eval2.82988415", 
                                          'checkpoints' : [750],
                                          'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (2, 3)}], 
                                          'batch_size' : 64, 
                                          'dataset' : "processed_data/nmrshiftdb_128_128_HCONFSPCl_1H.shifts.dataset.pickle"}

EXPERIMENTS["bootstrap_1H_eval2_cv45"] = {'model' : f"checkpoints/bootstrap_1H.cv_4,5.eval2.82988419", 
                                          'checkpoints' : [750],
                                          'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (4, 5)}], 
                                          'batch_size' : 64, 
                                          'dataset' : "processed_data/nmrshiftdb_128_128_HCONFSPCl_1H.shifts.dataset.pickle"}

EXPERIMENTS["bootstrap_1H_eval2_cv67"] = {'model' : f"checkpoints/bootstrap_1H.cv_6,7.eval2.82988427", 
                                          'checkpoints' : [750],
                                          'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (6, 7)}], 
                                          'batch_size' : 64, 
                                          'dataset' : "processed_data/nmrshiftdb_128_128_HCONFSPCl_1H.shifts.dataset.pickle"}

EXPERIMENTS["bootstrap_1H_eval2_cv89"] = {'model' : f"checkpoints/bootstrap_1H.cv_8,9.eval2.82988430", 
                                          'checkpoints' : [750],
                                          'cv_sets' : [{'how' : 'morgan_fingerprint_mod', 'mod' : 10, 'test' : (8, 9)}], 
                                          'batch_size' : 64, 
                                          'dataset' : "processed_data/nmrshiftdb_128_128_HCONFSPCl_1H.shifts.dataset.pickle"}


### Run a bunch of them 
EXPERIMENTS_CV = {'bootstrap_13C.cvtest.83070672' :
               {'model_prefix' : "checkpoints/bootstrap_13C.cvtest.83070672",
                'checkpoints' : [350, 500, 900],
                'cv_sets' : 'from_file',
                'batch_size' : 16,
                'dataset' : "processed_data/nmrshiftdb_128_128_HCONFSPCl_13C.shifts.dataset.pickle"}
               }

EXPERIMENTS_CV['bootstrap_13C.cvtest.83070672_eval64atoms'] = \
    {'model_prefix' : "checkpoints/bootstrap_13C.cvtest.83070672",
     'checkpoints' : [900],
     'cv_sets' : 'from_file',
     'batch_size' : 16,
     'dataset' : "processed_data/nmrshiftdb_64_64_HCONFSPCl_13C.shifts.dataset.pickle"}

               
def params():
    
    for exp_name, exp_config in EXPERIMENTS.items():
        
        outfiles = td(f"{exp_name}.meta.pickle"), td(f"{exp_name}.feather")
        yield None, outfiles, exp_config

    for exp_name, exp_agg_config in EXPERIMENTS_CV.items():
        # extract filename basename
        model_prefix = exp_agg_config['model_prefix']
        model_basename = os.path.basename(model_prefix)
        for meta_filename in glob(exp_agg_config['model_prefix'] + "*.meta"):
            meta_filename_base, _ = os.path.splitext(meta_filename)
            meta_basename = os.path.basename(meta_filename_base)
            
            meta = pickle.load(open(meta_filename, 'rb'))
            cv_set = meta['exp_data']['cv_split']
            exp_config = copy.deepcopy(exp_agg_config)
            exp_config['cv_sets'] = [cv_set]
            exp_config['model'] = meta_filename_base

            # simplify
            common_prefix = os.path.commonprefix([meta_basename, exp_name])
            
            remainder = meta_basename.replace(common_prefix, "")
            
            exp_name_specific = f"{exp_name}{remainder}"

            outfiles = td(f"{exp_name_specific}.meta.pickle"), td(f"{exp_name_specific}.feather")

            
            yield None, outfiles, exp_config
        
USE_CUDA = True

@mkdir(PRED_DIR)
@files(params)
def train_test_predict(infile, outfiles, config):
    
    meta_outfile, data_outfile = outfiles

    model_filename = config['model'] 
    dataset_filename = config['dataset']
    print("config=", config)
    print("loading dataset", dataset_filename)

    all_df = pickle.load(open(config['dataset'], 'rb'))
    if 'debug_max' in config:
        all_df = all_df.sample(config['debug_max'])
                                    

    atoms = np.concatenate([[a.GetSymbol() for a in m.GetAtoms()] for m in all_df.rdmol])

    print("unique atoms", np.unique(atoms))

    meta_filename = f"{model_filename}.meta"

    meta = pickle.load(open(meta_filename, 'rb'))

    allres = []

    tgt_df = all_df
    print(tgt_df.dtypes)
    rdmol_list = tgt_df.rdmol.tolist()
    spect_dict_list = tgt_df.spect_dict.tolist()

    whole_records = tgt_df.to_dict('records')


    metadata_res = []
    for checkpoint_i in config['checkpoints']:
        checkpoint_filename = f"{model_filename}.{checkpoint_i:08d}.model"
        print("running", checkpoint_filename)
        model = netutil.PredModel(meta_filename, 
                                  checkpoint_filename, 
                                  USE_CUDA)
        
        #tgt_df = all_df.copy()

        t1 = time.time()
        results_df = model.pred(rdmol_list, spect_dict_list, whole_records, 
                                prog_bar=True, BATCH_SIZE=config.get("batch_size", 32))
        t2 = time.time()
        print("calculated", len(tgt_df), "mols in ", t2-t1, "sec")
        # merge in molecule_id
        tgt_df_molid = tgt_df.reset_index()[['molecule_id', 
                                             'morgan4_crc32']].copy()
        results_df = results_df.join(tgt_df_molid, on='m_pos')
        results_df['epoch_i'] = checkpoint_i

        data_epoch_outfile = data_outfile + f".{checkpoint_i}"
        results_df.to_feather(data_epoch_outfile)
        metadata_res.append({'time' : t2-t1, 
                             'epoch_filename' : data_epoch_outfile, 
                             'epoch' : checkpoint_i,
                             'mol' : len(tgt_df)})

    metadata_df = pd.DataFrame(metadata_res)
    pickle.dump({'model_filename' : model_filename, 
                 'dataset_filename' : dataset_filename, 
                 'config' : config, 
                 'meta' : metadata_df, },
                open(meta_outfile, 'wb'))
    # data outfile done
    pickle.dump({'data' : True}, 
                open(data_outfile, 'wb'))
                 
@transform(train_test_predict, suffix(".meta.pickle"), 
           ".summary.pickle")
def compute_summary_stats(infile, outfile):
    meta_filename, _ = infile
    a = pickle.load(open(meta_filename, 'rb'))
    meta_df = a['meta']
    exp_config = a['config']

    cv_sets = exp_config['cv_sets']
    
    all_pred_df = []
    for set_idx, cv_i_set_params in enumerate(cv_sets):
        cv_func = netutil.CVSplit(**cv_i_set_params)

        for row_i, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
            pred_df = pd.read_feather(row.epoch_filename)
            
            pred_df['epoch'] = row.epoch
            pred_df['delta'] = pred_df.pred_mu - pred_df.value
            pred_df['delta_abs'] = np.abs(pred_df.delta)
            pred_df['cv_set_i'] = set_idx
            print("pred_df.dtype", pred_df.dtypes)
            pred_df['phase'] =  pred_df.apply(lambda pred_row : \
                                              cv_func.get_phase(None, 
                                                                pred_row.morgan4_crc32), 
                                              axis=1)
            

            all_pred_df.append(pred_df)
    all_pred_df = pd.concat(all_pred_df)

    pred_metrics = all_pred_df.groupby(['nuc', 'cv_set_i', 'phase', 'epoch']).apply(metrics.compute_stats, 
                                                                                    mol_id_field='molecule_id')

    feather_filename = outfile.replace(".pickle", ".feather")
    all_pred_df.reset_index(drop=True).to_feather(feather_filename)

    pickle.dump({'pred_metrics_df' : pred_metrics, 
                 'all_pred_df_filename' : feather_filename, 
                 'infile' : infile, 
                 'exp_config' : exp_config}, 
                open(outfile, 'wb'))


@transform(compute_summary_stats, 
           suffix(".summary.pickle"), 
           ".summary.txt")
def summary_textfile(infile, outfile):
    d = pickle.load(open(infile, 'rb'))
    pred_metrics_df = d['pred_metrics_df']

    with open(outfile, 'w') as fid:
        fid.write(pred_metrics_df.to_string())
        fid.write("\n")
    print("wrote", outfile)

@transform(compute_summary_stats, 
           suffix(".summary.pickle"), 
           ".per_conf_stats.pickle")
def per_confidence_stats(infile, outfile):
    """
    Generate the states broken down by conf
    """
    TGT_THOLDS = [0.1, 0.2, 0.5, 0.9, 0.95, 1.0]

    d = pickle.load(open(infile, 'rb'))
    all_pred_infile = d['all_pred_df_filename']
    df = pd.read_feather(all_pred_infile)
    all_mdf = []
    for (phase, epoch), g in df.groupby(['phase', 'epoch_i']):

        m, tholds, frac_data = metrics.sorted_bin_mean(np.array(g.pred_std), 
                                               np.array(g.delta_abs), 400)


        idx = np.searchsorted(frac_data, TGT_THOLDS)
        mdf = pd.DataFrame({'mae' : m[idx], 'frac_data' : TGT_THOLDS})
        mdf['phase'] = phase
        mdf['epoch'] = epoch
        all_mdf.append(mdf)
    mdf = pd.concat(all_mdf)

    pickle.dump({'infile' : infile, 
                 'all_pred_df_filename' : all_pred_infile, 
                 "df" : mdf}, open(outfile, 'wb'))
    
    r = pd.pivot_table(mdf, index=['phase', 'epoch'], columns=['frac_data'])
    with open(outfile.replace(".pickle", ".txt"), 'w') as fid:
        fid.write(r.to_string())
        fid.write('\n')
    
def per_ppm_stats():
    """

    """
    
if __name__ == "__main__":
    pipeline_run([train_test_predict, 
                  compute_summary_stats, 
                  summary_textfile, per_confidence_stats])
    
