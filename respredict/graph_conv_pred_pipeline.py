"""
Put the model prediction code into its own pipeline
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
import graph_conv_many_nuc_util
from ruffus import * 

PRED_DIR = "preds"

td = lambda x : os.path.join(PRED_DIR, x)

EXPERIMENTS = {
    # '13C_good_latest' : {'model' : "checkpoints/good_13C.41295797", 
    #                      'checkpoints' : [650], 
    #                      'dataset' : "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle"},
    # '1H_good_latest' : {'model' : "checkpoints/good_1H.41343857", 
    #                     'checkpoints' : [ 510], 
    #                     'dataset' : "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle"}, 
    # '13C_good_all' : {'model' : "checkpoints/good_13C.41295797", 
    #                   'checkpoints' : [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650], 
    #                   'dataset' : "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle"},
    # '1H_good_all' : {'model' : "checkpoints/good_1H.41343857", 
    #                  'checkpoints' : [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 510], 
    #                  'dataset' : "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle"}, 

    # 'kwan_1H' : {'model' : "checkpoints/good_1H.41343857", 
    #              'checkpoints' : [200, 300, 400], 
    #              'dataset' : "data/kwan_2015_naturalprod.1H.pickle"}, 
    # 'kwan_13C' : {'model' : "checkpoints/good_13C.41295797", 
    #              'checkpoints' : [200, 300, 400], 
    #              'dataset' : "data/kwan_2015_naturalprod.13C.pickle"}, 
    

    # '13C_good_cv0' : {'model' : "checkpoints/good_13C.41295797", 
    #                   'checkpoints' : [300], 
    #                   'dataset' : "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle"},

    # '13C_good_cv1' : {'model' : "checkpoints/good_13C.41783978",
    #                   'checkpoints' : [160], 
    #                   'dataset' : "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.1.mol_dict.pickle"},

}    



# good_13C = ["good_13C.41800827",
# "good_13C.41801643",
# "good_13C.41801703",
# "good_13C.41801759",
# "good_13C.41295797"]


# good_1H = ['good_1H.41343857',      
# "good_1H.41824366",
# "good_1H.41824376",
# "good_1H.41859989",
# 'good_1H.41908468'
#           ]

# for f in good_13C:
#     meta = pickle.load(open(f"checkpoints/{f}.meta", 'rb'))
#     cv = meta['infile'].split(".")[-3]
#     exp_name = f'new_13C_good_cv{cv}'
#     config = {'model' : f"checkpoints/{f}", 
#               'checkpoints' : [300], 
#               'dataset' : meta['infile']}

#     EXPERIMENTS[exp_name] = config
    
# for f in good_1H:
#     meta = pickle.load(open(f"checkpoints/{f}.meta", 'rb'))
#     cv = meta['infile'].split(".")[-3]
#     exp_name = f'new_1H_good_cv{cv}'
#     config = {'model' : f"checkpoints/{f}", 
#               'checkpoints' : [200, 300], 
#               'dataset' : meta['infile']}

#     EXPERIMENTS[exp_name] = config
    


# good_13C.41783978.00000160.state

EXPERIMENTS['custom_13c_debug_many'] = {'model' : "checkpoints/good_13C.57151192",
                                   'checkpoints' : [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], 
                                   'dataset' : "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.13C.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle"}


EXPERIMENTS['custom_1h_debug_many'] = {'model' : "checkpoints/good_1H.58731147",
                                       'checkpoints' : [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500], 
                                       'dataset' : "graph_conv_many_nuc_pipeline.datasets/graph_conv_many_nuc_pipeline.data.1H.nmrshiftdb_hconfspcl_nmrshiftdb.aromatic.64.0.mol_dict.pickle"}


def params():
    
    for exp_name, exp_config in EXPERIMENTS.items():
        
        outfiles = td(f"{exp_name}.meta.pickle"), td(f"{exp_name}.feather")
        yield None, outfiles, exp_config

@mkdir(PRED_DIR)
@files(params)
def train_test_predict(infile, outfiles, config):
    
    meta_outfile, data_outfile = outfiles

    model_filename = config['model'] 
    dataset_filename = config['dataset']
    print("loading dataset", dataset_filename)
    dataset = pickle.load(open(dataset_filename, 'rb'))
    print("done")

    train_df = dataset['train_df']
    train_df['phase'] = 'train'
    test_df = dataset['test_df']
    test_df['phase'] = 'test'
    all_df = pd.concat([train_df, test_df])



    m = train_df.iloc[0].rdmol
    atoms = np.concatenate([[a.GetSymbol() for a in m.GetAtoms()] for m in train_df.rdmol])

    print("unique atoms", np.unique(atoms))

    meta_filename = f"{model_filename}.meta"


    meta = pickle.load(open(meta_filename, 'rb'))



    allres = []

    tgt_df = all_df.copy()
    rdmol_list = tgt_df.rdmol.tolist()
    value_list = tgt_df.value.tolist()


    USE_CUDA = True
    metadata_res = []
    for checkpoint_i in config['checkpoints']:
        checkpoint_filename = f"{model_filename}.{checkpoint_i:08d}.model"
        print("running", checkpoint_filename)
        model = graph_conv_many_nuc_util.PredModel(meta_filename, checkpoint_filename, 
                                                   USE_CUDA)

        #tgt_df = all_df.copy()

        t1 = time.time()
        results_df = model.pred(rdmol_list, value_list, prog_bar=True)
        t2 = time.time()
        print("calculated", len(tgt_df), "mols in ", t2-t1, "sec")

        # merge in molecule_id
        tgt_df_molid = tgt_df.reset_index()[['molecule_id', 'phase']].copy()
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
                 'meta' : metadata_df, },
                open(meta_outfile, 'wb'))
    # data outfile done
    pickle.dump({'data' : True}, 
                open(data_outfile, 'wb'))
                 
@transform(train_test_predict, suffix(".meta.pickle"), 
           ".summary")
def compute_stats(infile, outfile):
    meta_filename, _ = infile
    a = pickle.load(open(meta_filename, 'rb'))
    meta_df = a['meta']

    all_pred_df = []
    for row_i, row in tqdm(meta_df.iterrows(), total=len(meta_df)):

        pred_df = pd.read_feather(row.epoch_filename)
        pred_df['epoch'] = row.epoch
        pred_df['delta'] = pred_df.pred_mu - pred_df.value
        pred_df['delta_abs'] = np.abs(pred_df.delta)
        all_pred_df.append(pred_df)
    all_pred_df = pd.concat(all_pred_df)

    pred_metrics = all_pred_df.groupby(['nuc', 'phase', 'epoch']).apply(metrics.compute_stats, 
                                                                        mol_id_field='molecule_id')
    with open(outfile, 'w') as fid:
        fid.write(pred_metrics.to_string())

if __name__ == "__main__":
    pipeline_run([train_test_predict, compute_stats])
    
