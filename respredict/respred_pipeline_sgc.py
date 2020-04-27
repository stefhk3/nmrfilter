"""
Pipeline for simple graph convolutions

"""

import numpy as np
import pickle
import pandas as pd
from ruffus import * 
from tqdm import  tqdm
from rdkit import Chem
import pickle
import os

from glob import glob
import json 

import time
import util
import nets

from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdMD

import torch
from torch import nn
from tensorboardX import SummaryWriter

from tqdm import  tqdm
from netdataio import * 
import netdataio
import itertools
import graph_conv_many_nuc_util
from graph_conv_many_nuc_util import move
import sys

DATASET_DIR = "../nmrdata/processed_data/"


td = lambda x : os.path.join(DATASET_DIR, x)


SPECT_SET = '13C' # or '13C'

TENSORBOARD_DIR = f"logs.{SPECT_SET}"

CHECKPOINT_DIR = "checkpoints" 
CHECKPOINT_EVERY = 50
DATASET_NAME = f'nmrshiftdb_64_64_HCONFSPCl_{SPECT_SET}'
#DATASET_NAME = 'qm9'

if SPECT_SET == '13C':
    NUC_LOSS_SCALE = {'13C' : 1.0} 
    NUC_STD_SCALE = {'13C' : 1.0} 
    STD_REGULARIZE = 0.1
else:
    NUC_LOSS_SCALE = {'1H' : 1.0/20.0}
    NUC_STD_SCALE = {'1H' : 10.0} 
    STD_REGULARIZE = 0.01

tgt_max_n = 64 
MAX_EPOCHS = 10000

extra_exp_name = sys.argv[1]
EXP_NAME = f"{extra_exp_name}_{SPECT_SET}"

def create_uncertain_validate_func(tgt_nucs, writer):
    def val_func(res, prefix, epoch_i): # val, mask, truth):
        mu = res['pred_mu']
        std = res['pred_std'] 
        mask = res['pred_mask']
        truth = res['pred_truth']
        mean_loss = res['mean_loss']
        res = {'mean_loss' : mean_loss, 
               'run_epoch_time' : res['runtime'], 
               'run_efficinecy' : res['run_efficiency'], 
               'run_pts_per_sec' : res['pts_per_sec']}
        for ni, n in enumerate(tgt_nucs):
            delta = (mu[:, :, ni] - truth[:, :, ni])[mask[:, :, ni] > 0].flatten()
            masked_std = (std[:, :, ni])[mask[:, :, ni] > 0].flatten()
            res[f"{n}/delta_std"] = np.std(delta)
            res[f"{n}/delta_max"] = np.max(np.abs(delta))
            res[f"{n}/delta_mean_abs"] = np.mean(np.abs(delta))
            res[f"{n}/delta_abs_90"] = np.percentile(np.abs(delta), 90)
            res[f"{n}/std/mean"] = np.mean(masked_std)
            res[f"{n}/std/min"] = np.min(masked_std)
            res[f"{n}/std/max"] = np.max(masked_std)
            writer.add_histogram(f"{prefix}{n}_delta_abs", 
                                 np.abs(delta), epoch_i)
            writer.add_histogram(f"{prefix}{n}_delta_abs_dB", 
                                 np.log10(np.abs(delta)+1e-6), epoch_i)

        for metric_name, metric_val in res.items():
            writer.add_scalar("{}{}".format(prefix, metric_name), 
                              metric_val, epoch_i)


    return val_func



USE_STD = False

net_params_base = {'init_noise' : 0.0, 
                   'resnet' : True, 
                   'int_d' : 256, 
                   'GS' : 6, 
                   'agg_func' : 'goodmax', 
                   'force_lin_init' : True, 
                   'g_feature_n' : -1, 
                   'resnet_out' : True, 
                   'out_std' : USE_STD, 
                   'batchnorm' : False, 
                   'input_batchnorm' : False, 
                   'graph_dropout' : 0.0, 
                   'resnet_blocks' : (1,1,1),
                   'resnet_d': 128, 
                   'out_std_exp' : False, 
                   'OUT_DIM' : 1, # update
                   'gml_nonlin' : 'relu', 
}

opt_params_base = {'optimizer' : 'adam', 
                   'amsgrad' : True, 
                   'lr' : 1e-4, 
                   'weight_decay' : 0, 
                   'scheduler_gamma' : 0.95, 
                   'eps' : 1e-8, 
                   'scheduler_step_size' : 10}


def params():
    CV_I = int(os.environ.get("CV_I", 0))
    data_file = td("{}.shifts.dataset.pickle".format(DATASET_NAME))
    meta_file = td("{}.meta.pickle".format(DATASET_NAME))
    
    seed = 1234                                                                                              

    outfile = f"respredict_pipeline_{SPECT_SET}.{seed}.{CV_I}.out"
    


    yield (meta_file, data_file), outfile, seed, CV_I
1
@mkdir(CHECKPOINT_DIR)
@files(params)
def train(infiles, outfile, seed, cv_i):
    print("infiles:", infiles)
    meta_infile, data_infile = infiles

    np.random.seed(seed)

    config = pickle.load(open(meta_infile, 'rb'))['config']

    tgt_nucs = config['spectra_nuc']
    MAX_N = config['max_atom_n']
    print("TGT_NUCS=", tgt_nucs)
    print("output is", outfile)
    USE_CUDA = True

    mu_scale = []
    std_scale = []
    for tn in tgt_nucs:
        for k, v in NUC_LOSS_SCALE.items():
            if k in tn:
                mu_scale.append(v)
        for k, v in NUC_STD_SCALE.items():
            if k in tn:
                std_scale.append(v)
    assert len(mu_scale) == len(tgt_nucs)
    assert len(std_scale) == len(tgt_nucs)
    print("NUC_LOSS_SCALE=", NUC_LOSS_SCALE)
    print("mu_scale=", mu_scale)
    print("std_scale=", std_scale)
    print("tgt_nucs=", tgt_nucs)

    ### Create datasets and data loaders

    BATCH_SIZE = 32

    dataset_hparams = graph_conv_many_nuc_util.DEFAULT_DATA_HPARAMS.copy()
    
    dataset_hparams['feat_vect_args']['mmff_atom_types_onehot'] = True
    dataset_hparams['feat_vect_args']['feat_atomicno'] = False
    dataset_hparams['adj_args']['mat_power'] = [1,2,3,4, 6, 8]
    dataset_hparams['adj_args']['split_weights'] = None
    dataset_hparams['adj_args']['edge_weighted'] = True

    ds_train, ds_test = graph_conv_many_nuc_util.make_datasets({'filename' : data_infile}, 
                                                               dataset_hparams, MAX_N, cv_i=cv_i)
                                                               
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, 
                                           shuffle=True,pin_memory=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE, 
                                          shuffle=True,pin_memory=True)

    net_params = net_params_base.copy()
    opt_params = opt_params_base.copy()

    net_params['g_feature_n'] = ds_test[0][1].shape[-1]
    net_params['OUT_DIM'] =  len(tgt_nucs)

    #net_name = 'nets.SGCModel'
    net_name = 'nets.PredVertOnly'
    
    net = eval(net_name)(**net_params)
 
    net = move(net, USE_CUDA)

    for n, p in net.named_parameters():
        print(n, p.shape)
    if USE_STD:
        loss_config = {'std_regularize' : STD_REGULARIZE, 
                       'mu_scale' : mu_scale, 
                       'std_scale' : std_scale}

        std_regularize = loss_config['std_regularize']
        mu_scale = move(torch.Tensor(loss_config['mu_scale']), USE_CUDA)
        std_scale = move(torch.Tensor(loss_config['std_scale']), USE_CUDA)
        criterion = nets.NormUncertainLoss(mu_scale, 
                                           std_scale,
                                           std_regularize = loss_config['std_regularize'])
    else:
        loss_config = {'norm' : 'huber', 
                       'scale' : 0.1}

        criterion = nets.NoUncertainLoss(**loss_config)


    opt_direct_params = {}
    optimizer_name = opt_params.get('optimizer', 'adam') 
    if optimizer_name == 'adam':
        for p in ['lr', 'amsgrad', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adam(net.parameters(), **opt_direct_params)
    elif optimizer_name == 'adamax':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adamax(net.parameters(), **opt_direct_params)
        
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=opt_params['scheduler_step_size'], 
                                                gamma=opt_params['scheduler_gamma'])

    MODEL_NAME = "{}.{:08d}".format(EXP_NAME,int(time.time() % 1e8))
    
    checkpoint_filename = os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".{epoch_i:08d}")
    print("checkpoint:", checkpoint_filename)
    checkpoint_func = graph_conv_many_nuc_util.create_checkpoint_func(CHECKPOINT_EVERY, checkpoint_filename)
                        
    writer = SummaryWriter("{}/{}".format(TENSORBOARD_DIR, MODEL_NAME))
    validate_func = create_uncertain_validate_func(tgt_nucs, writer)

    metadata = {'dataset_hparams' : dataset_hparams, 
                'net_params' : net_params, 
                'opt_params' : opt_params, 
                'meta_infile' : meta_infile, 
                'data_infile' : data_infile, 
                'tgt_nucs' : tgt_nucs, 
                'max_n' : MAX_N,
                'net_name' : net_name, 
                'batch_size' : BATCH_SIZE, 
                'loss_params' : loss_config}

    json.dump(metadata, open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".json"), 'w'), 
              indent=4)
    print(json.dumps(metadata, indent=4))
    print("MODEL_NAME=", MODEL_NAME)
    pickle.dump(metadata, 
                open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".meta"), 'wb'))

    graph_conv_many_nuc_util.generic_runner(net, optimizer, scheduler, criterion, 
                                            dl_train, dl_test, 
                                            MAX_EPOCHS=MAX_EPOCHS, 
                                            USE_CUDA=USE_CUDA, writer=writer, 
                                            validate_func= validate_func, 
                                            checkpoint_func= checkpoint_func)

    pickle.dump({'net_params' : net_params, 
                 'opt_params' : opt_params, 
                 'loss_params' : loss_config}, 
                open(outfile, 'wb'))

if __name__ == "__main__":
    pipeline_run([train])


