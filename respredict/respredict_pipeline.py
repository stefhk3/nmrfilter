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
import netutil
from netutil import move
import sys

DATASET_DIR = "../nmrdata/processed_data/"


td = lambda x : os.path.join(DATASET_DIR, x)


SPECT_SET = '1H' # or '13C'

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
    #NUC_LOSS_SCALE = {'1H' : 1.0}
    #NUC_STD_SCALE = {'1H' : 1.0} 
    STD_REGULARIZE = 0.01

tgt_max_n = 64 
MAX_EPOCHS = 10000

extra_exp_name = sys.argv[1]
EXP_NAME = f"{extra_exp_name}_{SPECT_SET}"


USE_STD = False

net_params_base = {'init_noise' : 0.0, 
                   'resnet' : True, 
                   'int_d' :  128, # 2048, 
                   'layer_n' : 8, # 10, 
                   'GS' : 4, 
                   'agg_func' : 'goodmax', 
                   'force_lin_init' : True, 
                   'g_feature_n' : -1, 
                   'resnet_out' : True, 
                   'out_std' : USE_STD, 
                   'batchnorm' : True, 
                   'input_batchnorm' : True, 
                   'graph_dropout' : 0.0, 
                   'resnet_blocks' : (3,),
                   'resnet_d': 128, 
                   'mixture_n' : 10, 
                   'out_std_exp' : False, 
                   'OUT_DIM' : 1, # update
                   'use_random_subsets' : False, 
}

opt_params_base = {'optimizer' : 'adam', 
                   #'amsgrad' : False, 
                   'lr' : 1e-4, 
                   #'weight_decay' : 1e-5, 
                   'scheduler_gamma' : 0.95, 
                   #'momentum' : 0.9,
                   'eps' : 1e-8, 
                   'scheduler_step_size' : 100}


def params():
    CV_I = int(os.environ.get("CV_I", 0))
    data_file = td("{}.shifts.dataset.pickle".format(DATASET_NAME))
    meta_file = td("{}.meta.pickle".format(DATASET_NAME))
    
    seed = 1234                                                                                              

    outfile = f"respredict_pipeline_{SPECT_SET}.{seed}.{CV_I}.out"
    


    yield (meta_file, data_file), outfile, seed, CV_I

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

    dataset_hparams =netutil.DEFAULT_DATA_HPARAMS
    
    dataset_hparams['feat_vect_args']['mmff_atom_types_onehot'] = True
    dataset_hparams['feat_vect_args']['feat_atomicno'] = False

    exp_data = {'filename' : data_infile, 
                'extra_data' : [
                    # {'name' : 'geom_radial', 
                    #  'combine_with' : 'vert', 
                    #  'fileglob' : 'features.geom/features.mmff94_64_opt1000_4_p01-default_bp_radial.dir/{molecule_id}.npy'}
                    ]}


    ds_train, ds_test =netutil.make_datasets(exp_data,  dataset_hparams, MAX_N, cv_i=cv_i)

    dataloader_name = config.get("dataloader",
                                    'torch.utils.data.DataLoader')
    dataloader_creator = eval(dataloader_name)
    
    dl_train = dataloader_creator(ds_train, batch_size=BATCH_SIZE, 
                                  shuffle=True,pin_memory=True)
    dl_test = dataloader_creator(ds_test, batch_size=BATCH_SIZE, 
                                 shuffle=True,pin_memory=True)

    net_params = net_params_base.copy()
    opt_params = opt_params_base.copy()

    net_params['g_feature_n'] = ds_test[0]['vect_feat'].shape[-1]
    #net_params['extra_vert_in_d'] = int(np.prod(ds_test[0][-1].shape[1:]))
    net_params['OUT_DIM'] =  len(tgt_nucs)
    
    net_name = 'nets.GraphVertModelBootstrap'
    net = eval(net_name)(**net_params)
 
    net = move(net, USE_CUDA)

    for n, p in net.named_parameters():
        print(n, p.shape)
    if USE_STD:

        loss_config = {'std_regularize' : STD_REGULARIZE, 
                       'mu_scale' : mu_scale, 
                       'norm' : 'l2', 
                       'loss_name' : 'UncertainLoss', 
                       'std_pow' : 2.0, 
                       'std_weight' : 8.0,
                       'use_reg_log' : False, 
                       'std_scale' : std_scale}
        loss_name = loss_config['loss_name']

        std_regularize = loss_config['std_regularize']
        mu_scale = move(torch.Tensor(loss_config['mu_scale']), USE_CUDA)
        std_scale = move(torch.Tensor(loss_config['std_scale']), USE_CUDA)
        if loss_name == 'NormUncertainLoss':
            criterion = nets.NormUncertainLoss(mu_scale, 
                                               std_scale,
                                               std_regularize = std_regularize)
        elif loss_name == 'UncertainLoss':
            criterion = nets.UncertainLoss(mu_scale, 
                                           std_scale,
                                           norm = loss_config['norm'], 
                                           std_regularize = std_regularize, 
                                           std_pow = loss_config['std_pow'], 
                                           use_reg_log = loss_config['use_reg_log'],
                                           std_weight = loss_config['std_weight'])
        else:
            raise ValueError(loss_name)
            
    else:
        loss_config = {'norm' : 'huber', 
                       'scale' : 0.1 * NUC_STD_SCALE[SPECT_SET]}

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
        
    elif optimizer_name == 'sgd':
        for p in ['lr', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.SGD(net.parameters(), **opt_direct_params)
        
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=opt_params['scheduler_step_size'], 
                                                gamma=opt_params['scheduler_gamma'])

    MODEL_NAME = "{}.{:08d}".format(EXP_NAME,int(time.time() % 1e8))
    
    checkpoint_filename = os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".{epoch_i:08d}")
    print("checkpoint:", checkpoint_filename)
    checkpoint_func =netutil.create_checkpoint_func(CHECKPOINT_EVERY, checkpoint_filename)
                        
    writer = SummaryWriter("{}/{}".format(TENSORBOARD_DIR, MODEL_NAME))
    validate_func = netutil.create_uncertain_validate_func(tgt_nucs, writer)

    metadata = {'dataset_hparams' : dataset_hparams, 
                'net_params' : net_params, 
                'opt_params' : opt_params, 
                'exp_data' : exp_data, 
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

    netutil.generic_runner(net, optimizer, scheduler, criterion, 
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


