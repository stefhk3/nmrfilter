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
import click

import yaml
import seminets
import mpnets
import vertedgenets
import geomnets


CHECKPOINT_DIR = "checkpoints" 
CHECKPOINT_EVERY = 50



def train(exp_config_name, exp_config, exp_extra_name="", 
          USE_CUDA = True,
          exp_config_filename=None,
          add_timestamp = True):
    
    #meta_infile = exp_config['meta_infile']
    
    #mkdir(CHEKCPOINT_DIR)
    EXP_NAME = exp_config_name
    if exp_extra_name is not None and len(exp_extra_name ) > 0:
        EXP_NAME += ".{}".format(exp_extra_name)

    if add_timestamp:
        MODEL_NAME = "{}.{:08d}".format(EXP_NAME,int(time.time() % 1e8))
    else:
        MODEL_NAME = EXP_NAME
    CHECKPOINT_BASENAME = os.path.join(CHECKPOINT_DIR, MODEL_NAME )
            
        
    np.random.seed(exp_config['seed'])

    #config = pickle.load(open(meta_infile, 'rb'))['config']

    tgt_nucs = exp_config['spectra_nucs']
    MAX_N = exp_config['tgt_max_n']


    BATCH_SIZE = exp_config['batch_size']
    dataset_hparams_update = exp_config['dataset_hparams']
    dataset_hparams = netutil.DEFAULT_DATA_HPARAMS
    util.recursive_update(dataset_hparams, 
                          dataset_hparams_update)

    exp_data = exp_config['exp_data']
    cv_func = netutil.CVSplit(**exp_data['cv_split'])
    
    datasets = {}
    for ds_config_i, dataset_config in enumerate(exp_data['data']):
        ds, phase_data = netutil.make_dataset(dataset_config, dataset_hparams, 
                                  MAX_N, cv_func)
        phase = dataset_config['phase']
        if phase not in datasets:
            datasets[phase] = []
        datasets[phase].append(ds)

        pickle.dump(phase_data,
                    open(CHECKPOINT_BASENAME + f".data.{ds_config_i}.{phase}.data", 'wb'))
        
    ds_train = datasets['train'][0] if len(datasets['train']) == 1 else torch.utils.data.ConcatDataset(datasets['train'])
    ds_test = datasets['test'][0] if len(datasets['test']) == 1 else torch.utils.data.ConcatDataset(datasets['test'])

    print("we are training with", len(ds_train))
    print("we are testing with", len(ds_test))

    dataloader_name = exp_config.get("dataloader_func",
                                 'torch.utils.data.DataLoader')

    dataloader_creator = eval(dataloader_name)
    
    dl_train = dataloader_creator(ds_train, batch_size=BATCH_SIZE, 
                                  shuffle=True,pin_memory=False)
    dl_test = dataloader_creator(ds_test, batch_size=BATCH_SIZE, 
                                 shuffle=True,pin_memory=False)
    

    net_params = exp_config['net_params']
    net_name = exp_config['net_name']

    net_params['g_feature_n'] = ds_test[0]['vect_feat'].shape[-1]
    net_params['GS'] = ds_test[0]['adj'].shape[0]
    #net_params['extra_vert_in_d'] = int(np.prod(ds_test[0][-1].shape[1:]))
    net_params['OUT_DIM'] =  len(tgt_nucs)

    print(net_params)
    
    net = eval(net_name)(**net_params)
 
    net = move(net, USE_CUDA)

    for n, p in net.named_parameters():
        print(n, p.shape)

    loss_params = exp_config['loss_params']
    loss_name = loss_params['loss_name']

    std_regularize = loss_params.get('std_regularize', 0.01)
    mu_scale = move(torch.Tensor(loss_params.get('mu_scale', [1.0])), USE_CUDA)
    std_scale = move(torch.Tensor(loss_params.get('std_scale', [1.0])), USE_CUDA)

    if loss_name == 'NormUncertainLoss':
        criterion = nets.NormUncertainLoss(mu_scale, 
                                           std_scale,
                                           std_regularize = std_regularize)
    elif loss_name == 'UncertainLoss':
        criterion = nets.UncertainLoss(mu_scale, 
                                       std_scale,
                                       norm = loss_params['norm'], 
                                       std_regularize = std_regularize, 
                                       std_pow = loss_params['std_pow'], 
                                       use_reg_log = loss_params['use_reg_log'],
                                       std_weight = loss_params['std_weight'])

    elif loss_name == "NoUncertainLoss":
        
        criterion = nets.NoUncertainLoss(**loss_params)
    elif loss_name == "PermMinLoss":
        
        criterion = nets.PermMinLoss(**loss_params)
    elif loss_name == "ReconLoss":
        
        criterion = seminets.ReconLoss(**loss_params)
    else:
        raise ValueError(loss_name)

    opt_params = exp_config['opt_params']
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
        
    elif optimizer_name == 'adagrad':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.Adagrad(net.parameters(), **opt_direct_params)
        
    elif optimizer_name == 'rmsprop':
        for p in ['lr', 'eps', 'weight_decay', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.RMSprop(net.parameters(), **opt_direct_params)
        
    elif optimizer_name == 'sgd':
        for p in ['lr', 'momentum']:
            if p in opt_params:
                opt_direct_params[p] = opt_params[p]

        optimizer = torch.optim.SGD(net.parameters(), **opt_direct_params)
        
    scheduler_name = opt_params.get('scheduler_name', 'steplr')
    if scheduler_name == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=opt_params['scheduler_step_size'], 
                                                    gamma=opt_params['scheduler_gamma'])
    elif scheduler_name == 'lambdalr':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda = eval(opt_params['scheduler_lr_lambda']))
        

        

    checkpoint_filename = CHECKPOINT_BASENAME + ".{epoch_i:08d}"
    print("checkpoint:", checkpoint_filename)
    checkpoint_func =netutil.create_checkpoint_func(CHECKPOINT_EVERY, checkpoint_filename)

    #save_output_func = netutil.create_save_val_func(CHECKPOINT_BASENAME, )
    
    TENSORBOARD_DIR = exp_config.get('tblogdir', f"tensorboard.logs")
    writer = SummaryWriter("{}/{}".format(TENSORBOARD_DIR, MODEL_NAME))
    validate_func = netutil.create_uncertain_validate_func(tgt_nucs, writer)
    validate_sorted_func = netutil.create_permutation_validate_func(tgt_nucs, writer)

    metadata = {'dataset_hparams' : dataset_hparams, 
                'net_params' : net_params, 
                'opt_params' : opt_params, 
                'exp_data' : exp_data, 
                #'meta_infile' : meta_infile, 
                'exp_config' : exp_config, 
                'tgt_nucs' : tgt_nucs, 
                'max_n' : MAX_N,
                'net_name' : net_name, 
                'batch_size' : BATCH_SIZE, 
                'loss_params' : loss_params}

    json.dump(metadata, open(CHECKPOINT_BASENAME + ".json", 'w'), 
              indent=4)
    print(json.dumps(metadata, indent=4))
    print("MODEL_NAME=", MODEL_NAME)
    pickle.dump(metadata, 
                open(CHECKPOINT_BASENAME + ".meta", 'wb'))

    with open(os.path.join(CHECKPOINT_DIR, 
                           "{}.yaml".format(MODEL_NAME,)), 'w') as fid:
        fid.write(open(exp_config_filename, 'r').read())
    
    netutil.generic_runner(net, optimizer, scheduler, criterion, 
                          dl_train, dl_test, 
                          MAX_EPOCHS=exp_config['max_epochs'], 
                          USE_CUDA=USE_CUDA, writer=writer, 
                          validate_funcs= [validate_func, validate_sorted_func],
                          checkpoint_func= checkpoint_func)

    # pickle.dump({'net_params' : net_params, 
    #              'opt_params' : opt_params, 
    #              'loss_params' : loss_params}, 
    #             open(outfile, 'wb'))


@click.command()
@click.argument('exp_config_name')
@click.argument('exp_extra_name', required=False)
@click.option('--skip-timestamp', default=False, is_flag=True)
def run(exp_config_name, exp_extra_name, skip_timestamp=False):
    

    # exp_config_name = sys.argv[1]
    # if len(sys.argv) > 2:
    #     exp_extra_name = sys.argv[2]
    # else:
    #     exp_extra_name = ""
    exp_config = yaml.load(open(exp_config_name, 'r'), Loader=yaml.FullLoader)
    exp_name = os.path.basename(exp_config_name.replace(".yaml", ""))
    train(exp_name, exp_config, exp_extra_name,
          exp_config_filename=exp_config_name, add_timestamp=not skip_timestamp)
    

if __name__ == "__main__":
    run()

