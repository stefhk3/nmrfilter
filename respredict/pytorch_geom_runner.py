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

import yaml
import geomnets


from torch_geometric.data import Dataset as GeomDataset
from torch_geometric.data import Data as GeomData
from torch_geometric.data import DataLoader as GeomDataLoader


class MyOwnDataset(GeomDataset):
    def __init__(self, input_ds, root="", transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.input_ds = input_ds

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def __len__(self):
        return len(self.input_ds)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        return

        # i = 0
        # for raw_path in self.raw_paths:
        #      # Read data from `raw_path`.
        #      data = Data(...)

        #      if self.pre_filter is not None and not self.pre_filter(data):
        #          continue

        #     if self.pre_transform is not None:
        #          data = self.pre_transform(data)

        #     torch.save(data, ops.join(self.processed_dir, 'data_{}.pt'.format(i)))
        #     i += 1

    def get(self, idx):
        #data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
        #return data
        d = self.input_ds[idx]
        adj = d['adj']
        vect_feat = d['vect_feat']
        input_mask = d['input_mask']
        pred_mask = d['pred_mask']
        vals = d['vals']

        #print(adj.shape)
        have_edge, _ = adj.max(dim=0)
        #print("have_edge.shape=", have_edge.shape)
        edges = np.argwhere(have_edge.numpy())


        #print("adj.shape=", adj.shape)
        #print("edges.shape=", edges.shape)

        edge_attr = [adj[:, i, j] for i, j in edges]

        edge_attr = torch.stack(edge_attr)
        #print("edge_attr.shape=", edge_attr.shape)

        edge_index_tensor = torch.Tensor(edges.T).long()
        #print("edge_index_tensor.shape=", edge_index_tensor.shape, "edge_index_tensor.dtype=", edge_index_tensor.dtype)
        
        y = torch.Tensor(vals) #.reshape(1, -1, 1)
        
        # FIXME maybe we need to select non-padded nodes here? 
        data = GeomData(x=torch.Tensor(vect_feat), 
                        edge_index=edge_index_tensor, 
                        edge_attr=edge_attr, y=y)

        data.pred_mask = torch.Tensor(pred_mask) # .reshape(1, -1, 1)
        data.adj = torch.Tensor(adj )
        # print("pred_mask.shape=", pred_mask.shape, 
        #       " returning data.pred_mask.shape=", data.pred_mask.shape, 
        #       "y.shape=", y.shape)
        return data



CHECKPOINT_DIR = "checkpoints" 
CHECKPOINT_EVERY = 50



# loss_params = {'std_regularize' : 0.01, 
#                'mu_scale' : [1.0/20.0],
#                'norm' : 'l2', 
#                'loss_name' : 'UncertainLoss', 
#                'std_pow' : 2.0, 
#                'std_weight' : 8.0,
#                'use_reg_log' : False, 
#                'std_scale' : [10.0]}

# BATCH_SIZE = 32

# tgt_max_n = 64 

#extra_exp_name = sys.argv[1]
#EXP_NAME = f"{extra_exp_name}_{SPECT_SET}"


# net_params = {'init_noise' : 0.0, 
#               'resnet' : True, 
#               'int_d' :  128, # 2048, 
#               'layer_n' : 8, # 10, 
#               'GS' : 4, 
#               'agg_func' : 'goodmax', 
#               'force_lin_init' : True, 
#               'g_feature_n' : -1, 
#               'resnet_out' : True, 
#               'out_std' : False, 
#               'batchnorm' : True, 
#               'input_batchnorm' : True, 
#               'graph_dropout' : 0.0, 
#               'resnet_blocks' : (3,),
#               'resnet_d': 128, 
#               'mixture_n' : 10, 
#               'out_std_exp' : False, 
#               'OUT_DIM' : 1, # update
#               'use_random_subsets' : False, 
# }

# net_name = 'nets.GraphVertModelBootstrap'

# opt_params = {'optimizer' : 'adam', 
#               #'amsgrad' : False, 
#               'lr' : 1e-4, 
#               #'weight_decay' : 1e-5, 
#               'scheduler_gamma' : 0.95, 
#               #'momentum' : 0.9,
#               'eps' : 1e-8, 
#               'scheduler_step_size' : 100}


# data_infile = td("{}.shifts.dataset.pickle".format(DATASET_NAME))
# meta_infile = td("{}.meta.pickle".format(DATASET_NAME))
    
# seed = 1234                                                                                              

# dataset_hparams = netutil.DEFAULT_DATA_HPARAMS

# dataset_hparams['feat_vect_args']['mmff_atom_types_onehot'] = True
# dataset_hparams['feat_vect_args']['feat_atomicno'] = False

# exp_data = {'filename' : data_infile, 
#             'extra_data' : [
#                 # {'name' : 'geom_radial', 
#                 #  'combine_with' : 'vert', 
#                 #  'fileglob' : 'features.geom/features.mmff94_64_opt1000_4_p01-default_bp_radial.dir/{molecule_id}.npy'}
#                 ]}


# loss_params = {'norm' : 'huber', 
#                'scale' : 1.0 }

from torch_geometric.nn import GCNConv

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set

def run_epoch(net, optimizer, criterion, dl, 
              pred_only = False, USE_CUDA=True,
              return_pred = False, desc="train", 
              print_shapes=False, progress_bar=True, 
              writer=None, epoch_i=None, MAX_N=None):
    t1_total= time.time()

    accum_pred = []
    running_loss = 0.0
    total_points = 0
    total_compute_time = 0.0
    if progress_bar:
        iterator =  tqdm(enumerate(dl), total=len(dl), desc=desc)
    else:
        iterator = enumerate(dl)

    input_row_count = 0
    for i_batch, batch in iterator:
        
        t1 = time.time()

        if not pred_only:
            optimizer.zero_grad()

        batch_t = batch.to('cuda')
        #print('before move, batch.y.shape=', batch.y.shape)

        res = net(batch_t)
        # for k, v in res.items():
        #     print(f"{k}.shape=", v.shape)
        # print('after move, batch_t.y.shape=', batch_t.y.shape)

        pred_mask_batch_t = batch_t.pred_mask.reshape(-1, MAX_N, 1)
        y_batch_t = batch_t.y.reshape(-1, MAX_N, 1)
        if return_pred:
            accum_pred_val = {}
            if isinstance(res, dict):
                for k, v in res.items():
                    accum_pred_val[k] = res[k].cpu().detach().numpy()
            else:
                accum_pred_val['res'] = res.cpu().detach().numpy()
            accum_pred_val['mask'] = pred_mask_batch_t.cpu().detach().numpy()
            accum_pred_val['truth'] = y_batch_t.cpu().detach().numpy()

            accum_pred.append(accum_pred_val)

        if criterion is None:
            loss = 0.0
        else:
            loss = criterion(res,  y_batch_t, pred_mask_batch_t)

        if not pred_only:
            loss.backward()
            
            optimizer.step()

        obs_points = batch.pred_mask.sum()
        if criterion is not None:
            running_loss += loss.item() * obs_points
        total_points +=  obs_points


        t2 = time.time()
        total_compute_time += (t2-t1)

        input_row_count += batch.adj.shape[0]
    t2_total = time.time()
    


    res =  {'timing' : 0.0, 
            'running_loss' : running_loss, 
            'total_points' : total_points, 
            'mean_loss' : running_loss / total_points,
            'runtime' : t2_total-t1_total, 
            'compute_time' : total_compute_time, 
            'run_efficiency' : total_compute_time / (t2_total-t1_total), 
            'pts_per_sec' : input_row_count / (t2_total-t1_total), 
            }
    if return_pred:
        keys = accum_pred[0].keys()
        for k in keys:
            #print(f"{k} accum_pred[0][{k}].shape=", accum_pred[0][k].shape)
            #print([a[k].shape for a in accum_pred])
            accum_pred_v = np.vstack([a[k] for a in accum_pred])
            res[f'pred_{k}'] = accum_pred_v
            
    return res



def generic_runner(net, optimizer, scheduler, criterion, 
                   dl_train, dl_test, MAX_N, 
                   MAX_EPOCHS=1000, 
                   USE_CUDA=True, use_std=False, 
                   writer=None, validate_func = None, 
                   checkpoint_func = None, prog_bar=True):


    # loss_scale = torch.Tensor(loss_scale)
    # std_scale = torch.Tensor(std_scale)


    for epoch_i in tqdm(range(MAX_EPOCHS)):
        if scheduler is not None:
            scheduler.step()

        running_loss = 0.0
        total_compute_time = 0.0
        t1_total = time.time()

        net.train()
        train_res = run_epoch(net, optimizer, criterion, dl_train, 
                              pred_only = False, USE_CUDA=USE_CUDA, 
                              return_pred=True, progress_bar=prog_bar,
                              desc='train', writer=writer, epoch_i=epoch_i, 
                              MAX_N = MAX_N)
        validate_func(train_res, "train_", epoch_i)
        writer.add_histogram("pred_mu", train_res['pred_mu'], epoch_i)
        if epoch_i % 5 == 0:
            net.eval()
            test_res = run_epoch(net, optimizer, criterion, dl_test, 
                                 pred_only = True, USE_CUDA=USE_CUDA, 
                                 progress_bar=prog_bar, 
                                 return_pred=True, desc='validate', MAX_N=MAX_N)
            validate_func(test_res, "validate_", epoch_i)
            
            
        if checkpoint_func is not None:
            checkpoint_func(epoch_i = epoch_i, net =net, optimizer=optimizer)




def train(exp_config_name, exp_config, exp_extra_name="", 
          USE_CUDA = True):
    
    #meta_infile = exp_config['meta_infile']
    
    #mkdir(CHEKCPOINT_DIR)
    EXP_NAME = exp_config_name
    if len(exp_extra_name ) > 0:
        EXP_NAME += ".{}".format(exp_extra_name)

    np.random.seed(exp_config['seed'])

    #config = pickle.load(open(meta_infile, 'rb'))['config']

    tgt_nucs = exp_config['spectra_nucs']
    MAX_N = exp_config['tgt_max_n']


    BATCH_SIZE = exp_config['batch_size']
    dataset_hparams_update = exp_config['dataset_hparams']
    dataset_hparams = netutil.DEFAULT_DATA_HPARAMS
    util.recursive_update(dataset_hparams, 
                          dataset_hparams_update)

    cv_i = exp_config.get('cv_i', 0)
    exp_data = exp_config['exp_data']
    #print(exp_data)
    ds_train, ds_test =netutil.make_datasets(exp_data,  dataset_hparams, MAX_N, cv_i=cv_i, 
                                             train_sample=exp_config.get('train_sample_max', 0))
                                                               

    geom_ds_train = MyOwnDataset(ds_train)
    dl_train = GeomDataLoader(geom_ds_train, batch_size=BATCH_SIZE, shuffle=True)

    geom_ds_test = MyOwnDataset(ds_test)
    dl_test = GeomDataLoader(geom_ds_test, batch_size=BATCH_SIZE, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_params = exp_config['net_params']
    net_name = exp_config['net_name']
    
    net_params['g_feature_n'] = ds_train[0]['vect_feat'].shape[1]
    net_params['MAX_N'] = MAX_N
    net = eval(net_name)(**net_params)

    net = move(net, USE_CUDA)

    # for n, p in net.named_parameters():
    #     print(n, p.shape)

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

    
    TENSORBOARD_DIR = f"logs" # fixME.format("".join(tgt_nucs))
    writer = SummaryWriter("{}/{}".format(TENSORBOARD_DIR, MODEL_NAME))
    validate_func = netutil.create_uncertain_validate_func(tgt_nucs, writer)

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

    json.dump(metadata, open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".json"), 'w'), 
              indent=4)
    print(json.dumps(metadata, indent=4))
    print("MODEL_NAME=", MODEL_NAME)
    pickle.dump(metadata, 
                open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".meta"), 'wb'))

    generic_runner(net, optimizer, scheduler, criterion, 
                   dl_train, dl_test, MAX_N, 
                   MAX_EPOCHS=exp_config['max_epochs'], 
                   USE_CUDA=USE_CUDA, writer=writer, 
                   validate_func= validate_func, 
                   checkpoint_func= checkpoint_func)
    

if __name__ == "__main__":
    exp_config_name = sys.argv[1]
    if len(sys.argv) > 2:
        exp_extra_name = sys.argv[2]
    else:
        exp_extra_name = ""
    exp_config = yaml.load(open(exp_config_name, 'r'))
    exp_name = os.path.basename(exp_config_name.replace(".yaml", ""))
    train(exp_name, exp_config, exp_extra_name)
    

      

