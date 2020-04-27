import numpy as np
import pandas as pd
import pickle

import torch
from torch import nn
import torch.nn.functional as F
import nets
import spinsys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import json
import os

if __name__ == "__main__":

    dataset_filename = "coupling.datasets/gissmo.moldict.32.0.HCONFSPCl.pickle"
    datasets = pickle.load(open(dataset_filename, 'rb'))
    train_df = datasets['train_df']
    test_df = datasets['test_df']
    MAX_N = datasets['MAX_N']



    default_atomicno = [1, 6, 7, 8, 9, 15, 16, 17]
    ### Create datasets and data loaders

    default_feat_vert_args = dict(feat_atomicno_onehot=default_atomicno, 
                                  feat_pos=False, feat_atomicno=True,
                                  feat_valence=True, aromatic=True, hybridization=True, 
                                  partial_charge=False, formal_charge=True,  # WE SHOULD REALLY USE THIS 
                                  r_covalent=False,
                                  total_valence_onehot=True, 

                                  r_vanderwals=False, default_valence=True, rings=True)

    default_feat_edge_args = dict(feat_distances = False, 
                                 feat_r_pow = None)

    split_weights = [1, 1.5, 2, 3]

    default_adj_args = dict(edge_weighted=False, 
                            norm_adj=False, add_identity=False, 
                            split_weights=split_weights)


    dataset_hparams = {'feat_vert_args' : default_feat_vert_args, 
                    'feat_edge_args' : default_feat_edge_args, 
                    'adj_args' : default_adj_args}


    BATCH_SIZE = 64

    ds = {}
    dl = {}
    for name, dataset_df in [('train', train_df), 
                             ('test', test_df)]:

        ds[name] = spinsys.MoleculeDatasetMulti(list(dataset_df.rdmol),
                                                list(dataset_df.shifts), 
                                                list(dataset_df.couplings), 
                                                MAX_N = MAX_N , 
                                                **dataset_hparams, 
                                         )



        dl[name] = torch.utils.data.DataLoader(ds[name], batch_size=BATCH_SIZE, 
                                                   shuffle=True)




    CHECKPOINT_DIR = "checkpoints"
    EPOCH_N = 10000

    #net = GraphMatModel(ds[0][2].shape[-1], [16]*3, noise=0.1 )

    graph_vert_edge=False
    if graph_vert_edge:


        net_params = dict(vert_f_in = ds['train'][0][1].shape[-1], 
                          edge_f_in = ds['train'][0][0].shape[-1],
                          MAX_N=MAX_N, layer_n=5, 
                          internal_d_vert=128,
                          internal_d_edge=128, resnet=False, 
                          use_batchnorm = True, force_lin_init = True, 
                          e_agg_func='sum', 
                          init_noise = 0.001)

        net = spinsys.GraphVertEdgeNet(**net_params)

        opt_params = {'lr' : 1e-2, 'eps' : 1e-9,
                      'amsgrad' : False, 'weight_decay' : 1e-9, 
                     'scheduler_step_size' : 40, 'scheduler_gamma' : 0.95}


    else:
        net_params = dict(g_feature_n = ds['train'][0][1].shape[-1], 
                          int_d=1024, layer_n=10, batch_norm=False, 
                          agg_func='goodmax', 
                          resnet=False, resnet_out=True, 
                          fc_out_depth=1, 
                          OUT_DIM=1, 
                          force_lin_init=True, init_noise=1e-3)
        net = spinsys.GraphConvMatOut(**net_params)

        opt_params = {'lr' : 1e-3, 'eps' : 1e-9,
                      'amsgrad' : False, 'weight_decay' : 1e-7, 
                      'scheduler_step_size' : 10, 'scheduler_gamma' : 0.95}



    net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=opt_params['lr'], 
                                 amsgrad=opt_params['amsgrad'], 
                                 eps=opt_params['eps'],
                                 weight_decay=opt_params['weight_decay'])
    if opt_params['scheduler_step_size'] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=opt_params['scheduler_step_size'], 
                                                    gamma=opt_params['scheduler_gamma'])
    else:
        scheduler = None

    criterion = nets.MaskedMSELoss()


    EXP_NAME = "jcouple2"
    MODEL_NAME = "{}.{:08d}".format(EXP_NAME,int(time.time() % 1e8))


    metadata = {'dataset_hparams' : dataset_hparams, 
                'net_params' : net_params, 
                'opt_params' : opt_params, 
                'batch_size' : BATCH_SIZE, 
                'infile' : dataset_filename, 
                'max_n' : MAX_N, 
                'network' : str(type(net))}


    TENSORBOARD_DIR = "logs.spinsys"
    writer = SummaryWriter("{}/{}".format(TENSORBOARD_DIR, MODEL_NAME))

    json.dump(metadata, open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".json"), 'w'), 
              indent=4)
    print(json.dumps(metadata, indent=4))

    print(MODEL_NAME)
    pickle.dump(metadata, 
                open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".meta"), 'wb'))


    def create_checkpoint_func(every_n, filename_str):
        def checkpoint(epoch_i, net, optimizer):
            if epoch_i % every_n > 0:
                return {}
            checkpoint_filename = filename_str.format(epoch_i = epoch_i)
            t1 = time.time()
            torch.save(net.state_dict(), checkpoint_filename + ".state")
            #torch.save(net, checkpoint_filename + ".model")
            t2 = time.time()
            return {'savetime' : (t2-t1)}
        return checkpoint

    checkpoint_filename = os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".{epoch_i:08d}")

    checkpoint_func = create_checkpoint_func(20, checkpoint_filename)


    for epoch_i in tqdm(range(EPOCH_N)):
        if scheduler is not None:
            scheduler.step()


        running_loss = 0.0
        net.train()
        total_points = 0 
        for a_i, (adj, vect_feat, mat_feat, 
                     vect_vals, 
                     vect_mask,
                     mat_vals, 
                     mat_mask) in enumerate(dl['train']):

            optimizer.zero_grad()
            #adj = adj.transpose(1, 3)
            res_v, res_e = net(vect_feat.cuda(), adj.cuda())
            loss = criterion(res_e, mat_vals.cuda(), mat_mask.cuda())
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            total_points += mat_mask.sum()

        writer.add_scalar("train/loss", running_loss, 
                          epoch_i)

        writer.add_scalar("train/loss_mean", running_loss/total_points, 
                          epoch_i)

        if epoch_i % 5 == 0:
            net.eval()
            optimizer.zero_grad()
            running_loss = 0.0
            total_points = 0
            pred_batches = []
            for a_i, (adj, vect_feat, mat_feat, 
                         vect_vals, 
                         vect_mask,
                         mat_vals, 
                         mat_mask) in enumerate(dl['test']):


                #adj = adj.transpose(1, 3)
                res_v, res_e = net(vect_feat.cuda(), adj.cuda())
                loss = criterion(res_e, mat_vals.cuda(), mat_mask.cuda())
                running_loss += float(loss.item())
                total_points += mat_mask.sum()

                pred_batches.append({'adj' : adj.detach().cpu().numpy(), 
                                     'vect_feat' : vect_feat.detach().cpu().numpy(), 
                                     'vect_mask' : vect_mask.detach().cpu().numpy(), 
                                     'mat_vals' : mat_vals.detach().cpu().numpy(), 
                                     'mat_mask' : mat_mask.detach().cpu().numpy(), 
                                     'res_e' : res_e.detach().cpu().numpy(), 
                                     'batch_i' : a_i, 'epoch_i' : epoch_i})
            pickle.dump(pred_batches, 
                        open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + f".{epoch_i:08d}.pred_batch"), 
                             'wb'))


            writer.add_scalar("test/loss", running_loss, 
                              epoch_i)

            writer.add_scalar("test/loss_mean", running_loss/total_points, 
                              epoch_i)

        checkpoint_func(epoch_i = epoch_i, net =net, optimizer=optimizer)
