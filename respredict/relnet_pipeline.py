import numpy as np
import pickle
import pandas as pd
from ruffus import * 
from tqdm import  tqdm
from rdkit import Chem
import pickle
import os

from glob import glob

import time
import util
import relnets

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors as rdMD

import torch
from torch import nn
from tensorboardX import SummaryWriter

from tqdm import  tqdm
from netdataio import * 
import netdataio
from util import move

CV_SETS = [range(5)]

def dataset_params():
    for CV_I in range(1):
        for nucleus in ['13C', '1H']:
            for kekulize_prop in ['kekulize', 'aromatic']:
                for dataset_name in ['nmrshiftdb_hconf2_nmrshiftdb', 
                                     'nmrshiftdb_hconfspcl_nmrshiftdb']:
                    outfile = 'relnet_pipeline.data.{}.{}.{}.{}.mol_dict.pickle'.format(nucleus, dataset_name, 
                                                                                            kekulize_prop, CV_I)
                    spectra_filename = f'dataset.named/spectra.nmrshiftdb_{nucleus}.feather'
                    mol_filename = f'dataset.named/molconf.{dataset_name}.pickle'
                    yield ((spectra_filename, mol_filename), 
                           outfile, CV_I, kekulize_prop)


MAX_N = 32

@files(dataset_params)
def create_dataset(infiles, outfile, cv_i, kekulize_prop):
    spectra_filename, mol_filename = infiles

    mol_subset_filename = 'predict.atomic/molconf.nmrshiftdb_hconfspcl_nmrshiftdb.subsets.pickle'
    cv_mol_subset = CV_SETS[cv_i]
    mol_subsets = pickle.load(open(mol_subset_filename, 'rb'))['splits_df']

    spectra_df = pd.read_feather(spectra_filename).rename(columns={'id' : 'peak_id'})


    molecules_df = pickle.load(open(mol_filename, 'rb'))['df']
    molecules_df['atom_n'] = molecules_df.rdmol.apply(lambda x: x.GetNumAtoms())
    molecules_df = molecules_df[molecules_df.atom_n <= MAX_N]
    

    def s_dict(r):
        return dict(zip(r.atom_idx, r.value))



    spect_dict_df = spectra_df.groupby(['molecule_id', 'spectrum_id']).apply(s_dict )

    data_df = spect_dict_df.reset_index()\
                           .rename(columns={0 : 'value'})\
                           .join(molecules_df, on='molecule_id').dropna()

    for row_i, row in tqdm(data_df.iterrows(), total=len(data_df)):
        mol = row.rdmol
        try:
            Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, 
                             catchErrors=True)
            mol.UpdatePropertyCache()
            Chem.SetAromaticity(mol)
            if kekulize_prop == 'kekulize':
                Chem.rdmolops.Kekulize(mol)
        except ValueError:
            pass
            

    ### Train/test split
    train_test_split = mol_subsets.subset20_i.isin(cv_mol_subset)
    train_mols = mol_subsets[~train_test_split].index.values
    test_mols = mol_subsets[train_test_split].index.values

    train_df = data_df[data_df.molecule_id.isin(train_mols)]
    test_df = data_df[data_df.molecule_id.isin(test_mols)]

    pickle.dump({'train_df' : train_df, 
                 'test_df' : test_df}, 
                open(outfile, 'wb'), -1)


TENSORBOARD_DIR = "logs"
CHECKPOINT_DIR = "checkpoints" 
        
def generic_runner(net, optimizer, scheduler, criterion, 
                   dl_train, dl_test, MODEL_NAME, 
                   MAX_EPOCHS=1000, CHECKPOINT_EVERY=20, 
                   data_feat = ['mat'], USE_CUDA=True):

    writer = SummaryWriter(f"{TENSORBOARD_DIR}/{MODEL_NAME}")
    
    for epoch_i in tqdm(range(MAX_EPOCHS)):
        if scheduler is not None:
            scheduler.step()
        running_loss = 0.0
        total_points = 0.0
        total_compute_time = 0.0
        t1_total = time.time()
        net.train()
        for i_batch, (adj, vect_feat, mat_feat, vals, mask) in \
                tqdm(enumerate(dl_train), total=len(dl_train)):
            t1 = time.time()
            optimizer.zero_grad()
            
            adj_t = move(adj, USE_CUDA)
            args = [adj_t]

            if 'vect' in data_feat:
                vect_feat_t = move(vect_feat, USE_CUDA)
                args.append(vect_feat_t)

            if 'mat' in data_feat:
                mat_feat_t = move(mat_feat, USE_CUDA)
                args.append(mat_feat_t)
                
            mask_t = move(mask, USE_CUDA)
            vals_t = move(vals, USE_CUDA)

            res = net(args)

            #print(mask_t.shape, vals_t.shape, res.shape)
            true_val = vals_t[mask_t>0].reshape(-1, 1)
            loss = criterion(res[mask_t > 0], true_val)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(true_val)
            total_points +=  len(true_val)

            t2 = time.time()
            total_compute_time += (t2-t1)
            #ksjndask
        t2_total = time.time()

        print("{} {:3.3f} compute={:3.1f}s total={:3.1f}s".format(epoch_i, 
                                                                  running_loss/total_points, 
                                                                 total_compute_time, t2_total-t1_total))

        writer.add_scalar("train_loss", running_loss/total_points, epoch_i)

        if epoch_i % 5 == 0:
            net.eval()
            optimizer.zero_grad()
            allres = []
            alltrue = []
            for i_batch, (adj, vect_feat, mat_feat, vals, mask) in enumerate(dl_test):

                adj_t = move(adj, USE_CUDA)

                args = [adj_t]

                if 'vect' in data_feat:
                    vect_feat_t = move(vect_feat, USE_CUDA)
                    args.append(vect_feat_t)

                if 'mat' in data_feat:
                    mat_feat_t = move(mat_feat, USE_CUDA)
                    args.append(mat_feat_t)

                mask_t = move(mask, USE_CUDA)
                vals_t = move(vals, USE_CUDA)

                res = net(args)
                y_est = res[mask_t > 0].squeeze()
                y = vals_t[mask_t>0]
                allres.append(y_est.squeeze().detach().cpu().numpy())
                alltrue.append(y.cpu().detach().numpy())

            allres = np.concatenate(allres) 
            alltrue =  np.concatenate(alltrue)
            delta = allres - alltrue
            writer.add_scalar("test_std_err",  np.std(delta), epoch_i)
            writer.add_scalar("test_mean_abs_err",  np.mean(np.abs(delta)), epoch_i)
            writer.add_scalar("test_max_error",  np.max(np.abs(delta)), epoch_i)

            print(epoch_i, np.std(delta))

        if epoch_i % CHECKPOINT_EVERY == 0:
            checkpoint_filename = os.path.join(CHECKPOINT_DIR, "{}.{:08d}".format(MODEL_NAME, epoch_i))

            torch.save(net.state_dict(), checkpoint_filename + ".state")
            torch.save(net, checkpoint_filename + ".model")

NUC = '13C'
MAT_PROPS='aromatic'
DATASET_NAME = 'nmrshiftdb_hconfspcl_nmrshiftdb'

@follows(create_dataset)
@files('relnet_pipeline.data.{}.{}.{}.{}.mol_dict.pickle'.format(NUC, DATASET_NAME, 
                                                                  MAT_PROPS, 0), "test.out")
def train(infile, outfile):

    d = pickle.load(open(infile, 'rb'))

    train_df = d['train_df']
    test_df = d['test_df']
    
    USE_CUDA = True


    if 'hconfspcl' in DATASET_NAME:
        atomicno = [1, 6, 7, 8, 9, 15, 16, 17]
    else:
        atomicno = [1, 6, 7, 8, 9]
        
    ### Create datasets and data loaders

    feat_vect_args = dict(feat_atomicno_onehot=atomicno, 
                          feat_pos=False, feat_atomicno=True,
                          feat_valence=True, aromatic=True, hybridization=True, 
                          partial_charge=False, formal_charge=True,  # WE SHOULD REALLY USE THIS 
                          r_covalent=False,
                          total_valence_onehot=True, 
                          r_vanderwals=False, default_valence=True, rings=True)

    feat_mat_args = dict(feat_distances = False, 
                         feat_r_pow = None) #[-1, -2, -3])

    split_weights = [1, 1.5, 2, 3]
    adj_args = dict(edge_weighted=False, 
                    #norm_adj=True, add_identity=True, 
                    split_weights=split_weights)

    BATCH_SIZE = 64 # 16
    COMBINE_MAT_VECT='row'

    INIT_NOISE = 1e-3

    print("len(train_df)=", len(train_df))
    ds_train = MoleculeDatasetNew(train_df.rdmol.tolist(), train_df.value.tolist(),  
                                  MAX_N, feat_vect_args, 
                                  feat_mat_args, adj_args, 
                                  combine_mat_vect=COMBINE_MAT_VECT)
    print("len(ds_train)=", len(ds_train))
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, 
                                           shuffle=True,pin_memory=True)

    ds_test = MoleculeDatasetNew(test_df.rdmol.tolist(), test_df.value.tolist(), 
                                 MAX_N, feat_vect_args, feat_mat_args, adj_args, 
                                 combine_mat_vect=COMBINE_MAT_VECT)     
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE//2, 
                                          shuffle=True,pin_memory=True)


    USE_RESNET=True

    D_VERT = 128
    D_EDGE = 4
    LAYER_N = 5
    g_feature_n = 29 + len(atomicno) # + 3
    
    
    net = relnets.GraphEdgeVertPred(g_feature_n, len(split_weights), MAX_N,  layer_n=LAYER_N, 
                                    internal_d_vert=D_VERT, internal_d_edge=D_EDGE, 
                                    resnet=USE_RESNET, INIT_NOISE=INIT_NOISE)
    data_feat =['vect']
    net = move(net, USE_CUDA)

    #new_net.load_state_dict(new_state)
    for n, p in net.named_parameters():
        print(n, p.shape)

    WEIGHT_DECAY=1e-7

    criterion = nn.MSELoss()
    #criterion = lambda x, y : torch.mean((x-y)**4)
    amsgrad = False
    LR = 1e-2
    MODEL_NAME = "{}_{}_relnet_{}_{}_{}_{}_{}_{}_{}_{}_{}.{:08d}".format(NUC, DATASET_NAME, 
                                                                         LAYER_N, D_VERT, D_EDGE, 
                                                                         amsgrad, 
                                                                         BATCH_SIZE, LR, INIT_NOISE, WEIGHT_DECAY, 
                                                                    g_feature_n, 
                                                                    int(time.time() % 1e7))

    
    print("MODEL_NAME=", MODEL_NAME)
    writer = SummaryWriter("logs/{}".format(MODEL_NAME))

    # weight_decay appears to help a bit

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, 
                                 amsgrad=amsgrad, eps=1e-9, weight_decay=WEIGHT_DECAY) #, momentum=0.9)
    scheduler = None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)


    pickle.dump({"feat_vect_args" : feat_vect_args, 
                 "feat_mat_args" : feat_mat_args, 
                 "adj_args" : adj_args, 
                 "init_noise" : INIT_NOISE, 
                 "lr" : LR, 
                 'BATCH_SIZE' : BATCH_SIZE, 
                 'USE_RESNET' : USE_RESNET, 
                 'INIT_NOISE' : INIT_NOISE, 
                 'D_VERT' : D_VERT, 
                 'D_EDGE' : D_EDGE, 
                 'MAX_N' : MAX_N, 
                 'data_feat' : data_feat, 
                 'train_filename' : infile, 
                 'g_feature_n' : g_feature_n, 
                 'LAYER_N' : LAYER_N, 
                 }, open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".meta"), 'wb'))


    generic_runner(net, optimizer, scheduler, criterion, 
                   dl_train, dl_test, MODEL_NAME, 
                   MAX_EPOCHS=10000, CHECKPOINT_EVERY=10, 
                   data_feat=data_feat, USE_CUDA=USE_CUDA)

    
if __name__ == "__main__":
    pipeline_run([create_dataset, train])
