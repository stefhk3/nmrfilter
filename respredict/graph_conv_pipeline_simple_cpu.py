import numpy as np
import pickle
from ruffus import * 


import pickle
import os
import torch.utils.data
from glob import glob

import time
import nets

import torch
from torch import nn
#from tensorboardX import SummaryWriter


MAX_N = 32

TENSORBOARD_DIR = "logs"
CHECKPOINT_DIR = "checkpoints" 


def generic_runner(net, optimizer, criterion, 
                   dl_train, dl_test, MODEL_NAME, 
                   MAX_EPOCHS=1000, CHECKPOINT_EVERY=20, 
                   data_feat = ['mat'], USE_CUDA=True):

    #writer = SummaryWriter(f"{TENSORBOARD_DIR}/{MODEL_NAME}")
    if 'OMP_NUM_THREADS' in os.environ:
        print("OMP_NUM_THREADS=", os.environ['OMP_NUM_THREADS'])

    for epoch_i in range(MAX_EPOCHS):
        print("epoch_i=", epoch_i)
        running_loss = 0.0
        total_points = 0.0
        total_compute_time = 0.0
        t1_total = time.time()
        net.train()
        for i_batch, (adj, vect_feat, mat_feat, vals, mask) in \
            enumerate(dl_train):
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
            if i_batch % 100 == 0:
                print("{:5d}:{:d} {:3.3f}s".format(epoch_i, i_batch, t2-t1))

        t2_total = time.time()

        print("{} {:3.3f} compute={:3.1f}s total={:3.1f}s".format(epoch_i, 
                                                                  running_loss/total_points, 
                                                                 total_compute_time, t2_total-t1_total))
        continue
        #writer.add_scalar("train_loss", running_loss/total_points, epoch_i)

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
            #writer.add_scalar("test_std_err",  np.std(delta), epoch_i)
            #writer.add_scalar("test_max_error",  np.max(np.abs(delta)), epoch_i)

            print(epoch_i, np.std(delta))

        if epoch_i % CHECKPOINT_EVERY == 0:
            checkpoint_filename = os.path.join(CHECKPOINT_DIR, "{}.{:08d}".format(MODEL_NAME, epoch_i))

            torch.save(net.state_dict(), checkpoint_filename + ".state")
            torch.save(net, checkpoint_filename + ".model")
            

def move(tensor, cuda=False):

    if cuda:
        if isinstance(tensor, nn.Module):
            return tensor.cuda()
        else:
            return tensor.cuda(non_blocking=True)
    else:
        return tensor.cpu()

NUC = '13C'
MAT_PROPS='aromatic'
DATASET_NAME = 'nmrshiftdb_hconfspcl_nmrshiftdb'

#@follows(create_dataset)
# @files('graph_conv_pipeline.data.{}.{}.{}.{}.mol_dict.pickle'.format(NUC, DATASET_NAME, 
#                                                                   MAT_PROPS, 0), "test.out")

class MoleculeNpyDataset(torch.utils.data.Dataset):
    def __init__(self, base_filename):
        self.fields = ['adj', 'vect_feat', 'mat_feat', 
                                   'vals', 'mask']
        self.filenames = [base_filename + ".{}.{}.npy".format(fi, f) for fi, f in enumerate(self.fields)]
        self.data = {f : np.load(fn) for f, fn in zip(self.fields, self.filenames) }
        
    def __len__(self):
        return self.data['adj'].shape[0]
    
    def __getitem__(self, idx):
        return [self.data[f][idx] for f in self.fields]
        
    

@files(None, 
       "out.whatever")
def train(infiles, outfile):
    print("torch.__version__=", torch.__version__)
    
    USE_CUDA = True

    atomicno = [1, 6, 7, 8, 9, 15, 16, 17]
        
    # ### Create datasets and data loaders

    # feat_vect_args = dict(feat_atomicno_onehot=atomicno, 
    #                       feat_pos=False, feat_atomicno=True,
    #                       feat_valence=True, aromatic=True, hybridization=True, 
    #                       partial_charge=False, formal_charge=True,  # WE SHOULD REALLY USE THIS 
    #                       r_covalent=False,
    #                       r_vanderwals=False, default_valence=True, rings=True)

    # feat_mat_args = dict(feat_distances = False, 
    #                      feat_r_pow = None) #[-1, -2, -3])

    split_weights = [1, 1.5, 2, 3]
    # adj_args = dict(edge_weighted=False, 
    #                 norm_adj=True, add_identity=True, 
    #                 split_weights=split_weights)

    #BATCH_SIZE = 16
    BATCH_SIZE = 64
    # MASK_ZEROOUT_PROB = 0.0
    # COMBINE_MAT_VECT='row'

    INIT_NOISE = 1e-3

    # ds_train = MoleculeDatasetNew(train_df.rdmol.tolist(), train_df.value.tolist(),  
    #                               MAX_N, feat_vect_args, 
    #                               feat_mat_args, adj_args, 
    #                               combine_mat_vect=COMBINE_MAT_VECT, 
    #                               mask_zeroout_prob = MASK_ZEROOUT_PROB)   

    ds_train = MoleculeNpyDataset("ds_train")
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, 
                                           shuffle=True,pin_memory=True)

    # ds_test = MoleculeDatasetNew(test_df.rdmol.tolist(), test_df.value.tolist(), 
    #                              MAX_N, feat_vect_args, feat_mat_args, adj_args, 
    #                              combine_mat_vect=COMBINE_MAT_VECT)     

    ds_test = MoleculeNpyDataset("ds_test")

    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=BATCH_SIZE//2, 
                                          shuffle=True,pin_memory=True)


    USE_RESNET=True

    INT_D =  256 #  #1024 # 1024 # 32 # 1024 # 1024
    LAYER_N = 10
    g_feature_n = 21 + len(atomicno) # + 3
    USE_GRAPH_MAT = False
    GS = 1 #  len(split_weights)
    AGG_FUNC = nets.goodmax
    if USE_GRAPH_MAT:
        net = nets.GraphMatModel( g_feature_n,[INT_D] * LAYER_N, resnet=USE_RESNET, 
                                  noise=INIT_NOISE, GS=GS)
        data_feat = ['mat']
    else:
        net = nets.GraphVertModel( g_feature_n,[INT_D] * LAYER_N, resnet=USE_RESNET, 
                                   noise=INIT_NOISE, agg_func=AGG_FUNC, GS=GS)
        data_feat = ['vect']
    
    net = move(net, USE_CUDA)

    #new_net.load_state_dict(new_state)
    for n, p in net.named_parameters():
        print(n, p.shape)
    criterion = nn.MSELoss()
    amsgrad = False
    LR = 1e-3
    MASK_ZEROOUT_PROB=0.0
    MODEL_NAME = "{}_{}_graph_conv_{}_{}_{}_{}_{}_{}_{}_{}.{:08d}".format(NUC, DATASET_NAME, LAYER_N, INT_D, amsgrad, 
                                                                    MASK_ZEROOUT_PROB, BATCH_SIZE, LR, INIT_NOISE,
                                                                    g_feature_n, 
                                                                    int(time.time() % 1e7))


    #writer = SummaryWriter("logs/{}".format(MODEL_NAME))

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, 
                                 amsgrad=amsgrad, eps=1e-6) #, momentum=0.9)


    # pickle.dump({#"feat_vect_args" : feat_vect_args, 
    #              #"feat_mat_args" : feat_mat_args, 
    #     #"adj_args" : adj_args, 
    #              "init_noise" : INIT_NOISE, 
    #              "lr" : LR, 
    #              'BATCH_SIZE' : BATCH_SIZE, 
    #              'USE_RESNET' : USE_RESNET, 
    #              'INIT_NOISE' : INIT_NOISE, 
    #              'INT_D' : INT_D, 
    #              'AGG_FUNC' : AGG_FUNC, 
    #              #'data_feat' : data_feat, 
    #              #'train_filename' : infile, 
    #              'g_feature_n' : g_feature_n, 
    #              'USE_GRAPH_MAT' : USE_GRAPH_MAT, 
    #              'LAYER_N' : LAYER_N, 
    #              }, open(os.path.join(CHECKPOINT_DIR, MODEL_NAME + ".meta"), 'wb'))


    generic_runner(net, optimizer, criterion, 
                   dl_train, dl_test, MODEL_NAME, 
                   MAX_EPOCHS=10, CHECKPOINT_EVERY=10, 
                   data_feat=data_feat, USE_CUDA=USE_CUDA)

    
if __name__ == "__main__":
    pipeline_run([train])
