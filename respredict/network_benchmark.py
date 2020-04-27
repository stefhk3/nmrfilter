
import numpy as np

import torch
from torch import nn
import pickle

import atom_features
from tqdm import tqdm
import pandas as pd
import torchvision.models

import threading
import queue
import util 
import time
import torchvision
import nets
from tensorboardX import SummaryWriter


infiles = ('dataset.named/spectra.nmrshiftdb_13C.feather', 
        'features.atomic/molconf.nmrshiftdb_hconf_nmrshiftdb.atom_neighborhood_vects.0.pickle', 
        'predict.atomic/molconf.nmrshiftdb_hconf_nmrshiftdb.subsets.pickle')
spectra_filename, features_filename, mol_subset_filename = infiles


mol_subsets = pickle.load(open(mol_subset_filename, 'rb'))['splits_df']

spectra_df = pd.read_feather(spectra_filename).rename(columns={'id' : 'peak_id'})
features_df = pickle.load(open(features_filename, 'rb'))['df']

# FIXME drop conformations except first
features_df = features_df[features_df.conf_i == 0].set_index(['mol_id', 'atom_idx'])

cv_sets = [range(5)]

#spectra_df = spectra_df[['molecule', 'spectrum_id', 'atom_idx', 'value']]
#with_features_df = spectra_df.join(features_df, on =['molecule', 'atom_idx']).dropna()

res = pd.merge(features_df, spectra_df, left_on=('mol_id', 'atom_idx'), right_on=('molecule_id', 'atom_idx'))
with_features_df = res[['molecule_id', 'conf_i', 'spectrum_id', 'atom_id', 'atom_idx', 'value', 
                        'feature' ]].copy()
with_features_df['idx'] = np.arange(len(with_features_df))

res = []
for cv_i, cv_mol_subset in enumerate(tqdm(cv_sets)):
    t1 = time.time()
    train_test_split = mol_subsets.subset20_i.isin(cv_mol_subset)
    train_mols = mol_subsets[~train_test_split].index.values
    test_mols = mol_subsets[train_test_split].index.values


    train_df = with_features_df[with_features_df.molecule_id.isin(train_mols)]
    test_df = with_features_df[with_features_df.molecule_id.isin(test_mols)]
    #print(len(train_df), len(test_df))
    break



class ConvRegression(nn.Module):

    def __init__(self, in_chan, in_pix):
        super(ConvRegression, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc = nn.Linear(16384, 1) # 512 * block.expansion *4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



train_idx = train_df.idx # [:20000]
test_idx = test_df.idx
X_data = with_features_df.feature.values
Y = np.array(with_features_df.value)


def proc_x(X):
    return np.stack([np.concatenate(x, axis=0) for x in X])



class MyDataset(torch.utils.data.Dataset):
    """
    This dataset contains a list of numbers in the range [a,b] inclusive
    """
    def __init__(self, X_data, idx, Y, debug=False):
        self.X_data = X_data
        self.idx = idx
        self.Y = Y
        self.debug = debug
        self.cache = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):

        if self.cache is not None:
            return self.cache[0].copy(), self.cache[1].copy()

        R = util.rand_rotation_matrix()
        X = row_feat_to_img(self.X_data[index], R)
        
        Xp = np.concatenate(X, axis=0)
        y = Y[index] # torch.Tensor(Y[idx])
        #Xp_tensor = torch.Tensor(Xp.astype(np.float32))
        #y_tensor = torch.Tensor(y.astype(np.float32))

        #xout, yout = Xp.astype(np.float32), y.astype(np.float32)
        xout, yout = Xp, y.astype(np.float32)
        if self.debug:
            self.cache = xout, yout

        return xout, yout
        
        #return  Xp_tensor, y_tensor # .cuda(), y_tensor.cuda()


CHAN_N = 15

gpu = 'v100' # 'k80' # v100'
if gpu =='v100':

    PIX_N = 64
    net = nets.PyTorchResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 
                             input_img_size = PIX_N, inplanes=64, 
                             first_kern_size=7, 
                             num_channels=CHAN_N, final_avg_pool_size=1 )
else:
    PIX_N = 32
    net = nets.PyTorchResNet(torchvision.models.resnet.BasicBlock, [2, 2], 
                             input_img_size = PIX_N, 
                             num_channels=CHAN_N, final_avg_pool_size=1 )
print("running", PIX_N, gpu)

def row_feat_to_img(rf, R=None):
    p = rf['pos']
    if R is not None:
        p = (R @ p.T).T
    img = atom_features.render_3view_points(rf['atomicno'], 
                                            p, [1, 6, 7, 8, 9], 
                                            0.1, PIX_N)
    return img

#net = net.cuda()


multigpu = False
if multigpu :
    BATCH_SIZE = 2048 # 2048 * 4
    net = nn.DataParallel(net).cuda()
    torch.backends.cudnn.benchmark = True

else:
    BATCH_SIZE = 512
    net = net.cuda()
    torch.backends.cudnn.benchmark = True
    
writer = SummaryWriter("logs2.13C/sphere_proc.2.{}".format(time.time()))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2) # , momentum=0.9) #, momentum=0.9)
#Xrad, Xang, Y = format_df(sub_df)

my_dataset = MyDataset(X_data, train_idx, Y)
#@profile
def run_stuff():
    

    

    data_train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=BATCH_SIZE,
                                                    pin_memory=True, shuffle=True, 
                                                    num_workers=16, drop_last=True)#. , num_workers=16)

    print("BATCH_SIZE=", BATCH_SIZE)
    print("An epoch is", len(my_dataset), "images")
    for epoch in range(5000):  # loop over the dataset multiple times
        total_points = 0
        t1 = time.time()
        running_loss = 0.0

        time_in_inner = 0
        for X_batch, y_batch in tqdm(data_train_loader):
            # get the inputs
            t1_inner = time.time()
            X_batch = X_batch.cuda(non_blocking=True)
            y_batch = y_batch.cuda(non_blocking=True)
            #print(type(X_batch), type(y_batch), X_batch.shape, y_batch.shape, \
            #      X_batch.device, y_batch.device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(X_batch.float())
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * len(y_batch)
            total_points += X_batch.shape[0]

            time_in_inner += (time.time() - t1_inner)
        t2 = time.time()
        writer.add_scalar("train_loss", running_loss/len(train_idx), epoch)

        if epoch % 1 == 0:
            print("epoch {:3d} took {:3.1f}s, time_in_inner= {:3.1f}s, {:3.1f} img/sec, loss={:3.1f}".format(epoch, t2-t1, time_in_inner, 
                                                                                               total_points / (t2-t1), 
                                                                                                             running_loss/len(train_idx)))
        if epoch % 5 == 0: # DEBUG
            test_idx_chunks = util.split_df(test_idx, BATCH_SIZE)
            test_res = []
            for idx in test_idx_chunks:

                X =np.stack([row_feat_to_img(X_data[i]) for i in idx])
                test_est = net(torch.Tensor(proc_x(X)).cuda()).detach().cpu().numpy().flatten()
                test_res.append(test_est)
            test_est =np.concatenate(test_res)
            delta = test_est - Y[test_idx]
            writer.add_scalar("test_std_err",  np.std(delta), epoch)

            print("std(delta)={:3.2f}".format(np.std(delta)))
        if epoch % 20 == 0:
            torch.save(net.state_dict(), "network_bench.model.{:08d}".format(epoch))


run_stuff()
