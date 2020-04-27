import torch
from torch import nn
import torch_geometric.nn

import netdataio
import netutil
from netutil import move
import numpy as np


from torch_geometric.data import Dataset as GeomDataset
from torch_geometric.data import Data as GeomData
from torch_geometric.data import DataLoader as GeomDataLoader

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
import nets


class MyOwnDataset(GeomDataset):
    def __init__(self, input_ds, root="", transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.input_ds = input_ds
        self.cache = {}
        
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
        if idx in self.cache:
            return self.cache[idx]
        
        d = self.input_ds[idx]
        adj = d['adj']
        vect_feat = d['vect_feat']
        input_mask = d['input_mask']
        pred_mask = d['pred_mask']
        vals = d['vals']
        coords = d['coords']

        #print(adj.shape)
        have_edge, _ = adj.max(dim=0)
        #print("have_edge.shape=", have_edge.shape)
        edges = np.argwhere(have_edge.numpy())


        #print("adj.shape=", adj.shape)
        #print("edges.shape=", edges.shape)

        edge_attr = [adj[:, i, j] for i, j in edges]
        edge_type = torch.Tensor([np.argwhere(adj[:, i, j])[0] for i, j in edges]).long()


        edge_attr = torch.stack(edge_attr)
        #print("edge_attr.shape=", edge_attr.shape)

        edge_index_tensor = torch.Tensor(edges.T).long()
        #print("edge_index_tensor.shape=", edge_index_tensor.shape, "edge_index_tensor.dtype=", edge_index_tensor.dtype)
        
        y = torch.Tensor(vals) #.reshape(1, -1, 1)
        
        # FIXME maybe we need to select non-padded nodes here? 
        data = GeomData(x=torch.Tensor(vect_feat), 
                        edge_index=edge_index_tensor,
                        edge_type = edge_type,
                        input_idx = idx,
                        pos = coords, 
                        edge_attr=edge_attr, y=y)

        data.pred_mask = torch.Tensor(pred_mask) # .reshape(1, -1, 1)
        data.input_mask = torch.Tensor(input_mask)
        data.adj = torch.Tensor(adj )
        # print("pred_mask.shape=", pred_mask.shape, 
        #       " returning data.pred_mask.shape=", data.pred_mask.shape, 
        #       "y.shape=", y.shape)

        self.cache[idx] = data
        return data


def create_dl_wrapper(ds, batch_size,
                      shuffle, pin_memory=False):


    geom_ds = MyOwnDataset(ds)
    
    dl = GeomDataLoader(geom_ds,
                        batch_size=batch_size,
                        shuffle=shuffle)
    return dl
    
class ARMAMulti(torch.nn.Module):
    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 layer_n=4,
                 num_stacks=1,
                 num_layers=1,
                 out_mlp_layern=3,
                 dropout=0.0,
                 out_norm = None, 
                 mixture_n =10, 
                 int_d = 256, **kwargs):
        super(ARMAMulti, self).__init__()

        for k, v in kwargs.items():
            print(f"No attribute {k} : {v}")

        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.layer_n = layer_n
        self.int_d = int_d
        self.module_list = []
        self.lin_embed = nn.Linear(g_feature_n, self.int_d)
        
        self.input_bn = nn.BatchNorm1d(self.int_d)
        
        for i in range(self.layer_n):
            self.module_list.append(torch_geometric.nn.ARMAConv(self.int_d, self.int_d,
                                                                num_stacks=num_stacks,
                                                                num_layers=num_layers, 
                                                                bias=True, dropout=dropout))

        self.glayers = nn.ModuleList(self.module_list)

        self.lin_out = nn.ModuleList([MLPModel(self.int_d, out_mlp_layern,
                                               input_d = self.int_d,
                                               output_d = 1, norm=None)
                                      
                                      for _ in range(mixture_n)])
                    
        self.out_norm = out_norm
        if self.out_norm == 'layer':
            self.out_norm_layer = nn.LayerNorm(self.int_d)
        elif self.out_norm == 'batch':
            self.out_norm_layer = nn.BatchNorm1d(self.int_d) 
        

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        input_idx = data.input_idx
        

        x = self.input_bn(self.lin_embed(x))
        
        #print("0: x.shape=", x.shape)
        for li, l in enumerate(self.glayers):
            x = l(x, edge_index)
            if li != (len(self.glayers)-1):
                x = F.relu(x)
                
        if self.out_norm is not None:
            x = self.out_norm_layer(x)

        # reshape to the right size 
        vert_x = x.reshape(-1, self.MAX_N, x.shape[-1])
        
        #print("2: x.shape=", x.shape)
        mix_x = torch.stack([l(vert_x) for l in self.lin_out], 0)

        mu, std = nets.bootstrap_compute(mix_x,
                                         input_idx,
                                         training=self.training)
        
        return {'mu' : mu, 
                'std' : std}


class NNEdgeAttrs(torch.nn.Module):
    # def __init__(self):
    #     super(Net, self).__init__()


    #     nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
    #     self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
    #     self.gru = GRU(dim, dim)

    #     self.set2set = Set2Set(dim, processing_steps=3)
    #     self.lin1 = torch.nn.Linear(2 * dim, dim)
    #     self.lin2 = torch.nn.Linear(dim, 1)

    # def forward(self, data):
    #     out = F.relu(self.lin0(data.x))
    #     h = out.unsqueeze(0)

    #     for i in range(3):
    #         m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
    #         out, h = self.gru(m.unsqueeze(0), h)
    #         out = out.squeeze(0)

    #     out = self.set2set(out, data.batch)
    #     out = F.relu(self.lin1(out))
    #     out = self.lin2(out)
    #     return out.view(-1)


    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 GS=4,
                 layer_n=4,
                 int_d = 256, **kwargs):
        super(NNEdgeAttrs, self).__init__()

        for k, v in kwargs.items():
            print(f"No attribute {k} : {v}")

        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.layer_n = layer_n
        self.int_d = int_d

        self.lin0 = torch.nn.Linear(g_feature_n, int_d)
        edge_int_d = 16

        nn = Sequential(Linear(GS, edge_int_d), ReLU(), Linear(edge_int_d, int_d * int_d))
        self.conv = NNConv(int_d, int_d, nn, aggr='mean', root_weight=False)
        self.gru = GRU(int_d, int_d)

        self.lin_out = torch.nn.Linear(self.int_d, 1)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.layer_n):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        
        x = self.lin_out(out)

        out = x.reshape(-1, self.MAX_N, 1)
        #print("out.shape=", out.shape, "data.pred_mask.shape=", data.pred_mask.shape)
        return {'mu' : out, 
                'std' : torch.zeros_like(out)}




class Gated(torch.nn.Module):
    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 layer_n=4,
                 dropout=0.0,
                 mixture_n = 1,
                 out_norm = None, 
                 int_d = 256, **kwargs):
        super(Gated, self).__init__()

        for k, v in kwargs.items():
            print(f"No attribute {k} : {v}")

        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.layer_n = layer_n
        self.int_d = int_d
        self.module_list = []
        self.lin_embed = nn.Linear(g_feature_n, self.int_d)
        self.bn = nn.BatchNorm1d(self.int_d)
        

        self.gl = torch_geometric.nn.GatedGraphConv(self.int_d, layer_n)
        
        self.lin_out = nn.ModuleList([MLPModel(self.int_d, 3,
                                               input_d = self.int_d,
                                               output_d = 1, norm=None)
                                      
                                      for _ in range(mixture_n)])
                    
        self.out_norm = out_norm
        if self.out_norm == 'layer':
            self.out_norm_layer = nn.LayerNorm(self.int_d)
        elif self.out_norm == 'batch':
            self.out_norm_layer = nn.BatchNorm1d(self.int_d) 
        
        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        input_idx = data.input_idx
        #print("0: x.shape=", x.shape)
        x_embed = self.bn(self.lin_embed(x))
        
        
        x = self.gl(x_embed, edge_index)
        x = F.relu(x)
        if self.out_norm is not None:
            x = self.out_norm_layer(x)
            
        # reshape to the right size 
        vert_x = x.reshape(-1, self.MAX_N, x.shape[-1])
        
        #print("2: x.shape=", x.shape)
        mix_x = torch.stack([l(vert_x) for l in self.lin_out], 0)

        mu, std = nets.bootstrap_compute(mix_x, input_idx, training=self.training)
        
        return {'mu' : mu, 
                'std' : std}


    


    
class MLP(nn.Module):
    def __init__(self, layers, activate_final=True):
        super().__init__()
        self.layers = layers
        nets = []
        for i in range(len(layers)-1):
            
            nets.append(nn.Linear(layers[i], layers[i+1]))
            if i < (len(layers)-2) or activate_final:
                nets.append(nn.ReLU())

        #nets.append(torch.nn.Dropout(p=0.1))
        self.net = nn.Sequential(*nets)

    def forward(self, x):
        return self.net(x)

class MLPModel(nn.Module):
    def __init__(self, internal_d, layer_n, 
                 input_d = None, output_d = None, 
                 norm = 'batch', activate_final=False):
        super().__init__()
        
        LAYER_SIZES = [internal_d] * (layer_n+1)

        if input_d is not None:
            LAYER_SIZES[0] = input_d
        if output_d is not None:
            LAYER_SIZES[-1] = output_d

        self.mlp = MLP(LAYER_SIZES, activate_final=False)

        self.normkind = norm
        if self.normkind == 'batch':
            self.norm = nn.BatchNorm1d(LAYER_SIZES[-1])
        elif self.normkind == 'layer':
            self.norm = nn.LayerNorm(LAYER_SIZES[-1])

    def forward(self, x):
        y = self.mlp(x)
        if self.normkind == 'batch':
            BATCH_N = y.shape[0]
            F = y.shape[-1]
            
            y_batch_flat = y.reshape(-1, F)
            z_flat = self.norm(y_batch_flat)
            z = z_flat.reshape(y.shape)
            return z
        elif self.normkind == 'layer':
            return self.norm(y)
        else:
            return y

class Rel(torch.nn.Module):
    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 layer_n=4,
                 dropout=0.0,
                 int_d = 256, **kwargs):
        super(Rel, self).__init__()

        for k, v in kwargs.items():
            print(f"No attribute {k} : {v}")

        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.layer_n = layer_n
        self.int_d = int_d
        self.module_list = []
        self.lin_embed = nn.Linear(g_feature_n, self.int_d)
        
        self.input_bn = nn.BatchNorm1d(self.int_d)
        
        for i in range(self.layer_n):
            self.module_list.append(torch_geometric.nn.RGCNConv(self.int_d, self.int_d,
                                                                4, 4))
        self.glayers = nn.ModuleList(self.module_list)

        self.lin_out = MLPModel(self.int_d, 3,
                                input_d = self.int_d,
                                output_d = 1, norm=None)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_type

        x = self.input_bn(self.lin_embed(x))
        
        #print("0: x.shape=", x.shape)
        for l in self.glayers:
            x = l(x, edge_index, edge_attr)
            x = F.relu(x)
        #print("2: x.shape=", x.shape)
        x = self.lin_out(x)
        out = x.reshape(-1, self.MAX_N, 1)
        #print("out.shape=", out.shape, "data.pred_mask.shape=", data.pred_mask.shape)
        return {'mu' : out, 
                'std' : torch.zeros_like(out)}

        

class GMM(torch.nn.Module):
    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 layer_n=4,
                 dim = 4,
                 kernel_size = 4, 
                 int_d = 256, **kwargs):
        super(GMM, self).__init__()

        for k, v in kwargs.items():
            print(f"No attribute {k} : {v}")

        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.layer_n = layer_n
        self.int_d = int_d
        self.module_list = []
        self.lin_embed = nn.Linear(g_feature_n, self.int_d)
        
        self.input_bn = nn.BatchNorm1d(self.int_d)
        
        for i in range(self.layer_n):
            self.module_list.append(torch_geometric.nn.GMMConv(self.int_d, self.int_d,
                                                               dim=dim, kernel_size=dim,
                                                               bias=True))

        self.glayers = nn.ModuleList(self.module_list)

        self.lin_out = MLPModel(self.int_d, 3,
                                input_d = self.int_d,
                                output_d = 1, norm=None)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.input_bn(self.lin_embed(x))
        
        #print("0: x.shape=", x.shape)
        for l in self.glayers:
            x = l(x, edge_index)
            x = F.relu(x)
        #print("2: x.shape=", x.shape)
        x = self.lin_out(x)
        out = x.reshape(-1, self.MAX_N, 1)
        #print("out.shape=", out.shape, "data.pred_mask.shape=", data.pred_mask.shape)
        return {'mu' : out, 
                'std' : torch.zeros_like(out)}

    

class GAT(torch.nn.Module):
    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 heads=1,
                 head_d = 8,
                 dropout=0.0,
                 int_d = 256, **kwargs):
        super(GAT, self).__init__()

        for k, v in kwargs.items():
            print(f"No attribute {k} : {v}")

        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.int_d = int_d
        self.module_list = []
        self.lin_embed = nn.Linear(g_feature_n, self.int_d)
        self.bn = nn.BatchNorm1d(self.int_d)
        
        self.dropout = dropout
        self.gl = torch_geometric.nn.GATConv(self.int_d, head_d, heads=heads,
                                             dropout=dropout, concat=True)
        self.gl2 = torch_geometric.nn.GATConv(head_d * heads, int_d, concat=True,
                                              dropout=dropout)
        
        

        self.lin_out = nn.Linear(self.int_d, 1)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #print("0: x.shape=", x.shape)
        x_embed = F.dropout(self.bn(self.lin_embed(x)),
                            p = self.dropout, training=self.training)
        
        
        x = F.elu(self.gl(x_embed, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        #print("here x.shape=", x.shape)
        x = self.gl2(x, edge_index)

        #print(x.shape)
        #print("2: x.shape=", x.shape)
        x = self.lin_out(x)
        out = x.reshape(-1, self.MAX_N, 1)
        #print("out.shape=", out.shape, "data.pred_mask.shape=", data.pred_mask.shape)
        return {'mu' : out, 
                'std' : torch.zeros_like(out)}


    


class NNAttr(torch.nn.Module):
    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 layer_n=4,
                 out_mlp_layern=3,
                 out_norm = None,
                 aggr='mean',
                 mixture_n =10,
                 edge_latent_d = 128,
                 int_d = 256, **kwargs):
        super(NNAttr, self).__init__()

        for k, v in kwargs.items():
            print(f"No attribute {k} : {v}")

        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.layer_n = layer_n
        self.int_d = int_d
        self.module_list = []
        self.lin_embed = nn.Linear(g_feature_n, self.int_d)
        
        self.input_bn = nn.BatchNorm1d(self.int_d)

        dim = self.int_d

        nnet = nn.Sequential(nn.Linear(4, edge_latent_d),
                             
                             nn.Sigmoid(),
                             nn.Linear(edge_latent_d, dim*dim))
        self.conv = torch_geometric.nn.NNConv(dim, dim, nnet,
                                              aggr=aggr)
        self.gru = nn.GRU(dim, dim)
        

        self.lin_out = nn.ModuleList([MLPModel(self.int_d, out_mlp_layern,
                                               input_d = self.int_d,
                                               output_d = 1, norm=None)
                                      
                                      for _ in range(mixture_n)])
                    
        self.out_norm = out_norm
        if self.out_norm == 'layer':
            self.out_norm_layer = nn.LayerNorm(self.int_d)
        elif self.out_norm == 'batch':
            self.out_norm_layer = nn.BatchNorm1d(self.int_d) 
        

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        input_idx = data.input_idx
        

        x = self.input_bn(self.lin_embed(x))


        h = x.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(x, data.edge_index, data.edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = x.squeeze(0)

        if self.out_norm is not None:
            x = self.out_norm_layer(x)

        # reshape to the right size 
        vert_x = x.reshape(-1, self.MAX_N, x.shape[-1])
        
        #print("2: x.shape=", x.shape)
        mix_x = torch.stack([l(vert_x) for l in self.lin_out], 0)

        mu, std = nets.bootstrap_compute(mix_x,
                                         input_idx,
                                         training=self.training)
        
        return {'mu' : mu, 
                'std' : std}

    


class GIN(torch.nn.Module):
    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 layer_n=4,
                 out_mlp_layern=3,
                 train_eps=False, 
                 dropout=0.0,
                 inner_norm = 'batch',
                 mixture_n =10, 
                 int_d = 256, **kwargs):
        super(GIN, self).__init__()

        for k, v in kwargs.items():
            print(f"No attribute {k} : {v}")

        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.layer_n = layer_n
        self.int_d = int_d
        self.module_list = []
        self.lin_embed = nn.Linear(g_feature_n, self.int_d)
        
        self.input_bn = nn.BatchNorm1d(self.int_d)


        self.glayers = nn.ModuleList()
        self.normlayers = nn.ModuleList()

        dim = int_d
        for i in range(self.layer_n):
            
            nn1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.glayers.append(torch_geometric.nn.GINConv(nn1, train_eps=train_eps))
            if inner_norm == 'batch':
                self.normlayers.append(torch.nn.BatchNorm1d(dim))
            elif inner_norm == 'layer':
                self.normlayers.append(torch.nn.LayerNorm(dim))
        
        self.lin_out = nn.ModuleList([MLPModel(self.int_d, out_mlp_layern,
                                               input_d = self.int_d,
                                               output_d = 1, norm=None)
                                      
                                      for _ in range(mixture_n)])
                    
        # self.out_norm = out_norm
        # if self.out_norm == 'layer':
        #     self.out_norm_layer = nn.LayerNorm(self.int_d)
        # elif self.out_norm == 'batch':
        #     self.out_norm_layer = nn.BatchNorm1d(self.int_d) 
        

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        input_idx = data.input_idx
        

        x = self.input_bn(self.lin_embed(x))
        
        #print("0: x.shape=", x.shape)
        for li, l in enumerate(self.glayers):
            x = l(x, edge_index)
            x = self.normlayers[li](x)
            
        # reshape to the right size 
        vert_x = x.reshape(-1, self.MAX_N, x.shape[-1])
        
        #print("2: x.shape=", x.shape)
        mix_x = torch.stack([l(vert_x) for l in self.lin_out], 0)

        mu, std = nets.bootstrap_compute(mix_x,
                                         input_idx,
                                         training=self.training)
        
        return {'mu' : mu, 
                'std' : std}



    
class Edge(torch.nn.Module):

    def MLP(self, channels, batch_norm=True):
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]),
                          nn.ReLU(), nn.BatchNorm1d(channels[i]))
            for i in range(1, len(channels))
        ])
    
    def __init__(self, g_feature_n=-1, MAX_N=-1, 
                 force_lin_init=False, 
                 init_noise=0.0, 
                 #layer_n=4,
                 out_mlp_layern=3,
                 dropout=0.0,
                 k = 20,
                 aggr='max', 
                 #inner_norm = 'batch',
                 mixture_n =10, 
                 int_d = 256, **kwargs):
        super(Edge, self).__init__()


        IN_EDGE_DIM = 4
        self.MAX_N = MAX_N

        self.int_d = int_d
        self.module_list = []
        self.lin_embed = nn.Linear(g_feature_n, self.int_d)
        
        self.input_bn = nn.BatchNorm1d(self.int_d)



        dim = int_d
        self.conv1 = torch_geometric.nn.DynamicEdgeConv(self.MLP([2 * 3, 64, 64, 64]), k, aggr)
        self.conv2 = torch_geometric.nn.DynamicEdgeConv(self.MLP([2 * 64, 128]), k, aggr)
        self.lin1 = self.MLP([128 + 64, int_d])

        
        self.lin_out = nn.ModuleList([MLPModel(self.int_d, out_mlp_layern,
                                               input_d = self.int_d,
                                               output_d = 1, norm=None)
                                      
                                      for _ in range(mixture_n)])
                    
        # self.out_norm = out_norm
        # if self.out_norm == 'layer':
        #     self.out_norm_layer = nn.LayerNorm(self.int_d)
        # elif self.out_norm == 'batch':
        #     self.out_norm_layer = nn.BatchNorm1d(self.int_d) 
        

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if init_noise > 0:
                        nn.init.normal_(m.weight, 0, init_noise)
                    else:
                        print("xavier init")
                        nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, data):
        input_idx = data.input_idx
        
        x, pos, batch = data.x, data.pos, data.batch
        #x = self.input_bn(self.lin_embed(x))
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x = self.lin1(torch.cat([x1, x2], dim=1))

            
        # reshape to the right size 
        vert_x = x.reshape(-1, self.MAX_N, x.shape[-1])

        #print("2: x.shape=", x.shape)
        mix_x = torch.stack([l(vert_x) for l in self.lin_out], 0)

        mu, std = nets.bootstrap_compute(mix_x,
                                         input_idx,
                                         training=self.training)
        
        return {'mu' : mu, 
                'std' : std}

    
