"""
Code to try and predict full spin systems
"""
import numpy as np
import torch
import atom_features
import molecule_features

from torch import nn
import torch
import torch.nn.functional as F
import torch.utils.data
import nets


class MoleculeDatasetMulti(torch.utils.data.Dataset):

    def __init__(self, mols, pred_vec_vals, pred_mat_vals, 
                 MAX_N, 
                 PRED_N = 1, 
                 feat_vert_args = {}, 
                 feat_edge_args = {}, 
                 adj_args = {}, combine_mat_vect = None, 
        ):
        self.mols = mols
        self.pred_vec_vals = pred_vec_vals
        self.pred_mat_vals = pred_mat_vals
        self.MAX_N = MAX_N
        self.cache = {}
        self.feat_vert_args = feat_vert_args
        self.feat_edge_args = feat_edge_args
        self.adj_args = adj_args
        #self.single_value = single_value
        self.combine_mat_vect = combine_mat_vect
        #self.mask_zeroout_prob = mask_zeroout_prob
        self.PRED_N = PRED_N

    def __len__(self):
        return len(self.mols)
    
    def mask_sel(self, v):
        return v

    def cache_key(self, idx, conf_idx):
        return (idx, conf_idx)

    def __getitem__(self, idx):

        mol = self.mols[idx]
        vect_pred_val = self.pred_vec_vals[idx]
        mat_pred_val = self.pred_mat_vals[idx]

        conf_idx = np.random.randint(mol.GetNumConformers())

        if self.cache_key(idx, conf_idx) in self.cache:
            return self.mask_sel(self.cache[self.cache_key(idx, conf_idx)])
        
        f_vect = atom_features.feat_tensor_atom(mol, conf_idx=conf_idx, 
                                                **self.feat_vert_args)
                                                
        DATA_N = f_vect.shape[0]
        
        vect_feat = np.zeros((self.MAX_N, f_vect.shape[1]), dtype=np.float32)
        vect_feat[:DATA_N] = f_vect

        f_mat = molecule_features.feat_tensor_mol(mol, conf_idx=conf_idx,
                                                  **self.feat_edge_args) 

        if self.combine_mat_vect:
            MAT_CHAN = f_mat.shape[2] + vect_feat.shape[1]
        else:
            MAT_CHAN = f_mat.shape[2]
        if MAT_CHAN == 0: # Dataloader can't handle tensors with empty dimensions
            MAT_CHAN = 1
        mat_feat = np.zeros((self.MAX_N, self.MAX_N, MAT_CHAN), dtype=np.float32)
        # do the padding
        mat_feat[:DATA_N, :DATA_N, :f_mat.shape[2]] = f_mat  
        
        if self.combine_mat_vect == 'row':
            # row-major
            for i in range(DATA_N):
                mat_feat[i, :DATA_N, f_mat.shape[2]:] = f_vect
        elif self.combine_mat_vect == 'col':
            # col-major
            for i in range(DATA_N):
                mat_feat[:DATA_N, i, f_mat.shape[2]:] = f_vect

        adj_nopad = molecule_features.feat_mol_adj(mol, **self.adj_args)
        adj = torch.zeros((adj_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj[:, :adj_nopad.shape[1], :adj_nopad.shape[2]] = adj_nopad
                        
        # create vect_mask and preds 
        
        vect_mask = np.zeros((self.MAX_N, self.PRED_N), 
                        dtype=np.float32)
        vect_vals = np.zeros((self.MAX_N, self.PRED_N), 
                             dtype=np.float32)
        #print(self.PRED_N, pred_val)
        for pn in range(self.PRED_N):
            for k, v in vect_pred_val[pn].items():
                vect_mask[int(k), pn] = 1.0
                vect_vals[int(k), pn] = v
        # create matrix mask and preds
        mat_mask = np.zeros((self.MAX_N, self.MAX_N, self.PRED_N), 
                        dtype=np.float32)
        mat_vals = np.zeros((self.MAX_N, self.MAX_N, self.PRED_N), 
                             dtype=np.float32)
        #print(self.PRED_N, pred_val)
        for pn in range(self.PRED_N):
            for (k1, k2), v in mat_pred_val[pn].items():
                mat_mask[int(k1), int(k2), pn] = 1.0
                mat_vals[int(k1), int(k2), pn] = v

                mat_mask[int(k2), int(k1), pn] = 1.0
                mat_vals[int(k2), int(k1), pn] = v
                
        # ADJ should be (N, N, features)
        adj = np.transpose(adj, axes=(1, 2, 0))
                

        v = (adj, vect_feat, mat_feat, 
             vect_vals, 
             vect_mask,
             mat_vals, 
             mat_mask)
        
        
        self.cache[self.cache_key(idx, conf_idx)] = v

        return self.mask_sel(v)




class GraphEdgeVertLayer(nn.Module):
    def __init__(self,vert_f_in, vert_f_out, 
                 edge_f_in, edge_f_out, 
                 out_func = F.relu, e_agg_func='sum'):
        """
        Note that to do the per-edge-combine-vertex layer we
        apply a per-vertex linear layer first and sum the result
        to the edge layer
        
        FIXME throw in some batch norms and some resnets because
            that appears to be a thing people do 
        """
        
        
        super(GraphEdgeVertLayer, self).__init__()
        
        self.edge_f_in = edge_f_in
        self.edge_f_out = edge_f_out
        self.vert_f_in = vert_f_in
        self.vert_f_out = vert_f_out
        
        self.e_vert_layer = nn.Linear(self.vert_f_in, self.edge_f_out)
        self.e_layer = nn.Linear(self.edge_f_in, self.edge_f_out)
        
        self.v_layer = nn.Linear(self.vert_f_in + self.edge_f_out, 
                                self.vert_f_out)
        
        self.e_agg = nets.parse_agg_func(e_agg_func)
        self.out_func = out_func
    
    def forward(self, v_in, e_in):
        BATCH_N = v_in.shape[0]
        assert v_in.shape[0] == e_in.shape[0]
        
        ### per-edge operations
        e_v = self.e_vert_layer(v_in)
        outer_v_sum = e_v.unsqueeze(1) + e_v.unsqueeze(2)
        e = self.e_layer(e_in)
        e_out = self.out_func(e + outer_v_sum)
        
        ### per-vertex operations
        per_v_e = self.e_agg(e_out, dim=1)
        vert_e_combined = torch.cat([per_v_e, v_in], dim=2)
        v = self.v_layer(vert_e_combined)
        v_out = self.out_func(v)
        
        return v_out, e_out
        
    



class GraphVertEdgeNet(nn.Module):
    def __init__(self, vert_f_in, edge_f_in, 
                 MAX_N, layer_n, internal_d_vert,
                 internal_d_edge, resnet=False, 
                 use_batchnorm = True, force_lin_init = False, 
                 e_agg_func='sum', 
                 init_noise = 0.01):
        """
        Note that to do the per-edge-combine-vertex layer we
        apply a per-vertex linear layer first and sum the result
        to the edge layer
        
        FIXME throw in some batch norms and some resnets because
            that appears to be a thing people do 
        """
        
        
        super(GraphVertEdgeNet, self).__init__()
        
        self.MAX_N = MAX_N
        self.vert_f_in = vert_f_in
        self.g1 = GraphEdgeVertLayer(vert_f_in, internal_d_vert, 
                                     edge_f_in, internal_d_edge,
                                     e_agg_func=e_agg_func)
        self.g_inner = nn.ModuleList([GraphEdgeVertLayer(internal_d_vert, 
                                                         internal_d_vert, 
                                                         internal_d_edge, 
                                                         internal_d_edge, e_agg_func=e_agg_func) for _ in range(layer_n)])
        self.g3 = GraphEdgeVertLayer(internal_d_vert, 1, internal_d_edge, 1, out_func = lambda x : x, 
                                     e_agg_func=e_agg_func)
        
        self.as_resnet = resnet
        
        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            self.bn_v_in = nn.BatchNorm1d(MAX_N * vert_f_in)
            self.bn_v_inner = nn.ModuleList([nn.BatchNorm1d(MAX_N, internal_d_vert) for _ in range(layer_n)])

        self.outLin = nn.Linear(MAX_N * MAX_N, 1)

        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_noise)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, v, e):
        if self.use_batchnorm:
            v = self.bn_v_in(v.view(-1, self.MAX_N * self.vert_f_in))
            v = v.view(-1, self.MAX_N, self.vert_f_in)
        v, e = self.g1(v, e)
        for i in range(len(self.g_inner)):
        
            v_next, e_next = self.g_inner[i](v, e)
            if self.as_resnet:
                v = v_next + v
                e = e_next + e
            else:
                v = v_next
                e = e_next
            if self.use_batchnorm:
                v = self.bn_v_inner[i](v)
        v, e = self.g3(v, e)
        return v, e



class GraphConvMatOut(nn.Module):
    def __init__(self, g_feature_n, g_feature_out_n=None, 
                 int_d = None, layer_n = None, 

                 resnet=True, 
                 init_noise=1e-5, agg_func=None, GS=1, OUT_DIM=1, 
                 batch_norm=False, out_std= False, 
                 fc_out_depth=1, 
                 resnet_out = False, resnet_blocks = (3, ), 
                 resnet_d = 128, 
                 graph_dropout=0.0, 
                 force_lin_init=False):
        
        """

        
        """
        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n


        super(GraphConvMatOut, self).__init__()
        self.gml = nets.GraphMatLayers(g_feature_n, g_feature_out_n, 
                                  resnet=resnet, noise=init_noise, agg_func=nets.parse_agg_func(agg_func), 
                                  GS=GS, dropout=graph_dropout)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(g_feature_n)
        else:
            self.batch_norm = None

        self.resnet_out = resnet_out
        if not resnet_out:
            end_lin_d = 32
            l = [nn.Sequential(nn.Linear(g_feature_out_n[-1], end_lin_d), nn.ReLU())]
            l += [nn.Sequential(nn.Linear(end_lin_d, end_lin_d), nn.ReLU()) for _ in range(fc_out_depth-1)]
            l += [nn.Linear(end_lin_d, OUT_DIM)]
            self.lin_out = nn.Sequential(*l)
                                                    
        else:
            self.lin_out = nets.ResNetRegression(g_feature_out_n[-1], 
                                                block_sizes = resnet_blocks, 
                                                INT_D = resnet_d, 
                                                FINAL_D=resnet_d, 
                                                OUT_DIM=OUT_DIM)




        if force_lin_init:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, init_noise)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x_G, G, return_g_features = False):

        # convert adj to last-col format
        G = torch.transpose(G, 1, 3)
        
        BATCH_N, MAX_N, F_N = x_G.shape

        if self.batch_norm is not None:
            x_G_flat = x_G.reshape(BATCH_N*MAX_N, F_N)
            x_G_out_flat = self.batch_norm(x_G_flat)
            x_G = x_G_out_flat.reshape(BATCH_N, MAX_N, F_N)
        
        G_features = self.gml(G, x_G)
        if return_g_features:
            return G_features

        v = G_features
        
        # there are a bunch of ways of doing this, and maybe we add some extra linear
        # layers here too
        e_v = v.unsqueeze(1) * v.unsqueeze(2)
        OUT_F = e_v.shape[-1]

        e_flat = e_v.reshape(BATCH_N*MAX_N*MAX_N, OUT_F)
        e_flat_out = self.lin_out(e_flat)
        e_out = e_flat_out.reshape(BATCH_N, MAX_N, MAX_N, -1)

        return None, e_out
        
