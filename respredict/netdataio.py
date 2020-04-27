import numpy as np
import torch
import pandas as pd
import atom_features
import molecule_features
import edge_features
import torch.utils.data
import util
from atom_features import to_onehot

class MoleculeDatasetMulti(torch.utils.data.Dataset):

    def __init__(self, mols, pred_vals, whole_records,
                 MAX_N, 
                 PRED_N = 1, 
                 feat_vert_args = {}, 
                 feat_edge_args = {}, 
                 adj_args = {},
                 mol_args = {}, 
                 combine_mat_vect = None,
                 combine_mat_feat_adj = False,
                 combine_mol_vect = False, 
                 extra_npy_filenames = [],
                 frac_per_epoch = 1.0,
                 shuffle_observations = False,
                 spect_assign = True,
                 extra_features = None,
                 allow_cache = True,
        ):
        self.mols = mols
        self.pred_vals = pred_vals
        self.whole_records = whole_records
        self.MAX_N = MAX_N
        if allow_cache:
            self.cache = {}
        else:
            self.cache = None
        self.feat_vert_args = feat_vert_args
        self.feat_edge_args = feat_edge_args
        self.adj_args = adj_args
        self.mol_args = mol_args
        #self.single_value = single_value
        self.combine_mat_vect = combine_mat_vect
        self.combine_mat_feat_adj = combine_mat_feat_adj
        self.combine_mol_vect = combine_mol_vect
        #self.mask_zeroout_prob = mask_zeroout_prob
        self.PRED_N = PRED_N

        self.extra_npy_filenames = extra_npy_filenames
        self.frac_per_epoch = frac_per_epoch
        self.shuffle_observations = shuffle_observations
        if shuffle_observations:
            print("WARNING: Shuffling observations")
        self.spect_assign = spect_assign
        self.extra_features = extra_features
        
    def __len__(self):
        return int(len(self.mols) * self.frac_per_epoch)
    

    def cache_key(self, idx, conf_idx):
        return (idx, conf_idx)

    def __getitem__(self, idx):
        if self.frac_per_epoch < 1.0:
            # randomly get an index each time
            idx = np.random.randint(len(self.mols))

        mol = self.mols[idx]
        pred_val = self.pred_vals[idx]
        whole_record = self.whole_records[idx]

        conf_idx = 0

        if self.cache is not None and self.cache_key(idx, conf_idx) in self.cache:
            return self.cache[self.cache_key(idx, conf_idx)]

        # mol features
        f_mol = molecule_features.whole_molecule_features(whole_record,
                                                          **self.mol_args)
        
        f_vect = atom_features.feat_tensor_atom(mol, conf_idx=conf_idx, 
                                                **self.feat_vert_args)
        if self.combine_mol_vect:
            f_vect = torch.cat([f_vect, f_mol.reshape(1, -1).expand(f_vect.shape[0], -1)], -1)
        # process extra data arguments
        for extra_data_config in self.extra_npy_filenames:
            filename = extra_data_config['filenames'][idx]
            combine_with = extra_data_config.get('combine_with', None)
            if combine_with == 'vert':
                npy_data = np.load(filename)
                npy_data_flatter = npy_data.reshape(f_vect.shape[0], -1)
                f_vect = torch.cat([f_vect, torch.Tensor(npy_data_flatter)], dim=-1)
            elif combine_with is None:
                continue
            else:

                raise NotImplementedError(f"the combinewith {combine_with} not working yet")
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


        if self.combine_mat_feat_adj:
            adj = torch.cat([adj, torch.Tensor(mat_feat).permute(2, 0, 1)], 0)
            
                

        ### Simple one-hot encoding for reconstruction
        adj_oh_nopad = molecule_features.feat_mol_adj(mol, split_weights=[1.0, 1.5, 2.0, 3.0], 
                                                      edge_weighted=False, norm_adj=False, add_identity=False)

        adj_oh = torch.zeros((adj_oh_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj_oh[:, :adj_oh_nopad.shape[1], :adj_oh_nopad.shape[2]] = adj_oh_nopad

        ## per-edge features
        feat_edge_dict = edge_features.feat_edges(mol, )
        
        # pad each of these
        edge_edge_nopad = feat_edge_dict['edge_edge']
        edge_edge = torch.zeros((edge_edge_nopad.shape[0], self.MAX_N, self.MAX_N))
        
        # edge_edge[:, :edge_edge_nopad.shape[1],
        #              :edge_edge_nopad.shape[2]] = torch.Tensor(edge_edge_nopad)
                
        edge_feat_nopad = feat_edge_dict['edge_feat']
        edge_feat = torch.zeros((self.MAX_N, edge_feat_nopad.shape[1]))
        # edge_feat[:edge_feat_nopad.shape[0]] = torch.Tensor(edge_feat_nopad)

        edge_vert_nopad = feat_edge_dict['edge_vert']
        edge_vert = torch.zeros((edge_vert_nopad.shape[0], self.MAX_N, self.MAX_N))
        # edge_vert[:, :edge_vert_nopad.shape[1],
        #              :edge_vert_nopad.shape[2]] = torch.Tensor(edge_vert_nopad)
                

        atomicnos, coords = util.get_nos_coords(mol, conf_idx)
        coords_t = torch.zeros((self.MAX_N, 3))
        coords_t[:len(coords), :] = torch.Tensor(coords)

        # create mask and preds 
        
        pred_mask = np.zeros((self.MAX_N, self.PRED_N), 
                        dtype=np.float32)
        vals = np.ones((self.MAX_N, self.PRED_N), 
                        dtype=np.float32) * util.PERM_MISSING_VALUE
        #print(self.PRED_N, pred_val)
        if self.spect_assign:
            for pn in range(self.PRED_N):
                if len(pred_val) > 0: # when empty, there's nothing to predict
                    atom_idx = [int(k) for k in pred_val[pn].keys()]
                    obs_vals = [pred_val[pn][i] for i in atom_idx]
                    # if self.shuffle_observations:
                    #     obs_vals = np.random.permutation(obs_vals)
                    for k, v in zip(atom_idx, obs_vals):
                        pred_mask[k, pn] = 1.0
                        vals[k, pn] = v
        else:
            if self.PRED_N > 1:
                raise NotImplementedError()
            vals[:] = util.PERM_MISSING_VALUE # sentinel value
            for k in pred_val[0][0]:
                pred_mask[k, 0] = 1
            for vi, v in enumerate(pred_val[0][1]):
                vals[vi] = v

        # input mask
        input_mask = torch.zeros(self.MAX_N) 
        input_mask[:DATA_N] = 1.0

        v = {'adj' : adj, 'vect_feat' : vect_feat, 
             'mat_feat' : mat_feat,
             'mol_feat' : f_mol, 
             'vals' : vals, 
             'adj_oh' : adj_oh,
             'pred_mask' : pred_mask,
             'coords' : coords_t, 
             'input_mask' : input_mask, 
             'input_idx' : idx,
             'edge_edge' : edge_edge,
             'edge_vert' : edge_vert,
             'edge_feat' : edge_feat}

        ## add on extra args 
        for ei, extra_data_config in enumerate(self.extra_npy_filenames):
            filename = extra_data_config['filenames'][idx]
            combine_with = extra_data_config.get('combine_with', None)
            if combine_with is None:
                ## this is an extra arg
                npy_data = np.load(filename)

                ## Zero pad
                npy_shape = list(npy_data.shape)
                npy_shape[0] = self.MAX_N
                t_pad = torch.zeros(npy_shape)
                t_pad[:npy_data.shape[0]] = torch.Tensor(npy_data)

                v[f'extra_data_{ei}'] = t_pad

        for k, kv in v.items():
            assert np.isfinite(kv).all()
        if self.cache is not None:
            self.cache[self.cache_key(idx, conf_idx)] = v

        return v

