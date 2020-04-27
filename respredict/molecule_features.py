import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from numba import jit
import scipy.spatial
from rdkit import Chem
from rdkit.Chem import AllChem
from util import get_nos_coords
from atom_features import to_onehot


def feat_tensor_mol(mol, feat_distances=False,
                    feat_r_pow = None,
                    mmff_opt_conf = False,
                    is_in_ring=False,
                    is_in_ring_size = None, 
                    MAX_POW_M = 2.0, conf_idx = 0,
                    add_identity=False,
                    edge_type_tuples = [], 
                    norm_mat=False, mat_power=1):
    """
    Return matrix features for molecule
    
    """
    res_mats = []
    if mmff_opt_conf:
         Chem.AllChem.EmbedMolecule(mol)
         Chem.AllChem.MMFFOptimizeMolecule(mol)
    
    atomic_nos, coords = get_nos_coords(mol, conf_idx)
    ATOM_N = len(atomic_nos)

    if feat_distances:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = np.abs((a - a.T))
        c = np.swapaxes(b, 2, 1)
        res_mats.append(c)
    if feat_r_pow is not None:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = (a - a.T)**2
        c = np.swapaxes(b, 2, 1)
        d = np.sqrt(np.sum(c, axis=2))
        e = (np.eye(d.shape[0]) + d)[:, :, np.newaxis]

                       
        for p in feat_r_pow:
            e_pow = e**p
            if (e_pow > MAX_POW_M).any():
               # print("WARNING: max(M) = {:3.1f}".format(np.max(e_pow)))
                e_pow = np.minimum(e_pow, MAX_POW_M)

            res_mats.append(e_pow)

    if len(edge_type_tuples) > 0:
        a = np.zeros((ATOM_N, ATOM_N, len(edge_type_tuples)))
        for et_i, et in enumerate(edge_type_tuples):
            for b in mol.GetBonds():
                a_i = b.GetBeginAtomIdx()
                a_j = b.GetEndAtomIdx()
                if set(et) == set([atomic_nos[a_i], atomic_nos[a_j]]):
                    a[a_i, a_j, et_i] = 1
                    a[a_j, a_i, et_i] = 1
        res_mats.append(a)
        
    if is_in_ring:
        a = np.zeros((ATOM_N, ATOM_N, 1), dtype=np.float32)
        for b in mol.GetBonds():
            a[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
            a[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
        res_mats.append(a)
        
    if is_in_ring_size is not None:
        for rs in is_in_ring_size:
            a = np.zeros((ATOM_N, ATOM_N, 1), dtype=np.float32)
            for b in mol.GetBonds():
                if b.IsInRingSize(rs):
                    a[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
                    a[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
            res_mats.append(a)
            
        
            
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)
    else: # Empty matrix
        M = np.zeros((ATOM_N, ATOM_N, 0), dtype=np.float32)


    M = torch.Tensor(M).permute(2, 0, 1)
    
    if add_identity:
        M = M + torch.eye(ATOM_N).unsqueeze(0)

    if norm_mat:
        res = []
        for i in range(M.shape[0]):
            a = M[i]
            D_12 = 1.0 / torch.sqrt(torch.sum(a, dim=0))
            assert np.min(D_12.numpy()) > 0
            s1 = D_12.reshape(ATOM_N, 1)
            s2 = D_12.reshape(1, ATOM_N)
            adj_i = s1 * a * s2 

            if isinstance(mat_power, list):
                for p in mat_power:
                    adj_i_pow = torch.matrix_power(adj_i, p)

                    res.append(adj_i_pow)

            else:
                if mat_power > 1: 
                    adj_i = torch.matrix_power(adj_i, mat_power)

                res.append(adj_i)
        M = torch.stack(res, 0)
    #print("M.shape=", M.shape)
    assert np.isfinite(M).all()
    return M.permute(1, 2, 0) 

def mol_to_nums_adj(m, MAX_ATOM_N=None):# , kekulize=False):
    """
    molecule to symmetric adjacency matrix
    """

    m = Chem.Mol(m)

    # m.UpdatePropertyCache()
    # Chem.SetAromaticity(m)
    # if kekulize:
    #     Chem.rdmolops.Kekulize(m)

    ATOM_N = m.GetNumAtoms()
    if MAX_ATOM_N is None:
        MAX_ATOM_N = ATOM_N

    adj = np.zeros((MAX_ATOM_N, MAX_ATOM_N))
    atomic_nums = np.zeros(MAX_ATOM_N)

    assert ATOM_N <= MAX_ATOM_N

    for i in range(ATOM_N):
        a = m.GetAtomWithIdx(i)
        atomic_nums[i] = a.GetAtomicNum()

    for b in m.GetBonds():
        head = b.GetBeginAtomIdx()
        tail = b.GetEndAtomIdx()
        order = b.GetBondTypeAsDouble()
        adj[head, tail] = order
        adj[tail, head] = order
    return atomic_nums, adj



def feat_mol_adj(mol, 
                 edge_weighted=False, 
                 edge_bin = False,
                 add_identity=False,
                 norm_adj=False, split_weights = None, mat_power = 1):
    """
    Compute the adjacency matrix for this molecule

    If split-weights == [1, 2, 3] then we create separate adj matrices for those
    edge weights

    NOTE: We do not kekulize the molecule, we assume that has already been done

    """
    
    atomic_nos, adj = mol_to_nums_adj(mol)
    ADJ_N = adj.shape[0]
    input_adj = torch.Tensor(adj)
    
    adj_outs = []

    if edge_weighted:
        adj_weighted = input_adj.unsqueeze(0)
        adj_outs.append(adj_weighted)

    if edge_bin:
        adj_bin = input_adj.unsqueeze(0).clone()
        adj_bin[adj_bin > 0] = 1.0
        adj_outs.append(adj_bin)

    if split_weights is not None:
        split_adj = torch.zeros((len(split_weights), ADJ_N, ADJ_N ))
        for i in range(len(split_weights)):
            split_adj[i] = (input_adj == split_weights[i])
        adj_outs.append(split_adj)
    adj = torch.cat(adj_outs,0)

    if norm_adj and not add_identity:
        raise ValueError()
        
    if add_identity:
        adj = adj + torch.eye(ADJ_N)

    if norm_adj:
        res = []
        for i in range(adj.shape[0]):
            a = adj[i]
            D_12 = 1.0 / torch.sqrt(torch.sum(a, dim=0))

            s1 = D_12.reshape(ADJ_N, 1)
            s2 = D_12.reshape(1, ADJ_N)
            adj_i = s1 * a * s2 

            if isinstance(mat_power, list):
                for p in mat_power:
                    adj_i_pow = torch.matrix_power(adj_i, p)

                    res.append(adj_i_pow)

            else:
                if mat_power > 1: 
                    adj_i = torch.matrix_power(adj_i, mat_power)

                res.append(adj_i)
        adj = torch.stack(res)
    return adj



def whole_molecule_features(full_record, possible_solvents=[]):
    """
    return a vector of features for the full molecule 
    """
    out_feat = []
    if len(possible_solvents) > 0:
        out_feat.append(to_onehot(full_record['solvent'], possible_solvents))

    if len(out_feat) == 0:
        return torch.Tensor([])
    print(out_feat)
    return torch.Tensor(np.concatenate(out_feat).astype(np.float32))
