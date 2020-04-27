"""
Per-atom features : Featurizations that return one per atom
[in contrast to whole-molecule featurizations]
"""

import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from numba import jit
import scipy.spatial
from rdkit import Chem
from util import get_nos_coords


BP_RADIAL_DEFAULT_R_BINS=30
BP_RADIAL_DEFAULT_CENTERS = np.logspace(np.log10(0.8), np.log10(8.0), BP_RADIAL_DEFAULT_R_BINS)
BP_RADIAL_DEFAULT_WIDTHS = np.logspace(np.log10(0.8), np.log10(8.0), BP_RADIAL_DEFAULT_R_BINS)/256

def bp_local_weight(R, Rc):
    a =  0.5 * np.cos(np.pi * R / Rc) + 0.5
    a[a > Rc] = 0.0
    return a


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        
def pairwise_angle(pos):
    """
    manual, slow, compute angle between each pair of vectors
    
    """
    N = len(pos)
    true_angles = np.zeros((N, N))        
    for i in range(N):
        for j in range(N):
            true_angles[i, j] = angle_between(pos[i], pos[j])
    return true_angles


def custom_bp_radial (atomic_nos, coords, 
                      r_centers=None, bin_widths=None, ATOMIC_NUM_TO_POS=None, 
                      neighbor_cutoff = 6.0,  output_atoms = None):
    """
    compute BP radial-like features for each atom in atomic_nos
    uses the index order of the molecule
    """
    
    ### FIXME make all of these parameterized
    if ATOMIC_NUM_TO_POS is None:
        ATOMIC_NUM_TO_POS = {6: 0, 1: 1, 7: 2, 8: 3}

    if bin_widths is None:
        bin_widths = BP_RADIAL_DEFAULT_WIDTHS
    if r_centers is None:
        r_centers = BP_RADIAL_DEFAULT_CENTERS

    if output_atoms is None:
        output_atoms = np.arange(len(atomic_nos))
    OUTPUT_ATOM_N = len(output_atoms)


    R_BINS = len(r_centers)
    assert len(r_centers) == len(bin_widths)
    not_self_dist_eps = 1e-4
    
    dists = sklearn.metrics.pairwise.euclidean_distances(coords)
    featurization = np.zeros((OUTPUT_ATOM_N, len(ATOMIC_NUM_TO_POS), len(r_centers)))

    for output_atom_i, atom_i in enumerate(output_atoms):
        atomic_num = atomic_nos[atom_i]
        distances = dists[atom_i]

        bin_dists = sklearn.metrics.pairwise.euclidean_distances(distances.reshape(-1, 1) , 
                                                                 r_centers.reshape(-1, 1) , 
                                                                 squared=True)
        bin_scores = np.exp(-bin_dists/bin_widths)

        neighbor_distance_weights = bp_local_weight(distances, neighbor_cutoff)

        for neighbor_atomic_num, neighbor_distance, scores, weight in zip(atomic_nos, distances, 
                                                                           bin_scores, neighbor_distance_weights):
            if neighbor_distance < not_self_dist_eps:
                continue # don't use self
            if neighbor_atomic_num in ATOMIC_NUM_TO_POS:
                featurization[output_atom_i, ATOMIC_NUM_TO_POS[neighbor_atomic_num]] += scores * weight
    return featurization


def pairwise_angle_fast(X):

    a = np.inner(X, X)
    b = np.linalg.norm(X, axis=1)
    c = np.clip(a / np.maximum(np.outer(b, b),1e-6),  -1.0, 1.0)
    d = np.arccos(c)
    return d

def custom_bp_angular (atomic_nos, coords, neighbor_cutoff = 6.0, 
                        pairings = [(6, 1), (6, 6)], rses= None, thetas=None, 
                      zeta=16.0, eta=2.0, output_atoms = None):
    ATOM_N = len(atomic_nos)Scoords
    
    for p1, p2 in pairings:
        if p1 > p2:
            raise ValueError("parings must list the lighter atom first")

    PAIRS_N = len(pairings)

    coords = coords.astype(np.float32)

    if thetas is None:
        thetas = np.linspace(0, np.pi, 5)
    thetas = thetas.astype(np.float32)
    if rses is None:
        rses = np.array([1.0, 2.0, 3.0])

    rses = rses.astype(np.float32)

    if output_atoms is None:
        output_atoms = np.arange(ATOM_N)
    OUTPUT_ATOM_N = len(output_atoms)

    output = np.zeros((OUTPUT_ATOM_N, len(pairings), len(thetas), len(rses)), dtype=np.float32)

    pairing_mask = np.zeros((PAIRS_N, ATOM_N, ATOM_N)).astype(np.float32)

    for i in range(ATOM_N):
        for j in range(i+1, ATOM_N):
            for pair_i, (atom_1, atom_2) in enumerate(pairings):
                true_atom_1 = atomic_nos[i]
                true_atom_2 = atomic_nos[j]

                if true_atom_1 > true_atom_2:
                    true_atom_1, true_atom_2 = true_atom_2, true_atom_1
                if (atom_1 == true_atom_1 ) & (atom_2 == true_atom_2):
                    pairing_mask[pair_i, i, j] = 1.0

    for out_atom_i, atom_i in enumerate(output_atoms):
        vects = coords - coords[atom_i]
        angles = pairwise_angle_fast(vects).astype(np.float32)

        #angles = featurize.pairwise_angle(vects)
        dists = np.linalg.norm(vects, axis=1)
        pair_sum_dists = np.add.outer(dists, dists)
        local_weights = bp_local_weight(dists, neighbor_cutoff)

        angular_terms = np.zeros((len(thetas), ATOM_N, ATOM_N))
        for theta_i, theta_s in enumerate(thetas):
            angular_terms[theta_i] = (1+ np.cos(angles - theta_s))**zeta
        dist_terms = np.zeros((len(rses), ATOM_N, ATOM_N))
        for rs_i, rs in enumerate(rses):
            dist_terms[rs_i] = np.exp(-eta * (pair_sum_dists/2.0 - rs)**2)

        local_weights_outer =  np.outer(local_weights, local_weights)
                
        for theta_i, theta_s in enumerate(thetas):
            angular_term = angular_terms[theta_i]

            for rs_i, rs in enumerate(rses):
                dist_term = dist_terms[rs_i] 

                scores = np.nan_to_num(2.0**(1.0 - zeta) * angular_term * dist_term * local_weights_outer)


                output[out_atom_i, :, theta_i, rs_i] += np.sum(scores * pairing_mask, axis=(1, 2))# [pair_i])
    return np.nan_to_num(output)


def torch_featurize_mat_pow(atomic_nums, adj, MAX_ATOM_N, MAX_POWER=2, 
                            UNIQUE_ATOMIC_NUMS=[1, 6, 7, 8, 9, 15, 16, 17]):
    # compute matrix powers
    A = torch.eye(MAX_ATOM_N)
    out_features = []
    for mp in range(1, MAX_POWER + 1):
        A = A.mm(adj)
        for element_i, atomic_num in enumerate(UNIQUE_ATOMIC_NUMS):
            row_mask = (atomic_nums == atomic_num).float()
            out_features.append(torch.sum(A * row_mask, dim=1))
    return torch.stack(out_features).transpose(0, 1)

def bp_radial_image(atomic_nos, coords, r_bin_edges=None, 
                 atomic_num_rows=None, output_atoms=None):
    if atomic_num_rows is None:
        atomic_num_rows = [6, 1, 7, 8, 9]
    if r_bin_edges is None:
        r_bin_edges = np.linspace(0.0, 5.0, 32)
    ATOMIC_NUM_TO_ROW = {k: i for i, k in enumerate(atomic_num_rows)}

    if output_atoms is None:
        output_atoms = np.arange(len(atomic_nos))
    OUTPUT_ATOM_N = len(output_atoms)
    #print("here 2", coords.shape, coords.dtype)
    #print(coords)

    dists = sklearn.metrics.pairwise.euclidean_distances(coords)
    #dists =scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(coords))
    #print("here 2.5", dists.shape)
    featurization = np.zeros((OUTPUT_ATOM_N, len(atomic_num_rows), len(r_bin_edges)), dtype=np.uint8)
    #print("here 3")
    for output_atom_i, atom_i in enumerate(output_atoms):
        atomic_num = atomic_nos[atom_i]
        distances = dists[atom_i]
        idx = np.searchsorted(r_bin_edges, distances, side='right')
        for ii, (a, d, i) in enumerate(zip(atomic_nos, distances, idx)):
            if a in ATOMIC_NUM_TO_ROW and i < len(r_bin_edges) and ii != atom_i and d > 0:
                featurization[output_atom_i, ATOMIC_NUM_TO_ROW[a], i] += 1
    return featurization      


def bp_angular_images (atomic_nos, coords,
                    pairings = [(1, 6), (6, 6)], r_bin_edges= None, theta_bin_edges=None, 
                    output_atoms = None):
    
    """
    for the indicated output atoms, return a len(pairing)-channel image
    of thetas by mean dist 

    """
    ATOM_N = len(atomic_nos)
    
    for p1, p2 in pairings:
        if p1 > p2:
            raise ValueError("parings must list the lighter atom first")

    PAIRS_N = len(pairings)

    coords = coords.astype(np.float32)
    
    
    if output_atoms is None:
        output_atoms = np.arange(len(atomic_nos))

    if theta_bin_edges is None:
        theta_bin_edges = np.linspace(0, np.pi*1.01, 17)

    if r_bin_edges is None:
        r_bin_edges = np.linspace(0.5, 4.5, 17)

    THETA_BIN_N = len(theta_bin_edges)-1
    R_BIN_N = len(r_bin_edges) - 1
        
    out_images = np.zeros((len(output_atoms), THETA_BIN_N, 
                           R_BIN_N, len(pairings)), 
                           dtype=np.uint8)
    pair_to_pos = {p : i for i, p in enumerate(pairings)}

    
    for out_atom_i, atom_i in enumerate(output_atoms):
        vects = coords - coords[atom_i]
        angles = pairwise_angle_fast(vects).astype(np.float32)
        dists = np.linalg.norm(vects, axis=1)
        pair_sum_dists = np.add.outer(dists, dists)
        theta_bin_vals = np.searchsorted(theta_bin_edges, angles) -1
        theta_bin_vals[theta_bin_vals >= (len(theta_bin_edges)-1)] = -1
        r_bin_vals = np.searchsorted(r_bin_edges, pair_sum_dists/2.0) -1
        r_bin_vals[r_bin_vals >= (len(r_bin_edges)-1)] = -1 
        _angular_images_fast_loop(ATOM_N, atom_i, out_atom_i, atomic_nos, 
                  #pair_to_pos, 
                  pairings, 
                  theta_bin_vals, r_bin_vals, out_images)
        # for i1 in range(ATOM_N):
        #     if i1 == atom_i :
        #         continue
        #     for i2 in range(i1+1, ATOM_N):
        #         if i2 == atom_i:
        #             continue

        #         theta_idx = theta_bin_vals[i1, i2]
        #         if theta_idx == -1:
        #             continue

        #         dist_idx = r_bin_vals[i1, i2]
        #         if dist_idx == -1:
        #             continue

        #         a1_num = atomic_nos[i1]
        #         a2_num = atomic_nos[i2]
        #         if a1_num > a2_num:
        #             a1_num, a2_num = a2_num, a1_num

        #         pi = pair_to_pos.get((a1_num, a2_num), None)
        #         if pi is not None:
        #             out_images[out_atom_i, theta_idx, dist_idx, pi] +=1

    return out_images

@jit(nopython=True)
def _angular_images_fast_loop(ATOM_N, atom_i, out_atom_i, atomic_nos, #pair_to_pos,
              pairings, 
              theta_bin_vals, r_bin_vals, out_images):
    for i1 in range(ATOM_N):
        if i1 == atom_i :
            continue
        for i2 in range(i1+1, ATOM_N):
            if i2 == atom_i:
                continue

            theta_idx = theta_bin_vals[i1, i2]
            if theta_idx < 0:
                continue

            dist_idx = r_bin_vals[i1, i2]
            if dist_idx < 0:
                continue

            a1_num = atomic_nos[i1]
            a2_num = atomic_nos[i2]
            if a1_num > a2_num:
                a1_num, a2_num = a2_num, a1_num

            # pi = pair_to_pos.get((a1_num, a2_num), None)
            # if pi is not None:
            #     out_images[out_atom_i, theta_idx, dist_idx, pi] +=1
            for pi, p in enumerate(pairings):
                if (a1_num, a2_num) == p:
                     out_images[out_atom_i, theta_idx, dist_idx, pi] +=1
                     break

@jit(nopython=True, nogil=True)
def render_3view_points(atomic_nos, vects, mol_chans, bin_width, bin_number, 
                        bound_in_3d=True):
    """
    Return a 3 x mol_chans image evaluated at bins
    note: output is uint8
    """
    CHAN_N = len(mol_chans)
    ATOM_N = len(atomic_nos)
    img = np.zeros((3, CHAN_N, bin_number, bin_number), dtype=np.uint8)
    mol_lut = np.ones(128, dtype=np.int8) * -1
    for i in range(CHAN_N):
        mol_lut[mol_chans[i]] = i
    vects_int = np.trunc(vects / float(bin_width))
    for i in range(ATOM_N):
        num = atomic_nos[i]
        b = vects_int[i] + float(bin_number // 2)
        p = np.trunc(b).astype(np.int64)

        if mol_lut[num] < 0:
            continue

        if bound_in_3d: # check if in bounding box
            if not (0 <= p[0] < bin_number) and (0 <= p[1] < bin_number) and (0 <= p[2] < bin_number) :
                continue
            
        # coord1 / coord2 
        if (0 <= p[0] < bin_number) and (0 <= p[1] < bin_number): 
            img[0, mol_lut[num], p[0], p[1]] += 1
            
        if (0 <= p[1] < bin_number) and (0 <= p[2] < bin_number): 
            img[1, mol_lut[num], p[1], p[2]] += 1
            
        if (0 <= p[0] < bin_number) and (0 <= p[2] < bin_number): 
            img[2, mol_lut[num], p[0], p[2]] += 1
    return img

def atom_adj_mat(mol, conformer_i, **kwargs):
    """
    OUTPUT IS ATOM_N x (adj_mat, tgt_atom, atomic_nos, dists )
    
    This is really inefficient given that we explicitly return the same adj
    matrix for each atom, and index into it
    
    Adj mat is valence number * 2
    
    
    """
    
    MAX_ATOM_N = kwargs.get('MAX_ATOM_N', 64)
    atomic_nos, coords = get_nos_coords(mol, conformer_i)
    ATOM_N = len(atomic_nos)

    atomic_nos_pad, adj = mol_to_nums_adj(mol, MAX_ATOM_N)
    

    features = np.zeros((ATOM_N,), 
                    dtype=[('adj', np.uint8, (MAX_ATOM_N, MAX_ATOM_N)), 
                           ('my_idx', np.int), 
                           ('atomicno', np.uint8, MAX_ATOM_N), 
                           ('pos', np.float32, (MAX_ATOM_N, 3,))])

    
    
    for atom_i in range(ATOM_N):
        vects = coords - coords[atom_i]
        features[atom_i]['adj'] = adj*2
        features[atom_i]['my_idx'] =  atom_i
        features[atom_i]['atomicno'] = atomic_nos_pad
        features[atom_i]['pos'][:ATOM_N] = vects
    return features

def advanced_atom_props(mol, conformer_i, **kwargs):
    import rdkit.Chem.rdPartialCharges
    pt = Chem.GetPeriodicTable()
    atomic_nos, coords = get_nos_coords(mol, conformer_i)
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol, Chem.rdmolops.SanitizeFlags.SANITIZE_ALL, 
                     catchErrors=True)
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    ATOM_N = len(atomic_nos)
    out = np.zeros(ATOM_N, 
                   dtype=[('total_valence', np.int),  
                          ('aromatic', np.bool), 
                          ('hybridization', np.int), 
                          ('partial_charge', np.float32),
                          ('formal_charge', np.float32), 
                          ('atomicno', np.int), 
                          ('r_covalent', np.float32),
                          ('r_vanderwals', np.float32),
                          ('default_valence', np.int),
                          ('rings', np.bool, 5), 
                          ('pos', np.float32, 3)])
    
      
    for i in range(mol.GetNumAtoms()):
        a = mol.GetAtomWithIdx(i)
        atomic_num = int(atomic_nos[i])
        out[i]['total_valence'] = a.GetTotalValence()
        out[i]['aromatic'] = a.GetIsAromatic()
        out[i]['hybridization'] = a.GetHybridization()
        out[i]['partial_charge'] = a.GetProp('_GasteigerCharge')
        out[i]['formal_charge'] = a.GetFormalCharge()
        out[i]['atomicno'] = atomic_nos[i]
        out[i]['r_covalent'] =pt.GetRcovalent(atomic_num)
        out[i]['r_vanderwals'] =  pt.GetRvdw(atomic_num)
        out[i]['default_valence'] = pt.GetDefaultValence(atomic_num)
        out[i]['rings'] = [a.IsInRingSize(r) for r in range(3, 8)]
        out[i]['pos'] = coords[i]
                          
    return out

