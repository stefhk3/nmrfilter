import pandas as pd
import numpy as np


def compute_stats(g, mol_id_field='mol_id', return_per_mol=False):
    """
    Compute statistics for groups g

    foodf.grouby(bar).apply(compute_stats)

    """
    per_mol = g.groupby( mol_id_field)\
                .agg({'delta_abs' : 'mean', 
                    'delta' : lambda x: np.sqrt(np.mean(x**2)), })\
                    .rename(columns={'delta_abs' : 'mol_MAE', 
                                        'delta' : 'mol_MSE'})
    b = g.groupby(mol_id_field).apply(lambda x: np.sqrt(np.mean(np.sort(x['value']) - np.sort(x['pred_mu']))**2))
    per_mol['sorted_mol_MSE'] = b

    b = g.groupby(mol_id_field).apply(lambda x: np.mean(np.abs(np.sort(x['value']) - np.sort(x['pred_mu']))))
    per_mol['sorted_mol_MAE'] = b

    res = per_mol.mean()
    res['mean_abs'] = g.delta_abs.mean()
    res['std'] = g.delta.std()
    res['n'] = len(g)
    res['mol_n'] = len(g[mol_id_field].unique())
    
    if return_per_mol:
        return res, per_mol
    else:
        return res


def sorted_bin_mean(conf, data, BIN_N=None, aggfunc = np.mean, bins=None):
    """
    Useful for sorting the data by confidence interval and binning. 

    imagine you want to plot error in estimate as a function of 
    confidence interval. you could just plot the rolling mean, 
    but at the very low confidence intervals you'll have very few
    points and so you'll get a high-variance estimator. 

    this basically bins the confidence regions such that you always
    have the same # of datapoints in each bin

    
    """
    conf = np.array(conf)
    data = np.array(data)
    sort_idx = np.argsort(conf)
    conf = conf[sort_idx]
    data = data[sort_idx]

    if bins is None:
        bins = np.linspace(0.0, 1.0, BIN_N)
    else:
        BIN_N = len(bins)
    tholds = np.array([conf[min(len(conf)-1, int(i*len(conf)))] for i in bins[1:]])

    m = np.array([aggfunc(data[conf <= tholds[i]]) for i in range(BIN_N-1)])

    frac_data = np.linspace(0.0, 1.0, len(m))
    return m, tholds, frac_data

