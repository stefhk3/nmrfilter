import pandas as pd
import numpy as np
from rdkit import Chem
import rdkit

def explode_df(df, lst_cols, fill_value=''):
    """
    Take a data frame with a column that's a list of entries and return
    one with a row for each element in the list
    
    From https://stackoverflow.com/a/40449726/1073963
    
    """
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, df[lst_cols[0]].str.len())
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens==0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]
            

def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

def split_df(dfm, chunk_size):
   """
   For splitting a df in to chunks of approximate size chunk_size
   """
   indices = index_marks(dfm.shape[0], chunk_size)
   return np.array_split(dfm, indices)


def array_to_conf(mat):
    """
    Take in a (N, 3) matrix of 3d positions and create
    a conformer for those positions. 
    
    ASSUMES atom_i = row i so make sure the 
    atoms in the molecule are the right order!
    
    """
    N = mat.shape[0]
    conf = Chem.Conformer(N)
    
    for ri in range(N):
        p = rdkit.Geometry.rdGeometry.Point3D(*mat[ri])                                      
        conf.SetAtomPosition(ri, p)
    return conf


