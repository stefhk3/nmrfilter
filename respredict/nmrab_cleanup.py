"""

spectra:
Separate files for each nucleus
have 'molecule_id', 'spectrum_id', 'atom_idx', 'value', 'id' [peak-id] fields

molecules: 
dataframe is 'df' field

subsets:
splits_df field


"""

import pickle
import pandas as pd
import numpy as np
from ruffus import * 
import os



DATASET_DIR = "dataset.named"
td = lambda x : os.path.join(DATASET_DIR, x)

AB_DIR = "../nmrabinitio/"
ts = lambda x : os.path.join(AB_DIR, x)

SHIFT_RANGES = {'13C' : (-10, 210), 
                '1H' : (-1, 10)}

@files(ts("sim.output/qm9.cheshire_g09_01_nmr.shifts.feather"), 
       (td("spectra.qm9.cheshire_g09_01_nmr.1H.feather"),
        td("spectra.qm9.cheshire_g09_01_nmr.13C.feather")))
def convert_shifts(infile, outfiles):
    outfile_h, outfile_c = outfiles

    shifts_infile = infile
    shifts_df = pd.read_feather(shifts_infile)

    for elt, nuc, outfile in [('C', '13C', outfile_c), 
                              ('H', '1H', outfile_h)]:
        shift_range = SHIFT_RANGES[nuc]
        s_df = shifts_df[shifts_df.element == elt].copy()
        s_df = s_df.rename(columns = {'id' : 'molecule_id', 'shift' : 'value'})
        s_df['nuc'] = nuc
        s_df['id'] = np.arange(len(s_df)) # peak id
        s_df['spectrum_id'] = 0

        # reject errant extreme values
        s_df = s_df[(s_df['value'] >= shift_range[0] ) & 
                    (s_df['value'] <= shift_range[1])]

        s_df = s_df.reset_index()

        
        del s_df['index']
        s_df.to_feather(outfile)
        
@files(ts("data/qm9.rdmol.pickle"), 
       td("molconf.qm9.pickle"))
def convert_mols(infile, outfile):
    df = pickle.load(open(infile, 'rb'))
    df = pd.DataFrame({'rdmol' : df})
    df = df[~pd.isnull(df.rdmol)]    
    df.index.name = 'mol_id'

    pickle.dump({'df' : df}, 
                open(outfile, 'wb'))
        

# infile = os.path.join(AB_DIR, "data/qm9.unique_subsets.feather")

@files(ts("data/qm9.unique_subsets.feather"),
       td("qm9.subsets.pickle"))
def convert_subsets(infile, outfile):
    
    df = pd.read_feather(infile).rename(columns={'mol_id' : 'molecule_id'}).set_index('molecule_id')
    pickle.dump({'splits_df' : df}, 
                open(outfile, 'wb'))

if __name__ == "__main__":
    pipeline_run([convert_shifts, convert_mols, convert_subsets])
