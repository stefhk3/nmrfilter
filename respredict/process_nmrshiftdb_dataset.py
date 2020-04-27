import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors as rdMD
import pickle
import util

import sqlalchemy
from sqlalchemy import sql, func
import dbconfig
from ruffus import *


### first filter through all the molecules that have the indicated
### kind of spectra

### then select on those spectra

### emit the resulting feather files with smiles, canonical spectra conditioning, spectra array

HCONF = ['H', 'C', 'O', 'N', 'F']
HCONFSPCl = ['H', 'C', 'O', 'N', 'F', 'S', 'P', 'Cl']

DEFAULT_CONFIG = {                  
    'max_atom_n' : 64, 
    'max_heavy_atom_n' : 64, 
    'elements' : HCONF, 
    'allow_radicals' : False, 
    'allow_atom_formal_charge' : False, 
    'max_ring_size' : 14, 
    'min_ring_size' : 2, 
    'allow_unknown_hybridization' : False, 
    'spectra_nuc' : ['13C', '1H'],
    'allow_mol_formal_charge' : False
}

DATASETS = {
    'nmrshiftdb_64_64_HCONF_13C' : {'source' : ['nmrshiftdb'], 
                                    'spectra_nuc' : ['13C']}, 
    'nmrshiftdb_64_64_HCONF_1H' : {'source' : ['nmrshiftdb'], 
                                   'spectra_nuc' : ['1H']}, 
    'nmrshiftdb_64_64_HCONFSPCl_13C' : {'source' : ['nmrshiftdb'], 
                                        'spectra_nuc' : ['13C'], 
                                        'elements' : HCONFSPCl}, 
    'nmrshiftdb_64_64_HCONFSPCl_1H' : {'source' : ['nmrshiftdb'], 
                                       'spectra_nuc' : ['1H'], 
                                       'elements' : HCONFSPCl}, 

    'nmrshiftdb_128_128_HCONFSPCl_13C' : {'source' : ['nmrshiftdb'], 
                                          'spectra_nuc' : ['13C'], 
                                          'max_atom_n' : 128, 
                                          'max_hevy_atom_n' : 128, 
                                          'elements' : HCONFSPCl}, 
    'nmrshiftdb_128_128_HCONFSPCl_1H' : {'source' : ['nmrshiftdb'], 
                                         'spectra_nuc' : ['1H'], 
                                         'max_atom_n' : 128, 
                                         'max_hevy_atom_n' : 128, 
                                         'elements' : HCONFSPCl}, 

    'gissmo_128_128_HCONFSPCl_1H' : {'source' : ['gissmo'], 
                                     'spectra_nuc' : ['1H'], 
                                     'max_atom_n' : 128, 
                                     'max_hevy_atom_n' : 128, 
                                     'elements' : HCONFSPCl}, 
    'nmrshiftdb_128_128_HCONFSPCl_wcharge_13C' : {'source' : ['nmrshiftdb'], 
                                                  'spectra_nuc' : ['13C'], 
                                                  'max_atom_n' : 128, 
                                                  'max_hevy_atom_n' : 128, 
                                                  'elements' : HCONFSPCl,
                                                  'allow_atom_formal_charge' : True,
                                                  'allow_mol_formal_charge' : True}, 
    'nmrshiftdb_128_128_HCONFSPCl_wcharge_1H' : {'source' : ['nmrshiftdb'], 
                                                 'spectra_nuc' : ['1H'], 
                                                 'max_atom_n' : 128, 
                                                 'max_hevy_atom_n' : 128, 
                                                 'elements' : HCONFSPCl,
                                                 'allow_atom_formal_charge' : True,
                                                 'allow_mol_formal_charge' : True}, 

    
            }


SOURCE_DIR = os.path.dirname(os.path.realpath(__file__))


#DB_URL = f"sqlite:////data/ericj/nmr/nmrdata/nmrdata.db"
DB_URL = f"sqlite:////data/ericj/spectdata/nmrdata.db"

OUTPUT_DIR = "processed_data" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params_shifts():
    for exp_name, ec in DATASETS.items():
        config = DEFAULT_CONFIG.copy()
        config.update(ec)

        yield (None, 
               (td("{}.shifts.mol.feather".format(exp_name)), 
                td("{}.shifts.spect.feather".format(exp_name)),  
                td("{}.meta.pickle".format(exp_name))),
               config)

@mkdir(OUTPUT_DIR)
@files(params_shifts)
def preprocess_data_shifts(infile, outfile, config):            
    
    mol_outfile, spect_outfile, meta_outfile = outfile
    ### construct the query


    engine = sqlalchemy.create_engine(DB_URL)
    conn = engine.connect()

    meta = sqlalchemy.MetaData()

    molecules = sqlalchemy.Table('molecule', meta, autoload=True, 
                                autoload_with=engine)
    spectra = sqlalchemy.Table('spectrum_meta', meta, autoload=True, 
                                autoload_with=engine)
    peaks = sqlalchemy.Table('peak', meta, autoload=True, 
                                autoload_with=engine)

    couplings = sqlalchemy.Table('coupling', meta, autoload=True, 
                                autoload_with=engine)


    stmt = sql.select([molecules.c.id, molecules.c.source_id, molecules.c.source, molecules.c.mol])\
           .where(molecules.c.id == spectra.c.molecule_id)
    ## filter by source
    if 'source' in config:
        stmt = stmt.where(molecules.c.source.in_(config['source']))

    stmt = stmt.where(spectra.c.nucleus.in_(config['spectra_nuc']))

    stmt = stmt.distinct()
    print(str(stmt))

    output_mol_df, skip_reason_df = util.filter_mols(conn.execute(stmt), 
                                                     config, other_attributes = ['source', 'source_id', 'mol'])
    print(output_mol_df.head())
    output_mol_df.to_feather(mol_outfile)

    # now we select the spectra
    stmt = sql.select([peaks.c.id, peaks.c.atom_idx, peaks.c.multiplicity, 
                       peaks.c.value, peaks.c.spectrum_id, spectra.c.molecule_id])\
                      .where(peaks.c.spectrum_id == spectra.c.id) \
                      .where(spectra.c.nucleus.in_(config['spectra_nuc']))\
                      .where(spectra.c.molecule_id.in_(output_mol_df.molecule_id))
    peak_df = pd.read_sql(stmt, engine)
    peak_df.to_feather(spect_outfile)
    #skip_reason_df = pd.DataFrame(skip_reason)

    print(skip_reason_df.reason.value_counts())

    pickle.dump({'skip_reason_df' : skip_reason_df, 
                 'config' : config},
                open(meta_outfile, 'wb'))


@transform(preprocess_data_shifts, 
           suffix(".mol.feather"), 
           ".dataset.pickle")
def create_clean_dataset(infiles, outfile):
    shifts_infile, spect_infile, meta_infile = infiles

    mol_df = pd.read_feather(shifts_infile).set_index('molecule_id')
    spect_df = pd.read_feather(spect_infile)


    results = []
    for (molecule_id, spectrum_id), g in spect_df.groupby(['molecule_id', 'spectrum_id']):
        mol_row = mol_df.loc[molecule_id]
        spect_dict = [{row['atom_idx'] : row['value'] for _, row in g.iterrows()}]
        mol = Chem.Mol(mol_row.mol)
        results.append({'molecule_id' : molecule_id, 
                        'rdmol' : mol, 
                        'spect_dict': spect_dict, 
                        'smiles' : mol_row.simple_smiles, 
                        'morgan4_crc32' : util.morgan4_crc32(mol),
                        'spectrum_id' : spectrum_id})
    results_df = pd.DataFrame(results)
    print(results_df.morgan4_crc32.value_counts())

    print("dataset has", len(results_df), "rows")
    pickle.dump(results_df, open(outfile,'wb'))

if __name__ == "__main__":
    pipeline_run([preprocess_data_shifts, create_clean_dataset])
