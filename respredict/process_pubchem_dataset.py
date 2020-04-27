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
import zlib

import sqlalchemy
from sqlalchemy import sql, func
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
    'pubchem_64_64_HCONF_10' : {'chunk_start' : 0 , 
                                 'chunk_end' : 10, 
    }, 

    'pubchem_128_128_HCONFSPCl_2' : {'chunk_start' : 0, 
                                      'chunk_end' : 2, 
                                      'max_atom_n' : 128, 
                                      'max_hevy_atom_n' : 128, 
                                      'elements' : HCONFSPCl}, 

    'pubchem_128_128_HCONFSPCl_5' : {'chunk_start' : 0, 
                                      'chunk_end' : 5, 
                                      'max_atom_n' : 128, 
                                      'max_hevy_atom_n' : 128, 
                                      'elements' : HCONFSPCl}, 

    'pubchem_128_128_HCONFSPCl_10' : {'chunk_start' : 0, 
                                      'chunk_end' : 10, 
                                      'max_atom_n' : 128, 
                                      'max_hevy_atom_n' : 128, 
                                      'elements' : HCONFSPCl}, 

    'pubchem_128_128_HCONFSPCl_50' : {'chunk_start' : 0, 
                                      'chunk_end' : 50, 
                                      'max_atom_n' : 128, 
                                      'max_hevy_atom_n' : 128, 
                                      'elements' : HCONFSPCl}, 

}

PUBCHEM_BASE_DIR =  "/data/data/pubchem"
PUBCHEM_DB_DIR = f"{PUBCHEM_BASE_DIR}/db/"


OUTPUT_DIR = "processed_data" 
td = lambda x : os.path.join(OUTPUT_DIR, x)

def params_shifts():
    for exp_name, ec in DATASETS.items():
        config = DEFAULT_CONFIG.copy()
        config.update(ec)

        yield (None, 
               (td("{}.shifts.mol.feather".format(exp_name)), 
                td("{}.meta.pickle".format(exp_name))),
               config)

@mkdir(OUTPUT_DIR)
@files(params_shifts)
def preprocess_data(infile, outfile, config):            
    
    mol_outfile, meta_outfile = outfile
    ### construct the query
    output_mols = []
    skip_reason_df = []
    for chunk in tqdm(range(config['chunk_start'], config['chunk_end'] + 1), desc='chunks'):
        field_start = chunk * 25000 + 1
        field_end = chunk * 25000 + 25000

        db_file = os.path.join(PUBCHEM_DB_DIR, f'Compound_{field_start:09d}_{field_end:09d}.sdf.db')
        if not os.path.exists(db_file):
            print("skipping", db_file)
        DB_URL = f"sqlite:///{db_file}"
        engine = sqlalchemy.create_engine(DB_URL)
        conn = engine.connect()
        meta = sqlalchemy.MetaData()
        meta.reflect(engine)
        molecules  = sqlalchemy.Table('molecules', meta, autoload=True, autoload_with=engine)

        stmt = sql.select([molecules]).where(sql.and_(molecules.c.valid_sanitize, 
                                                      molecules.c.frags == 1))


        mol_rows = []
        for mr in conn.execute(stmt):
            mol_rows.append({'id' : mr['id'], 
                             'mol' : zlib.decompress(mr['bmol']),
                             'morgan4_crc32' : mr['morgan4_crc32']})

        output_mol_df, skip_df = util.filter_mols(mol_rows, config, 
                                                         other_attributes = ['mol', 'morgan4_crc32'])
        output_mols.append(output_mol_df)
        skip_reason_df.append(skip_df)

    output_mol_df = pd.concat(output_mols).reset_index()
    output_mol_df.to_feather(mol_outfile)


    skip_reason_df = pd.concat(skip_reason_df)
    print(skip_reason_df.reason.value_counts(dropna=False))

    pickle.dump({'skip_reason_df' : skip_reason_df, 
                 'config' : config},
                open(meta_outfile, 'wb'))


@transform(preprocess_data, 
           suffix(".mol.feather"), 
           ".dataset.pickle")
def create_clean_dataset(infiles, outfile):
    shifts_infile, meta_infile = infiles

    mol_df = pd.read_feather(shifts_infile).set_index('molecule_id')

    mol_df['spect_dict'] = [list() for _ in range(len(mol_df))]

    mol_df = mol_df.rename(columns = {'simple_smiles': 'smiles'})
    mol_df['rdmol'] = mol_df.mol.apply(Chem.Mol)

    del mol_df['mol']
    mol_df['spectrum_id'] = 0 

    print(mol_df.morgan4_crc32.value_counts())

    print("dataset has", len(mol_df), "rows")
    pickle.dump(mol_df, open(outfile,'wb'))

if __name__ == "__main__":
    pipeline_run([preprocess_data,
                  create_clean_dataset
    ])
