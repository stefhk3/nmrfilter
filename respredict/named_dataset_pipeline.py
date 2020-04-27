"""
Cache database queries to local pandas dataframe
with pickled RDKit structures with conformers

"""

import pandas as pd
from rdkit import Chem
from sqlalchemy import sql
import sqlalchemy
from ruffus import *
import os
import dbconfig
from tqdm import tqdm
import numpy as np
import pickle
import util

HCONF = ['H', 'C', 'O', 'N', 'F']

MOLCONF_DATASETS = {'nmrshiftdb_hconf2_nmrshiftdb' : {'molecule_source'  : 'nmrshiftdb', 
                                                       'elements' : HCONF,  
                                                       'conformer_source' : 'nmrshfitdb'}, 
                    'nmrshiftdb_hconfspcl_nmrshiftdb' : {'molecule_source'  : 'nmrshiftdb', 
                                  'elements' : HCONF + ['S', 'P', 'Cl'],  
                                                         'conformer_source' : 'nmrshfitdb'}, 
}

WORKING_DIR = "dataset.named"
td = lambda x: os.path.join(WORKING_DIR, x)

def molconf_named_params():
    for ds_name, ds_conf in MOLCONF_DATASETS.items():
        infile = None
        outfile = td(f"molconf.{ds_name}.pickle")
        yield infile, outfile, ds_name, ds_conf

@mkdir(WORKING_DIR)
@files(molconf_named_params)
def get_named_molconf_dataset(infile, outfile, ds_name, ds_conf):

    conf_source = ds_conf['conformer_source']
    mol_source = ds_conf['molecule_source']
    elements = ds_conf['elements']
    tgt_atomicnum = [Chem.GetPeriodicTable().GetAtomicNumber(e) for e in elements]


    engine = sqlalchemy.create_engine(dbconfig.AWS_RDS_DB_STR)
    conn = engine.connect()
    meta = sqlalchemy.MetaData()
    molecules = sqlalchemy.Table('molecule', meta, autoload=True, 
                                autoload_with=engine)
    atoms = sqlalchemy.Table('atom', meta, autoload=True, 
                                autoload_with=engine)
    conformations = sqlalchemy.Table('conformation', meta, autoload=True, 
                                autoload_with=engine)

    stmt = sql.select([molecules.c.id, molecules.c.mol, conformations.c.mol.label("conf_mol"), 
                        conformations.c.id.label("conf_id")])\
                .where(atoms.c.molecule_id == molecules.c.id)\
                .where(molecules.c.source == mol_source)\
                .where(conformations.c.molecule_id == molecules.c.id)\
                .where(conformations.c.source == mol_source)\
                .distinct(conformations.c.id)
    df = pd.read_sql(stmt, engine)

    # find complement of tgt molecules
    stmt = sql.select([atoms.c.molecule_id])\
                .where(atoms.c.atomicnum.in_(set(range(128)) - set(tgt_atomicnum )))\
                .distinct(atoms.c.molecule_id)
    not_tgt_elt_df = pd.read_sql(stmt, engine)

    df = df[~df.id.isin(not_tgt_elt_df.molecule_id)].copy()

    df = df.sort_values(['id', 'conf_id'])

    res = []
    # there may be more than one conformer per molecule
    for gi, g in tqdm(df.groupby('id'), total=len(np.unique(df.id))):
        m = Chem.MolFromMolBlock(g.iloc[0].mol, sanitize=False)
        m.RemoveAllConformers()
        for m_i, m_r in g.iterrows():
            m_c = Chem.MolFromMolBlock(m_r.conf_mol, sanitize=False)
            m_c.SetIntProp("id", m_r['conf_id'])
            if util.conf_not_null(m_c, 0):
                m.AddConformer(m_c.GetConformers()[0])
        if m.GetNumConformers() > 0:
            
            res.append({'molecule_id': gi, 
                        'rdmol' : m})
        else:
            print(f"Warning, {gi} had no valid conformers!")

    df = pd.DataFrame(res).set_index('molecule_id')
    pickle.dump({'df' : df, 
                 'ds_conf' : ds_conf}, 
                open(outfile, 'wb'))

@transform(get_named_molconf_dataset, suffix(".pickle"), 
           ".sdf")
def molconf_to_sdf(infile, outfile):

    data = pickle.load(open(infile, 'rb'))
    mol_df = data['df']

    w = Chem.rdmolfiles.SDWriter(outfile)

    for mol_id, row in mol_df.iterrows():
        m = Chem.Mol(row.rdmol)
        m.SetIntProp("id", mol_id)
        w.write(m)
    w.close()


SPLIT_MAX_N = 64
@transform(get_named_molconf_dataset, suffix(".pickle"), 
           f".{SPLIT_MAX_N}.split")
def molconf_to_sdf_5foldsplit(mol_filename, outfile):

    CV_SETS = [np.arange(4) + i*4 for i in range(5)]
    
    molecules_df = pickle.load(open(mol_filename, 'rb'))['df']
    molecules_df['atom_n'] = molecules_df.rdmol.apply(lambda x: x.GetNumAtoms())
    molecules_df = molecules_df[molecules_df.atom_n <= SPLIT_MAX_N]
    molecules_df = molecules_df.reset_index()
    print(molecules_df.dtypes)

    mol_subset_filename = 'predict.atomic/molconf.nmrshiftdb_hconfspcl_nmrshiftdb.subsets.pickle'

    mol_subsets = pickle.load(open(mol_subset_filename, 'rb'))['splits_df']

    for cv_i, cv_mol_subset in enumerate(CV_SETS):
        train_test_split = mol_subsets.subset20_i.isin(cv_mol_subset)
        train_mols = mol_subsets[~train_test_split].index.values
        test_mols = mol_subsets[train_test_split].index.values

        train_df = molecules_df[molecules_df.molecule_id.isin(train_mols)]
        test_df = molecules_df[molecules_df.molecule_id.isin(test_mols)]

        for phase, df in [('train', train_df), 
                          ('test', test_df)]:

            outfile_name = outfile.replace(".split", 
                                           f".{cv_i}.{phase}.sdf")
            w = Chem.rdmolfiles.SDWriter(outfile_name)

            for _, row in df.iterrows():
                m = Chem.Mol(row.rdmol)
                m.SetIntProp("id", row.molecule_id)
                w.write(m)
            w.close()
    pickle.dump({}, open(outfile, 'wb'))

SPECTRA_DATASETS = {'nmrshiftdb_13C' : {'spectra_source'  : 'nmrshiftdb', 
                                        'nucleus' : '13C'},
                    'nmrshiftdb_1H' : {'spectra_source'  : 'nmrshiftdb', 
                                        'nucleus' : '1H'}, 
                    'nmrshiftdb_13C_cdcl3' : {'spectra_source'  : 'nmrshiftdb', 
                                              'nucleus' : '13C', 
                                              'solvent' : ['Chloroform-D1 (CDCl3)']},
                    'nmrshiftdb_1H_cdcl3' : {'spectra_source'  : 'nmrshiftdb', 
                                             'nucleus' : '1H', 
                                             'solvent' : ['Chloroform-D1 (CDCl3)']}, 

}


def spectra_named_params():
    for ds_name, ds_conf in SPECTRA_DATASETS.items():
        infile = None
        outfile = td(f"spectra.{ds_name}.feather")
        yield infile, outfile, ds_name, ds_conf

@mkdir(WORKING_DIR)
@files(spectra_named_params)
def get_named_spectra_dataset(infile, outfile, ds_name, ds_conf):

    spec_source = ds_conf['spectra_source']
    nucleus = ds_conf['nucleus']

    engine = sqlalchemy.create_engine(dbconfig.AWS_RDS_DB_STR)
    conn = engine.connect()
    meta = sqlalchemy.MetaData()
    spectra_meta = sqlalchemy.Table('spectrum_meta', meta, autoload=True, 
                                    autoload_with=engine)
    peaks = sqlalchemy.Table('peak', meta, autoload=True, 
                                autoload_with=engine)
    atoms = sqlalchemy.Table('atom', meta, autoload=True, 
                                autoload_with=engine)

    stmt = sql.select([peaks, spectra_meta.c.molecule_id, atoms.c.idx.label('atom_idx') ])\
                .where(spectra_meta.c.nucleus == nucleus)\
                .where(spectra_meta.c.source == spec_source)\
                .where(spectra_meta.c.id == peaks.c.spectrum_id)\
                .where(peaks.c.atom_id == atoms.c.id)
    if 'solvent' in ds_conf:
        solv_clauses = [spectra_meta.c.solvent == sol for sol in ds_conf['solvent']]
        stmt = stmt.where(sql.or_(*solv_clauses))

    peak_df = pd.read_sql(stmt, engine)        
    peak_df.to_feather(outfile)

@follows(get_named_spectra_dataset)
@files([td('spectra.nmrshiftdb_1H.feather'), 
        td('molconf.nmrshiftdb_hconfspcl_nmrshiftdb.pickle')], 
       [td("spectra.nmrshiftdb_hconfspcl_1H_bonded.feather"), 
        td("spectra.nmrshiftdb_hconfspcl_1H_Cbonded.feather")])
def get_bonded_spect(infiles, outfiles):
    spect_file, mol_file = infiles 
    spect_df = pd.read_feather(spect_file)
    mol_df = pickle.load(open(mol_file, 'rb'))['df']

    outfile_bonded, outfile_c_bonded = outfiles
    def get_neighbor(row):
        try:
            mol =  mol_df.loc[row.molecule_id].iloc[0]
        except:
            return None
        a = mol.GetAtomWithIdx(int(row.atom_idx))
        ns = a.GetNeighbors()
        n_0 = ns[0]
        return pd.Series({'bonded_idx': n_0.GetIdx(), 'bonded_atomicno' : n_0.GetAtomicNum()})

    a= spect_df
    spect_h_df = pd.concat([a, a.apply(get_neighbor, axis=1)], axis=1)
    
    spect_h_df.reset_index().to_feather(outfile_bonded)

    spect_h_c_df = spect_h_df[spect_h_df.bonded_atomicno == 6].copy()
    
    spect_h_c_df.reset_index().to_feather(outfile_c_bonded)

@files(get_bonded_spect,        
       [td("spectra.nmrshiftdb_hconfspcl_1H_as_bonded.feather"), 
         td("spectra.nmrshiftdb_hconfspcl_1HCbonded_as_bonded.feather")])
def spect_H_as_bonded(infiles, outfiles):
    """
    return a spectra database where each carbon has the average
    of its attached hydrogens
    """

    for infile, outfile in zip(infiles, outfiles):

        spect_h_c_df = pd.read_feather(infile)



        mean_for_idx = spect_h_c_df.groupby(['molecule_id', 'spectrum_id',  'bonded_idx']).agg({'value' : 'mean'})
        spect_h_as_c_df = spect_h_c_df.join(mean_for_idx.rename(columns={'value' : 'mean_H_value'}), 
                                          on=('molecule_id', 'spectrum_id', 'bonded_idx') )

        a= spect_h_as_c_df.drop_duplicates(subset=('molecule_id', 'spectrum_id', 'bonded_idx') )
        del a['value']
        del a['atom_idx']
        a = a.rename(columns={'mean_H_value' : 'value', 'bonded_idx' : 'atom_idx'})

        a.reset_index().to_feather(outfile)



@transform(get_named_spectra_dataset, suffix(".feather"), 
           ".csv")
def spectra_to_csv(infile, outfile):
    spect_df = pd.read_feather(infile)
    
    
    sub_df = spect_df[['molecule_id', 'atom_idx', 'value']].sort_values(['molecule_id', 'atom_idx'])
    sub_df.to_csv(outfile, index=False)


if __name__ == "__main__":
    pipeline_run([get_named_molconf_dataset, molconf_to_sdf, 
                  get_named_spectra_dataset, spectra_to_csv,
                  molconf_to_sdf_5foldsplit, 
                  get_bonded_spect,  spect_H_as_bonded])
