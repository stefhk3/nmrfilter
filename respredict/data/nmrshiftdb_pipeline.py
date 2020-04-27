import numpy as np
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from lxml import etree as ET
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors as rdMD
import pickle
import os
from ruffus import * 
import sqlalchemy
import sys; sys.path.append("../")
import dbconfig
from sqlalchemy import sql
import datautil


"""

Generic code for importing from nmrshiftdb

Note that it's a little round-about in that it creates pickle files
and then inserts those into the database, as it was created
first for local pickle access

"""



Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

import datautil

DATA_DIR = "nmrshiftdb.data"

td  = lambda x : os.path.join(DATA_DIR, x)

NMRSHIFTDB_XML_FILENAME = "../../data/nmrshiftdb/nmrshiftdb2.xml"
NMRSHIFTDB_3D_XML_FILENAME = "../../data/nmrshiftdb/nmrshiftdb2_3d.xml"

def pp(s):
    
    print(ET.tostring(s, pretty_print=True).decode('utf-8'))

@mkdir(DATA_DIR)
@files(NMRSHIFTDB_3D_XML_FILENAME, 
       td("nmrshiftdb.molecules.pickle"))
def extract_molecules(xml_3d_filename, outfile):
    """
    Extract molecules and then stick into database
    """

    tree3d = ET.parse(xml_3d_filename)
    root = tree3d.getroot()

    molecules = []
    for molecule in root.findall('{http://www.xml-cml.org/schema}molecule'):
        molecules.append(molecule)

    MAX_DEBUG_ITER = 100000000

    molecules_df = []
    for m, _ in tqdm(zip(molecules, range(MAX_DEBUG_ITER)), total=len(molecules)):
        mol_id = m.attrib['id']

        mol = Chem.RWMol()
        mol.SetProp("id", mol_id)
        name = ""
        if 'title' in m.attrib:
            name = m.attrib['title']
        mol.SetProp("name", name)
        atomArray = m.find("{http://www.xml-cml.org/schema}atomArray")
        bondArray = m.find("{http://www.xml-cml.org/schema}bondArray")
        atom_pos_map = {}

        atoms_3dloc = []
        for ai, a in enumerate(atomArray):
            #print(a.attrib)
            atom = Chem.Atom(a.attrib['elementType'])
            x3 = float(a.attrib['x3'])
            y3 = float(a.attrib['y3'])
            z3 = float(a.attrib['z3'])
            

            #atom.SetIsotope(int(a.attrib['isotopeNumber']))
            atom.SetFormalCharge(int(a.attrib['formalCharge']))
            #atom.SetNumExplicitHs(int(a.attrib['hydrogenCount']))
            atom.SetProp('id', a.attrib['id'])
            idx = mol.AddAtom(atom)
            atom_pos_map[a.attrib['id']] = idx
            assert idx == ai

            atoms_3dloc.append((x3, y3, z3))

        for b in bondArray:
            atom_refs = b.attrib['atomRefs2']
            bond_order = b.attrib['order']
            a1, a2 = atom_refs.split(" ")
            if bond_order == 'S':
                bond = Chem.rdchem.BondType.SINGLE
            elif bond_order == 'D':
                bond = Chem.rdchem.BondType.DOUBLE
            elif bond_order == 'T':
                bond = Chem.rdchem.BondType.TRIPLE
            else:
                raise NotImplementedError()

            mol.AddBond(atom_pos_map[a1], atom_pos_map[a2], order=bond)


            C_count = np.sum([a.GetSymbol() == 'C' for a in mol.GetAtoms()])
            H_count = np.sum([a.GetSymbol() == 'H' for a in mol.GetAtoms()])


        try:
            Chem.SanitizeMol(mol)
            formula = rdMD.CalcMolFormula(mol)    

            error_msg = ""
            valid = True
        except ValueError as e:
            print("error sanitizing", name, e)
            error_msg = str(e)
            valid=False

        mol = mol.GetMol()
        c = datautil.array_to_conf(np.array(atoms_3dloc))
        mol.AddConformer(c)

        molecules_df.append({'mol_id' : mol_id, 
                             'name' : name, 
                             'C_count' : C_count, 
                             'H_count' : H_count, 
                             'formula' : formula, 
                             'error_msg' : error_msg, 
                             'mol' : mol, 
                             'valid' : valid})
    molecules_df = pd.DataFrame(molecules_df).set_index('mol_id')


    out = []
    for row_i, row in tqdm(molecules_df.iterrows(), total=len(molecules_df)):
        nmrshift_mol = row.mol

        id_to_pos = {nmrshift_mol.GetAtomWithIdx(i).GetProp('id'): i for i in range(nmrshift_mol.GetNumAtoms())}
        for id_str, pos in id_to_pos.items():
            out.append({'atom' : id_str, 
                       'atom_idx' : pos, 
                       'molecule' : row_i})
    mol_atomid_to_idx = pd.DataFrame(out).set_index(['molecule', 'atom'])


    pickle.dump({'molecules_df' : molecules_df, 
                 'mol_atomid_to_idx' : mol_atomid_to_idx }, 
                open(outfile, 'wb'))

DB_DATA_SOURCE = "nmrshiftdb"


@transform(extract_molecules, suffix(".pickle"), 
           ".uploaded.pickle")
def molecules_to_db(infile, outfile):
    d = pickle.load(open(infile, 'rb'))
    molecules_df = d['molecules_df']

    engine = sqlalchemy.create_engine(dbconfig.AWS_RDS_DB_STR)
    conn = engine.connect()

    meta = sqlalchemy.MetaData()
    
    t1 = time.time()
    molecules = sqlalchemy.Table('molecule', meta, autoload=True, 
                                autoload_with=engine)
    atoms = sqlalchemy.Table('atom', meta, autoload=True, 
                                autoload_with=engine)
    conformations = sqlalchemy.Table('conformation', meta, autoload=True, 
                                autoload_with=engine)

    stmt = atoms.delete().\
           where(atoms.c.molecule_id == molecules.c.id).\
           where(molecules.c.source == DB_DATA_SOURCE)
    conn.execute(stmt)

    stmt = conformations.delete().\
           where(conformations.c.source == DB_DATA_SOURCE)
    conn.execute(stmt)

    stmt = molecules.delete().where(molecules.c.source == DB_DATA_SOURCE)
    conn.execute(stmt)
    
    for molecules_sub_df in tqdm(datautil.split_df(molecules_df, 4000)):

        mol_values = []
        conf_values = []
        for row_i, row in tqdm(molecules_sub_df.iterrows(), 
                               total=len(molecules_sub_df)):
            if not row.valid:
                continue
            mol = row.mol
            # create the molecule
            data_dict =   {'name' : row['name'], 
                           'smiles' : Chem.MolToSmiles(mol), 
                           'mol' : Chem.MolToMolBlock(mol), 
                           'source' : DB_DATA_SOURCE, 
                           'source_id' : row_i, }
            mol_values.append(data_dict)
            conf_values.append({'mol' : Chem.MolToMolBlock(mol),
                                'source' : DB_DATA_SOURCE, 
                                'meta' : {'note' : 'default from xml'}})

        res = conn.execute(molecules.insert(mol_values).returning(molecules.c.id))
        
        molecule_ids = [r['id'] for r in res]
        for conf_value, molecule_id in zip(tqdm(conf_values), molecule_ids):
            conf_value['molecule_id'] = molecule_id

        res = conn.execute(conformations.insert(conf_values))

        atom_values = []
        for mol_values, molecule_id in zip(tqdm(mol_values), molecule_ids):

            source_id = mol_values['source_id']

            mol = molecules_sub_df.loc[source_id].mol
            # create the atoms
            for atom_idx in range(mol.GetNumAtoms()):
                a = mol.GetAtomWithIdx(atom_idx)
                atom_values.append({'molecule_id' : molecule_id, 
                                  'idx' : atom_idx, 
                                  'atomicnum' : a.GetAtomicNum()})


        stmt = atoms.insert(atom_values)
        conn.execute(stmt)

    t2 = time.time()

    pickle.dump({'time' : t2-t1, 
                 'rowcount' : len(molecules_df)},
                open(outfile, 'wb'))
    
    



@files(NMRSHIFTDB_XML_FILENAME, 
       [td("nmrshiftdb.spectra_meta.raw.feather"), 
        td("nmrshiftdb.peaks.raw.feather")]
)
def extract_spectra_raw(xml_filename, outfiles):
    """
    Raw import for clean-up later
    """
    meta_outfile, peaks_outfile = outfiles


    tree = ET.parse(xml_filename)

    root = tree.getroot()
    spectra = []
    for spectrum in root.findall('{http://www.xml-cml.org/schema}spectrum'):
        spectra.append(spectrum)

    spectra_meta = []
    spectra_peaks = []


    for spectrum in tqdm(spectra):
        molecule_ref = spectrum.attrib['moleculeRef']
        spectrum_id = spectrum.attrib['id']
        conditionList = spectrum.find("{http://www.xml-cml.org/schema}conditionList")
        substanceList = spectrum.find("{http://www.xml-cml.org/schema}substanceList")
        metadataList = spectrum.find("{http://www.xml-cml.org/schema}metadataList")
        metadata_dict = {}
        if metadataList is not None:
            for m in metadataList:
                name = m.attrib['name']
                content = m.attrib['content']
                if content.lower() in ['unreported', 'unknown']:
                    continue
                metadata_dict[name] = content
        if conditionList is not None:
            for m in conditionList:
                name = m.attrib['dictRef']
                content = m.text.strip()
                if content.lower()in  ['unreported', 'unknown']:
                    continue
                metadata_dict[name] = content

        if substanceList is not None:
            for m in substanceList:
                name = m.attrib['dictRef']
                content = None
                if name == 'cml:solvent':
                    content = m.attrib['title']

                if content is not None:
                    if content.lower() in ['unreported', 'unknown']:
                        continue

                    metadata_dict[name] = content.strip()

        spectrum_meta = {'id' : spectrum_id, 
                         'molecule_ref' : molecule_ref}

        spectrum_meta.update(metadata_dict)

        spectra_meta.append(spectrum_meta)

        peakList = spectrum.find("{http://www.xml-cml.org/schema}peakList")
        for p in peakList:
            peak = {}
            peak['value'] = float(p.attrib['xValue'])
            peak['units'] = p.attrib['xUnits']
            peak['shape'] = p.attrib['peakShape']
            peak['peakMultiplicity'] = p.attrib.get('peakMultiplicity', None)
            peak['id'] = p.attrib['id']
            peak['atomRefs'] = p.attrib.get('atomRefs', "")
            peak['molecule'] = molecule_ref
            peak['spectrum_id'] = spectrum_id

            spectra_peaks.append(peak)

    meta_df = pd.DataFrame(spectra_meta)
    meta_df.to_feather(meta_outfile)
    
    peaks_df = pd.DataFrame(spectra_peaks)
    peaks_df.to_feather(peaks_outfile)


@files(extract_spectra_raw, 
       td("nmrshiftdb.spectra_meta.feather"))
def cleanup_spectra_meta(infile, outfile):

    spectra_meta_raw_df = pd.read_feather(infile[0])
    

    spectra_meta_df = spectra_meta_raw_df[['id', 'molecule_ref', 'nmr:Program',
                                           'nmr:OBSERVENUCLEUS', 'nmr:assignmentMethod', 
                                           'cml:temp', 'cml:field', 
                                           'cml:solvent']].rename(columns={'nmr:OBSERVENUCLEUS': 'nucleus', 
                                                                           'nmr:assignmentMethod' : 'howassign', 
                                                                           'cml:temp' : 'temp', 
                                                                           'cml:field' : 'field', 
                                                                           'cml:solvent': 'solvent'
                                           })
    # clean up field
    def clean_field(x):
        if x is None:
            return x
        if isinstance(x, str):
            if "Not" in x:
                return None

            return float(x.split(" ")[0])
        else:
            return x
    spectra_meta_df['field'] = spectra_meta_df.field.apply(clean_field)
    spectra_meta_df['sim'] = ~pd.isnull(spectra_meta_df['nmr:Program'])
    del spectra_meta_df['nmr:Program']

    spectra_meta_df.to_feather(outfile)


@follows(molecules_to_db)
@transform(cleanup_spectra_meta, suffix(".feather"), 
           ".uploaded.pickle")
def spectra_meta_to_db(infile, outfile):
    spectra_meta_df = pd.read_feather(infile)

    engine = sqlalchemy.create_engine(dbconfig.AWS_RDS_DB_STR)
    conn = engine.connect()

    meta = sqlalchemy.MetaData()
    
    t1 = time.time()
    molecules = sqlalchemy.Table('molecule', meta, autoload=True, 
                                 autoload_with=engine)
    spectra_meta = sqlalchemy.Table('spectrum_meta', meta, autoload=True, 
                                    autoload_with=engine)

    stmt = spectra_meta.delete()
    conn.execute(stmt)

    # build lookup table
    stmt = sql.select([molecules.c.id, molecules.c.source_id]).where(molecules.c.source == DB_DATA_SOURCE)
    source_id_df = pd.read_sql(stmt, engine).set_index("source_id")

    for spectra_meta_sub_df in tqdm(datautil.split_df(spectra_meta_df, 1000)):

        spec_values = []

        for row_i, row in spectra_meta_sub_df.iterrows():
            molecule_ref = row['molecule_ref']
            if molecule_ref not in source_id_df.index:
                continue
            molecule_id =  int(source_id_df.loc[molecule_ref]['id'])
            # create the spectrum
            try:
                data_dict =   {'molecule_id' : molecule_id, 
                               'nucleus' : row['nucleus'], 
                               'temp' : row['temp'], 
                               'field' : row['field'], 
                               'solvent' : row['solvent'],
                               'source' : DB_DATA_SOURCE, 
                               'source_id' : row['id'], }
                assert(data_dict['molecule_id'] is not None)
                if (data_dict['nucleus'] is  None) or (data_dict['nucleus'].strip() == ""):
                    continue

                spec_values.append(data_dict)
            except KeyError:
                pass
        if len(spec_values) > 0:
            stmt = spectra_meta.insert(spec_values)
            conn.execute(stmt)

    t2 = time.time()

    pickle.dump({'time' : t2-t1, 
                 'rowcount' : len(spectra_meta_df)},
                open(outfile, 'wb'))
    
    


@files(extract_spectra_raw, 
       td("nmrshiftdb.peaks.feather"))
def cleanup_spectra_peaks(infile, outfile):
    peaks_raw_df = pd.read_feather(infile[1])
    peaks_raw_df['atom_refs_list'] = peaks_raw_df.atomRefs.str.split(" ")

    per_ref_df = datautil.explode_df(peaks_raw_df, lst_cols=['atom_refs_list']) .rename(columns={'atom_refs_list': "atom"})

    per_ref_df.to_feather(outfile)

@follows(spectra_meta_to_db)
@transform(cleanup_spectra_peaks, 
           suffix(".feather"), 
           ".uploaded.feather")
def spectra_peaks_to_db(infile, outfile):


    d = pickle.load(open(td("nmrshiftdb.molecules.pickle"), 'rb'))
    mol_atomid_to_idx_df = d['mol_atomid_to_idx']
    molecules_df = d['molecules_df']

    peaks_df = pd.read_feather(infile)

    a = mol_atomid_to_idx_df.reset_index().set_index(['molecule', 'atom'])

    peaks_df = peaks_df.join(a, on=['molecule', 'atom'])

    engine = sqlalchemy.create_engine(dbconfig.AWS_RDS_DB_STR)
    conn = engine.connect()

    meta = sqlalchemy.MetaData()

    t1 = time.time()
    spectra_meta = sqlalchemy.Table('spectrum_meta', meta, autoload=True, 
                                    autoload_with=engine)

    molecules = sqlalchemy.Table('molecule', meta, autoload=True, 
                                    autoload_with=engine)
    peaks = sqlalchemy.Table('peak', meta, autoload=True, 
                             autoload_with=engine)

    atoms = sqlalchemy.Table('atom', meta, autoload=True, 
                             autoload_with=engine)
    stmt = peaks.delete().\
           where(peaks.c.spectrum_id == spectra_meta.c.id).\
           where(spectra_meta.c.source == DB_DATA_SOURCE)
    
    conn.execute(stmt)

    
    stmt = sql.select([spectra_meta.c.id.label("spectrum_id"), spectra_meta.c.source_id,spectra_meta.c.molecule_id, 
                       molecules.c.source_id.label("mol_source_id")]).where(sql.and_(spectra_meta.c.source == DB_DATA_SOURCE, 
                                                              spectra_meta.c.molecule_id == molecules.c.id))
    spectra_id_df = pd.read_sql(stmt, engine).set_index("source_id")

    atoms_df = pd.read_sql(sql.select([atoms]), engine).set_index(['molecule_id', 'idx'])

    units_map = {"units:ppm" : "ppm"}

    for peaks_sub_df in tqdm(datautil.split_df(peaks_df, 80)):

        allpeaks = []
        for peak_i, peak in peaks_sub_df.iterrows():

            try:

                specdata = spectra_id_df.loc[peak.spectrum_id]
                atom_rec = atoms_df.loc[(specdata.molecule_id, peak.atom_idx)]
            except:
                continue
            atom_id = atom_rec.id
            data_dict = {'spectrum_id' : int(specdata.spectrum_id) , 
                         'atom_id' : int(atom_id), 
                         'units' : units_map[peak.units], 
                         'multiplicity' : peak.peakMultiplicity, 
                         'shape' : peak.shape, 
                         'value' :  peak.value}
            allpeaks.append(data_dict)
        if len(allpeaks) > 0:
            stmt = peaks.insert(allpeaks)
            conn.execute(stmt)

    pickle.dump({}, open(outfile, 'wb'))


if __name__ == "__main__":
    pipeline_run([extract_molecules,
                  molecules_to_db, 
                  cleanup_spectra_meta , 
                  cleanup_spectra_peaks, 
                  spectra_meta_to_db, 
                  spectra_peaks_to_db, 

    ])
                  #count_atoms, 
                  #split_spectra, compute_3d_etkdg
