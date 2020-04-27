import numpy as np
import pickle
import pandas as pd
import time
import atom_features_old
from rdkit import Chem


import atom_features

def angular_benchmark():

    molecules_df = pickle.load(open("dataset.named/molconf.nmrshiftdb_hconf_nmrshiftdb.pickle", 'rb'))['df']

    per_row_features = []
    per_row_atomic_nos = []
    pos = 0

    t1 = time.time()
    for row_i, row in molecules_df.iterrows():

        atomic_nos, coords = atom_features.get_nos_coords(row.rdmol, 0)
        output_atoms = None

        f = atom_features.angular_images(atomic_nos, coords, pairings=[(1, 6), (1, 7), (1, 8), (6, 6), (6, 7), (6, 8)])
        per_row_features.append(f)

        per_row_atomic_nos.append(atomic_nos)
        #display(row.rdmol)

        pos += 1
        if pos > 1000:
            break
    t2 = time.time()
    print("total time", t2-t1)
    m_s = pos / (t2-t1)
    print(m_s, "mol/sec")
    print("would take", len(molecules_df)/m_s/60.0, "min for all data")


def image_3view_benchmark_random():

    MOL_N = 2000
    MOL_SIZE = 64
    RENDER_PIX = 64
    coords_all = np.random.normal(0, 2, (MOL_N, MOL_SIZE, 3))
    atomic_nos_all = np.random.choice([1, 6, 7, 8, 9], (MOL_N, MOL_SIZE))

    t1 = time.time()
    for atomic_nos, coords in zip(atomic_nos_all, coords_all):


        for atom_i in range(len(atomic_nos)):
            vects = coords - coords[atom_i]

            f = atom_features.render_3view_points(atomic_nos, vects, [1, 6, 7, 8, 9], 
                                                  0.2, RENDER_PIX)

    t2 = time.time()
    print("total time", t2-t1)
    m_s = MOL_N / (t2-t1)
    print(m_s, "mol/sec")
    print(MOL_SIZE * m_s, "atoms/sec")
    TOTAL_MOLECULES = 60000
    print("would take", TOTAL_MOLECULES/m_s/60.0, "min for all data")

def image_3view_benchmark():

    molecules_df = pickle.load(open("dataset.named/molconf.nmrshiftdb_hconf_nmrshiftdb.pickle", 'rb'))['df']

    per_row_features = []
    per_row_atomic_nos = []
    pos = 0

    t1 = time.time()
    for row_i, row in molecules_df.iterrows():

        atomic_nos, coords = atom_features.get_nos_coords(row.rdmol, 0)
        for atom_i in range(len(atomic_nos)):
            vects = coords - coords[atom_i]


            f = atom_features.render_3view_points(atomic_nos, vects, [1, 6, 7, 8, 9], 
                                                  0.2, 64)
        # per_row_features.append(f)

        # per_row_atomic_nos.append(atomic_nos)
        #display(row.rdmol)

        pos += 1
        if pos > 1000:
            break
    t2 = time.time()
    print("total time", t2-t1)
    m_s = pos / (t2-t1)
    print(m_s, "mol/sec")
    print("would take", len(molecules_df)/m_s/60.0, "min for all data")

@profile
def radial_benchmark():
    mol = Chem.MolFromSmiles("c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N")

    mol = Chem.AddHs(mol)
    
    mol_3d = Chem.Mol(mol)
    Chem.AllChem.EmbedMolecule(mol_3d)

    import time
    t1 = time.time()
    ITERS = 1000
    for i in range(ITERS):

        c = mol_3d.GetConformer(0)
        atomic_nos = [a.GetAtomicNum() for a in mol_3d.GetAtoms()]
        coords = c.GetPositions()
        res = atom_features_old.custom_bp_radial(atomic_nos, coords)
    t2 = time.time()

    rate = ITERS / (t2-t1)
    print(f"{rate:3.1f} mols/sec")
    #pylab.imshow(res[tgt_atom])


if __name__ == "__main__":
    radial_benchmark()
