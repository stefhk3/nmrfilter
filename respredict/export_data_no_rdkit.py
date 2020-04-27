import numpy as np
import pickle
import pandas as pd
from ruffus import * 
from tqdm import  tqdm
from rdkit import Chem
import pickle
import os

from glob import glob

import time
import util
import nets

from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdMolDescriptors as rdMD

import torch
from torch import nn
from tensorboardX import SummaryWriter

from tqdm import  tqdm
from netdataio import * 
import netdataio
import graph_conv_many_nuc_util

CV_SETS = [range(5)]


SPECT_SET = '13C'
MAT_PROPS='aromatic'
DATASET_NAME = 'nmrshiftdb_hconfspcl_nmrshiftdb'
tgt_max_n = 64
CV_I = 0

DATASET_DIR = "graph_conv_many_nuc_pipeline.datasets"
td = lambda x : os.path.join(DATASET_DIR, x)



@files(td('graph_conv_many_nuc_pipeline.data.{}.{}.{}.{:d}.{:d}.mol_dict.pickle'.format(SPECT_SET, DATASET_NAME, 
                                                                  MAT_PROPS, tgt_max_n, CV_I)), "test.out")

def create_dataset_for_later(infile, outfile):


    dataset_hparams = graph_conv_many_nuc_util.DEFAULT_DATA_HPARAMS

    ds_train, ds_test = graph_conv_many_nuc_util.make_datasets({'filename' : infile}, 
                                                               dataset_hparams)

    for name, dl in [('ds_train', ds_train), 
                     ('ds_test', ds_test)]:
        
    
        rows = []
        for r in tqdm(dl):
            rows.append(r)
        for i, f in enumerate(['adj', 'vect_feat', 'mat_feat', 
                               'vals', 'mask',]):
            s = np.stack([r[i] for r in rows])
            
            np.save("{}.{}.{}.npy".format(name, i, f), s)
    
    
if __name__ == "__main__":
    pipeline_run([create_dataset_for_later])
