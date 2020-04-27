from rdkit import Chem
import numpy as np
from datetime import datetime
import click
import pickle
import time
import json 
import sys
import util
import netutil
import warnings
import torch

warnings.filterwarnings("ignore")

nuc_to_atomicno = {'13C' : 6, 
                   '1H' : 1}

def predict_single_mol(raw_mol, pred, MAX_N, tgt_nucs = None, 
                       add_h = True, sanitize=True, 
                       add_default_conf=True):

    t1 = time.time()
    if add_h:
        mol = Chem.AddHs(raw_mol)
    else:
        mol = raw_mol

    # sanity check
    if mol.GetNumAtoms() > MAX_N:
        raise ValueError("molecule has too many atoms")
    if sanitize:
        Chem.SanitizeMol(mol)

    if len( mol.GetConformers()) == 0 and add_default_conf:
        util.add_empty_conf(mol)

    value_list = []
    for t in tgt_nucs:
        value_list.append({i : 0.0 for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() == nuc_to_atomicno[t]})

    r = pred.pred([mol], [value_list], [{}])

    t2 = time.time()

    out_dict =  {'smiles' : Chem.MolToSmiles(raw_mol), 
                 'runtime' : t2-t1}

    if len(r) == 0:
        warnings.warn("molecule {} had no {} shifts".format(Chem.MolToSmiles(raw_mol), tgt_nucs))
        out_dict['shifts'] = []
    else:
        out_dict['shifts'] = r[['atom_idx', 'pred_mu', 'pred_std']].to_dict(orient='records')

    out_dict['success'] = True
    return out_dict

DEFAULT_FILES = {'13C' : {'meta' : 'models/default_13C.meta', 
                         'checkpoint' : 'models/default_13C.checkpoint' }, 
                 '1H' : {'meta' : 'models/default_1H.meta', 
                          'checkpoint' : 'models/default_1H.checkpoint'}}


@click.command()
@click.option('--filename', help='filename of file to read, or stdin if unspecified', default=None)
@click.option('--format', help='file format (sdf, rdkit, smiles)', default='sdf', 
              type=click.Choice(['rdkit', 'sdf', 'smiles'], case_sensitive=False))
@click.option('--nuc', help='Nucleus (1H or 13C)', default='13C', 
              type=click.Choice(['1H', '13C'], case_sensitive=True))
@click.option('--model_meta_filename')
@click.option('--model_checkpoint_filename')
@click.option('--output', default=None)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--sanitize/--no-sanitize', help="sanitize the input molecules", default=True)
@click.option('--addhs', help="Add Hs to the input molecules", default=False)
@click.option('--skip-molecule-errors/--no-skip-molecule-errors', help="skip any errors", default=True)
def predict(filename, format, nuc, model_meta_filename, 
            model_checkpoint_filename, cuda=False, 
            output=None, sanitize=True, addhs=True, 
            skip_molecule_errors=True):

    ts_start = time.time()

    if model_meta_filename is None:
        # defaults
        model_meta_filename = DEFAULT_FILES[nuc]['meta']
        model_checkpoint_filename = DEFAULT_FILES[nuc]['checkpoint']

    meta = pickle.load(open(model_meta_filename, 'rb'))
    
    MAX_N = meta['max_n']
    tgt_nucs = meta['tgt_nucs']
    if cuda and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available, running with CPU")
        cuda = False
    pred = netutil.PredModel(model_meta_filename, 
                             model_checkpoint_filename, 
                             cuda)


    if format == 'sdf':
        if filename is None:
            mol_supplier = Chem.ForwardSDMolSupplier(sys.stdin.buffer)
        else:
            mol_supplier = Chem.SDMolSupplier(filename)
    elif format == 'rdkit':
        if filename is None:
            bin_data = sys.stdin.buffer.read()
            mol_supplier = [Chem.Mol(m) for m in pickle.loads(bin_data)]
        else:
            mol_supplier = [Chem.Mol(m) for m in pickle.load(open(filename, 'rb'))]
    elif format == 'smiles':
        if filename is None:
            mol_supplier = Chem.SmilesMolSupplier(sys.stdin.buffer)
        else:
            mol_supplier = Chem.SmilesMolSupplier(filename)

        
    all_results = []
    for m_input in mol_supplier:
        try:
            result = predict_single_mol(m_input, pred, MAX_N, tgt_nucs, 
                                        add_h=addhs, sanitize=sanitize)
            all_results.append(result)
        except Exception as e:
            if not skip_molecule_errors:
                raise
            warnings.warn("molecule raise exception {}".format(e))

            all_results.append({'runtime' : 0.0, 
                                'success' : False, 
                                'exception' : str(e)})
                      

    ts_end = time.time()
    output_dict = {'predictions' : all_results, 
              'meta' : {'max_n' : MAX_N, 
                        'tgt_nucs' : tgt_nucs, 
                        'model_checkpoint_filename' : model_checkpoint_filename, 
                        'model_meta_filename' : model_meta_filename, 
                        'ts_start' : datetime.fromtimestamp(ts_start).isoformat(), 
                        'ts_end': datetime.fromtimestamp(ts_end).isoformat(), 
                        'runtime_sec' : ts_end - ts_start, 
                        'use_cuda' : cuda}
              }
    json_str = json.dumps(output_dict, sort_keys=False, indent=4)
    if output is None:
        print(json_str)
    else:
        with open(output, 'w') as fid:
            fid.write(json_str)



if __name__ == "__main__":
    predict()
