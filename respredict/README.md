## Setting up the anaconda environment

We only support the [Anaconda python distribution](https://www.anaconda.com/distribution/), given the complexities of the machine learning frameworks that we depend on. We provide two environments, one for GPU users and one for CPU users. 

To create with GPU support
```
conda env create --name respredict -f environment.yaml
```

To create the CPU-only environment
```
conda env create --name respredict -f envrionment-cpu.yaml
```

Then to activate this environment run
```
conda activate respredict
```


## Using standalone mode

If you have molecules whose shifts you would like to predict you can
use the standalone runner. This runner can accept either an sdf file
or a pickled list of rdkit molecules. 

You can create example files of each of these types by running `standalone_example_files.py` which will create both an sdf file `example.sdf` and an example RDKit file `example.rdkit`. 

To test with the RDKit file:

```
python predict_standalone.py --filename example.rdkit --format rdkit 
```

which should generate something like the following to stdout (numbers may vary slightly due to both floating point issues and possibly updated model files):

```
{
    "predictions": [
        {
            "smiles": "[H]OC(=O)[C@@]([H])(c1c([H])c([H])c(C([H])([H])C([H])(C([H])([H])[H])C([H])([H])[H])c([H])c1[H])C([H])([H])[H]",
            "runtime": 0.3396494388580322,
            "shifts": [
                {
                    "atom_idx": 0,
                    "pred_mu": 22.200803756713867,
                    "pred_std": 1.0572324991226196
                },
                {
                    "atom_idx": 1,
                    "pred_mu": 29.968860626220703,
                    "pred_std": 0.9373214244842529
[...clipped...]
```
                
                



## Forward model

# To train:
```
python respredict_yaml_runner.py  expconfig/bootstrap_13C_big.yaml  test
```

# To generate evaluation data
```
python respred_pred_pipeline.py
```

## Other files / pipelines

`process_nmrshiftdb_dataset.py` : Process the nmrshiftdb data into our dataset files
