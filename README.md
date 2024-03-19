
Nmrfilter v1.5 is an updated version of [Nmrfilter](https://github.com/stefhk3/nmrfilter), originally developed by S. Kuhn, S. Colreavy-Donnelly, J. Santana de Souza and R. M. Borges to demonstrate their [improved software pipeline for NMR mixture analysis](https://pubs.rsc.org/en/content/articlelanding/2019/fd/c8fd00227d).

Nmrfilter v1.5 can be used to find the most likely candidate substance from a set of candidates provided by there user, thereby automatically identifying the substance from its NMR spectrum data. Nmrfilter v1.5 outputs a list of the provided candidate substances and their individual match ratings in comparison to the original substance with plots visualizing the similarities between the compounds.

Installation
============

Requirements are Java and Python. For Java, version 1.8 or higher is needed. A JRE (Java Runtime Environment) is enough, a JDK is not required.
 
Python must be version >= 3.10.2. (3.10.2 an 3.11.2 have been tested and used).

Nothing further is needed, as the program creates a Python virtual environment installid all the needed packages.

Anaconda
--------

There are 2 Anaconda environments available for use:

`nmrfilter` for running the program using the CPU

Install by running
`conda env create -f environment-cpu.yml`

`nmrfiltergpu`  for running the program using the GPU
Install by running
`conda env create -f environment-gpu.yml`

Activate the environments by running `conda activate <environment name>`

When using the Anaconda, it is recommended to use the run-script `nmrfilternovenv.sh` to remove redundant operations.

Use of Respredict
=================


[Respredict](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0374-3) is another way of predicting NMR spectral properties using machine learning. Possibly yielding better results, respredict is available for use in Nmrfilter v1.5. It can be enabled in the properties file, see **Glossary**"**.

Use of Respredict requires extra packages, rendering the use of bundled Anaconda environments necessary. See **Anaconda** for further information.

Running
=======

Nmrfilter v1.5 works on projects, where each project is a folder. Example projects can be found [here](https://github.com/stefhk3/nmrfilterprojects) The folder must contain the following files:
- A list of candidate structures in the form of SMILES, one structure per line. File name can be configured by the `msmsinput` property
- Measured spectrum data in a .csv file. The file must be a list of shifts, coordinates seperated by a tab. ^13^C shift in the first dimension, ^1^H shift in the second. Each row corresponds to one shift. HMBC and HSQC shifts should be included. File name can be configured by the `spectruminput` property. 


Before running, in the `nmrproc.properties` file, the `datadir` property must be set to the absolute path of the folder containing project folders. Additionally, in the `nmrproc.properties` select the right solvent used in the mixture. See **Glossary** for available options.

With these files in place, the program can be run by using `./nmrfilter.sh <project name>` on Linux or `./nmrfilter.bat <project name>` on Windows. Replace `<project name>` with the name of the project (folder name) you want the predictions for. When running, the program creates a Python virtual environment installing everything needed for its use. If you prefer to use the external environment, see `requirements.txt` for all the packages needed to install and use the `nmrfilternovenv` script instead.


Following optional features are available:
- HSQCTOCSY shifts can be included. Include the shifts in the `spectruminput` file and set the `usehsqctocsy` property to `true`.
- Debug output can be produced by setting the property `debug` to true. As a prequisite, a file `testallnames.txt` needs to be included in the project folder. The file should contain the names of the compounds in the same order as in the `msmsinput` file.
- Parameters for tolerances and resolutions for the clustering and network algorithm can be set. See **Glossary** for details.
- To use respredict, set the property `usedeeplearning` to `true`. See **Use of Respredict** for further information. 

See **Glossary** for information about all the configurable properties.

Glossary
=======

The following table contains all the properties in the `nmrproc.properties` file with explanations, options and default values shown.

| Property | Description | Default | 
| ----------- | ----------- | ---------- |
| datadir | Path to the absolute directory containing project folders to be used for input.  | /home/karl/nmrfilterprojects |
| msmsinput | Name of the file containing the list of can-didate substances in a project folder.  | testall.smi |
| predictionoutput | Name of the file containing simulated spectra shifts of the candidate substances .  | resultpre-diction.csv |
| result | Name of solvent if used. Choices available are `Methanol-D4 (CD3OD)`, `Chloro-form-D1 (CDC13)` and `Dimethylsulph-oxide-D6 (DMSO-D6, C2D6SO)`. Other-wise use `Unreported`.  | `Methanol-D4 (CD3OD)` |
| tolerancec | Tolerance for the 13C axis.  | 0.2 |
| toleranceh | Tolerance for the 1H axis.  | 0.02 |
| spectuminput | Name of the file containing measured spectrum data.  | realspec-trum.csv |
| clusteringoutput | Name of a file created containing initial found cross peaks.  | cluster.txt |
| rberresolution | Resolution parameter for the RBER algo-rithm provided by the Louvain library, which changes the size of the clusters. Larger the value, smaller the clusters.  | 0.2 |
| usehmbc | Boolean. Define the use of HMBC or not.  | true |
| dotwobonds | TBA TBA TBA  | false |
| usedeeplearning | Boolean. If set “true”, uses respredict prediction instead of a HOSE code based one.  | false |
| debug | TBA TBA TBA  | false |


