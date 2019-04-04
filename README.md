This is the code used in the paper https://pubs.rsc.org/en/content/articlelanding/2019/fd/c8fd00227d. The data in realspectrum.csv are the example for P. boldus. Running the script nmrfilter will produce the list of hits for P. boldus.

#Installation

Requirements are Java and Python. For Java, version 1.8 or higher is needed. A JRE (Java Runtime Environment) is enough, a JDK is not required. The default of any operating system should do.
 
Python version must be 3 (3.66 was tested). You need numpy, scipy, louvain, and python-igraph libraires. Using pip, you can install them by doing

pip3 install numpy

pip3 install scipy

pip3 install python-igraph

pip3 install louvain

Notice igraph is a different library. If install python-igraph gives an error about missing C libraries, try using a wheel, following https://stackoverflow.com/questions/34113151/how-to-install-igraph-for-python-on-windows

#Running

The settings for a run are contained in the nmrproc.properties file. It also gives the names of the input/output files. You can change these, but you do not need to do so.

The following data/files need to be supplied to run an anlysis:
* A list of candidate SMILES. This is in the files specified by the `msmsinput` property (default testall.smi). This must have one structure per line.
* The measured spectra in `spectruminput` (default realspectrum.csv). This must be a list of shifts, separated by tab. One row is one shift. As a standard, HMBC and HSQC shifts are included here.
* Set `solvent` property to the solvent used if it is `Methanol-D4 (CD3OD)` or `Chloroform-D1 (CDCl3)`. Otherwise, use `Unreported`.

Once these files are in place, run `nmrfilter.sh` (linux) or `nmrfilter.bat` (windows). This should produce the result list.

The following features are optional:
* You can include HSQCTOCSY shifts. For this, set `usehsqctocsy=true` and include the HSQCTOSY shifts in the `spectruminput` file.
* You can produce some debug output by setting `debug=true`. You need a file called `testallnames.txt` for this, which has the names of the compounds in the same order as in the `msmsinput` file.
* You can set paraemteres for tolerances and resolutions. Normally these do not need to be modified. 