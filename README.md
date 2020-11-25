This is the code used in the paper https://pubs.rsc.org/en/content/articlelanding/2019/fd/c8fd00227d. The data in realspectrum.csv are the example for P. boldus. Running the script nmrfilter will produce the list of hits for P. boldus.

#Installation

Requirements are Java and Python. For Java, version 1.8 or higher is needed. A JRE (Java Runtime Environment) is enough, a JDK is not required. The default of any operating system should do.
 
Python version must be 3 (3.66 was tested). You need numpy, scipy, louvain, and python-igraph libraires. Using pip, you can install them by doing

pip3 install numpy

pip3 install scipy

pip3 install python-igraph

pip3 install louvain

Notice igraph is a different library. If install python-igraph gives an error about missing C libraries, try using a wheel, following https://stackoverflow.com/questions/34113151/how-to-install-igraph-for-python-on-windows

#Use of respredict

If you want to use the respredict prediction (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0374-3) giving better results, you need to install more packages. The easiest way is to use the two yaml files environment-cpu.yml (for usage of the CPU only) or environment.yml (for using the GPU). They use Anaconda to install an environment. The command is `conda env create  -f environment-cpu.yml` respectively `conda env create -f environment.yml`. You can then activate the environment with `conda activate nmrfilter`.

#Running

The program works on projects, where each project is a folder. It must contain the required files (see below) and any results will be written to it. If you checkout the repository https://github.com/stefhk3/nmrfilterprojects you can use this as example projects.

The settings for a run are contained in the nmrproc.properties file. It also gives the names of the input/output files. You can change these, but you do not need to do so. The property datadir is where the projects/folders are searched for. If you have downloaded the nmrfilterprojects examples, set this to the nmrfilterprojects directory.
 directory.
The following data/files need to be supplied to run an anlysis in the project folder you want to work on:
* A list of candidate SMILES. This is in the files specified by the `msmsinput` property (default testall.smi). This must have one structure per line.
* The measured spectra in `spectruminput` (default realspectrum.csv). This must be a list of shifts, separated by tab. One row is one shift. As a standard, HMBC and HSQC shifts are included here.
* Set `solvent` property to the solvent used if it is `Methanol-D4 (CD3OD)` or `Chloroform-D1 (CDCl3)`. Otherwise, use `Unreported`.

Once these files are in place, run `nmrfilter.sh <projectname>` (linux) or `nmrfilter.bat <projectname>` (windows). This should produce the result list. Replace <projectname> by the name of the project/folder you want to work on.

The following features are optional:
* You can include HSQCTOCSY shifts. For this, set `usehsqctocsy=true` and include the HSQCTOSY shifts in the `spectruminput` file.
* You can produce some debug output by setting `debug=true`. You need a file called `testallnames.txt` for this, which has the names of the compounds in the same order as in the `msmsinput` file.
* You can set paraemteres for tolerances and resolutions. Normally these do not need to be modified. 
* With `usedeeplearning=true` you can activate the respredict prediction (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0374-3) instead of the HOSE code based one. This gives better results, but requires the installation as described above. 
