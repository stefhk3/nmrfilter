This is the code used in the paper [add ref once available]. The data in realspectrum.csv are the example for P. boldus. Running the script nmrfilter will produce the list of hits for P. boldus.
 
Requirements are Java and Python. For Java, version 1.8 or higher is needed. A JRE (Java Runtime Environment) is enough, a JDK is not required. The default of any operating system should do.
 
Python version must be 3 (3.66 was tested). You need numpy, scipy, louvain, and python-igraph libraires. Using pip, you can install them by doing

pip3 install numpy

pip3 install scipy

pip3 install python-igraph

pip3 install louvain

Notice igraph is a different library. If install python-igraph gives an error about missing C libraries, try using a wheel, following https://stackoverflow.com/questions/34113151/how-to-install-igraph-for-python-on-windows