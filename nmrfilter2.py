import configparser
import os
import subprocess
import sys
from clustering import *
from clusterlouvain import *
from similarity import *
from util import *

project=sys.argv[1]
cp = readprops(project)
datapath=cp.get('datadir')

predictionoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('predictionoutput')
clusteringoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('clusteringoutput')
louvainoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('louvainoutput')

print("Clustering the peaks in the measured spectrum...")
cluster2dspectrum(cp, project)
print("Detecting communities in the measures spectrum...")
cluster2dspectrumlouvain(cp, project)
print("Calculating best hits in your compounds...")
similarity(cp, project)

