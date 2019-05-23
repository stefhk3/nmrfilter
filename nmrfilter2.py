import configparser
import os
import subprocess
import sys
from clustering import *
from clusterlouvain import *
from similarity import *

cp = configparser.SafeConfigParser()
cp.readfp(open('nmrproc.properties'))
datapath=cp.get('onesectiononly', 'datadir')
project=sys.argv[1]

predictionoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'predictionoutput')
clusteringoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'clusteringoutput')
louvainoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'louvainoutput')

print("Clustering the peaks in the measured spectrum...")
cluster2dspectrum(cp, project)
print("Detecting communities in the measures spectrum...")
cluster2dspectrumlouvain(cp, project)
print("Calculating best hits in your compounds...")
similarity(cp, project)

