import configparser
import os
import subprocess
import sys

cp = configparser.SafeConfigParser()
cp.readfp(open('nmrproc.properties'))
datapath=cp.get('onesectiononly', 'datadir')
project=sys.argv[1]

predictionoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'predictionoutput')
if os.path.exists(predictionoutputfile):
	os.remove(predictionoutputfile)
clusteringoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'clusteringoutput')
if os.path.exists(clusteringoutputfile):
	os.remove(clusteringoutputfile)
louvainoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'louvainoutput')
if os.path.exists(louvainoutputfile):
	os.remove(louvainoutputfile)
predictionoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'predictionoutput')+'hsqc'
if os.path.exists(predictionoutputfile):
	os.remove(predictionoutputfile)
predictionoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'predictionoutput')+'hmbc'
if os.path.exists(predictionoutputfile):
	os.remove(predictionoutputfile)
predictionoutputfile=datapath+os.sep+project+os.sep+cp.get('onesectiononly', 'predictionoutput')+'hsqctocsy'
if os.path.exists(predictionoutputfile):
	os.remove(predictionoutputfile)

if os.path.exists(datapath+os.sep+project+os.sep+"plots"):
	for f in os.listdir(datapath+os.sep+project+os.sep+"plots"):
		if f.endswith(".png"):
	            os.remove(os.path.join(datapath+os.sep+project+os.sep+"plots", f))
else:
	os.mkdir(datapath+os.sep+project+os.sep+"plots")

print("Simulating spectra for your compounds...")

