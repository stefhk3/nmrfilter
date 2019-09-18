import configparser
import os
import subprocess
import sys
from util import *

if not len(sys.argv)!=1:
    print("You need to give a project name as parameter!")
    sys.exit(0)

project=sys.argv[1]
cp = readprops(project)
datapath=cp.get('datadir')

if not os.path.exists(datapath+os.sep+project):
	print("There is no directory "+datapath+os.sep+project+" - please check!")

if os.path.exists(datapath+os.sep+project+os.sep+"result"):
	predictionoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('predictionoutput')
	if os.path.exists(predictionoutputfile):
		os.remove(predictionoutputfile)
	clusteringoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('clusteringoutput')
	if os.path.exists(clusteringoutputfile):
		os.remove(clusteringoutputfile)
	louvainoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('louvainoutput')
	if os.path.exists(louvainoutputfile):
		os.remove(louvainoutputfile)
	predictionoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('predictionoutput')+'hsqc'
	if os.path.exists(predictionoutputfile):
		os.remove(predictionoutputfile)
	predictionoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('predictionoutput')+'hmbc'
	if os.path.exists(predictionoutputfile):
		os.remove(predictionoutputfile)
	predictionoutputfile=datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('predictionoutput')+'hsqctocsy'
	if os.path.exists(predictionoutputfile):
		os.remove(predictionoutputfile)
else:
	os.mkdir(datapath+os.sep+project+os.sep+"result")

if os.path.exists(datapath+os.sep+project+os.sep+"plots"):
	for f in os.listdir(datapath+os.sep+project+os.sep+"plots"):
		if f.endswith(".png"):
	            os.remove(os.path.join(datapath+os.sep+project+os.sep+"plots", f))
else:
	os.mkdir(datapath+os.sep+project+os.sep+"plots")

print("Simulating spectra for your compounds...")

