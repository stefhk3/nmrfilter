import configparser
import os


def readprops(project=""):
	result={}
	cp = configparser.SafeConfigParser()
	cp.readfp(open('nmrproc.properties'))
	for (each_key, each_val) in cp.items('onesectiononly'):
		result[each_key]=each_val
	datapath=cp.get('onesectiononly', 'datadir')
	if not project=="" and os.path.exists(datapath+os.sep+project+os.sep+'nmrproc.properties'):
		cp2 = configparser.SafeConfigParser()
		cp2.readfp(open(datapath+os.sep+project+os.sep+'nmrproc.properties'))
		for (each_key, each_val) in cp2.items('onesectiononly'):
			result[each_key]=each_val
	return result
		
def checkprojectdir(datapath, project, cp):
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
	if not os.path.exists(datapath+os.sep+project+os.sep+"result"+os.sep+"smart"):
		os.mkdir(datapath+os.sep+project+os.sep+"result"+os.sep+"smart")

	if os.path.exists(datapath+os.sep+project+os.sep+"plots"):
		for f in os.listdir(datapath+os.sep+project+os.sep+"plots"):
			if f.endswith(".png"):
			    os.remove(os.path.join(datapath+os.sep+project+os.sep+"plots", f))
	else:
		os.mkdir(datapath+os.sep+project+os.sep+"plots")
