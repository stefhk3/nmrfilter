import configparser
import os


def readprops(project):
	result={}
	cp = configparser.SafeConfigParser()
	cp.readfp(open('nmrproc.properties'))
	for (each_key, each_val) in cp.items('onesectiononly'):
		result[each_key]=each_val
	datapath=cp.get('onesectiononly', 'datadir')
	if os.path.exists(datapath+os.sep+project+os.sep+'nmrproc.properties'):
		cp2 = configparser.SafeConfigParser()
		cp2.readfp(open(datapath+os.sep+project+os.sep+'nmrproc.properties'))
		for (each_key, each_val) in cp2.items('onesectiononly'):
			result[each_key]=each_val
	return result
		

