from igraph import *
import configparser
import csv
import louvain

def Two_Column_List(file):
    with open(file) as input:
        mycsv = csv.reader(input, delimiter='\t', skipinitialspace=True)
        peaks = []
        i=0
        for cols in mycsv:
            if len(cols)==2:
                peaks.append([i,float(cols[0].strip()),float(cols[1].strip())])
                i+=1
    return peaks

def cluster2dspectrumlouvain(cp, project):
	datapath=cp.get('datadir')

	realpeaks = Two_Column_List(datapath+os.sep+project+os.sep+cp.get('spectruminput'))
	#print(realpeaks)
	g=Graph.Read_Edgelist(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('clusteringoutput'),directed=False)
	#print(g)
	louvainresult= louvain.find_partition(g, louvain.RBERVertexPartition, resolution_parameter=float(cp.get('rberresolution')))
	#print(louvainresult)
	f=open(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('louvainoutput'),'w')
	for cluster in louvainresult:
		if len(cluster)>0:
			f.write('/\n')	
			for peak in cluster:
				for realpeak in realpeaks:
					if realpeak[0]==peak:
						f.write(str(realpeak[0])+','+str(realpeak[1])+','+str(realpeak[2])+'\n')
	f.close()
