from igraph import *
import configparser
import csv
import louvain

def Two_Column_List(file):
    with open(file) as input:
        type=""
        mycsv = csv.reader(input, delimiter='\t', skipinitialspace=True)
        peaks = []
        i=0
        for cols in mycsv:
            if len(cols)==2:
                peaks.append([i,float(cols[0].strip()),float(cols[1].strip()),type])
                i+=1
            elif len(cols)==1:
                type=cols[0]
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
	i=0
	for cluster in louvainresult:
		if len(cluster)>0:
			fsmarts=open(datapath+os.sep+project+os.sep+'result'+os.sep+'smart'+os.sep+'smart'+str(i)+'.csv','w')
			fsmarts.write('13C,1H\n')
			f.write('/\n')	
			for peak in cluster:
				for realpeak in realpeaks:
					if realpeak[0]==peak:
						f.write(str(realpeak[0])+','+str(realpeak[1])+','+str(realpeak[2])+'\n')
						if("HSQC" in realpeak[3] and not "TOCSY" in realpeak[3]):
							fsmarts.write(str(realpeak[1])+','+str(realpeak[2])+'\n')
			i += 1
	f.close()
