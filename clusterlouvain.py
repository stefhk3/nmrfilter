from igraph import *
import configparser
import csv
import louvain

def Three_Column_List(file):
    with open(file) as input:
        mycsv = csv.reader(input, delimiter='\t', skipinitialspace=True)
        peaks = []
        for cols in mycsv:
            peaks.append([int(cols[0].strip()),float(cols[1].strip()),float(cols[2].strip())])
    return peaks

cp = configparser.SafeConfigParser()
cp.readfp(open('nmrproc.properties'))

realpeaks = Three_Column_List(cp.get('onesectiononly', 'spectruminput'))
#print(realpeaks)
g=Graph.Read_Edgelist(cp.get('onesectiononly', 'clusteringoutput'),directed=False)
#print(g)
louvainresult= louvain.find_partition(g, louvain.RBERVertexPartition, resolution_parameter=float(cp.get('onesectiononly', 'rberresolution')))
#print(louvainresult)
f=open(cp.get('onesectiononly', 'louvainoutput'),'w')
for cluster in louvainresult:
	if len(cluster)>0:
		f.write('/\n')	
		for peak in cluster:
			for realpeak in realpeaks:
				if realpeak[0]==peak:
					f.write(str(realpeak[0])+','+str(realpeak[1])+','+str(realpeak[2])+'\n')
f.close()
