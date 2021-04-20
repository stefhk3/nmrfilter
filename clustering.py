import csv
from numpy import *
import configparser
import os

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

def setofy(peaks):
	yvalues=set()
	for peak in peaks:
		yvalues.add(peak[2])
	return yvalues

def cluster2dspectrum(cp, project):
	datapath=cp.get('datadir')
	
	C_LIMIT=float(cp.get('tolerancec'))
	H_LIMIT=float(cp.get('toleranceh'))

	peaks = Two_Column_List(datapath+os.sep+project+os.sep+cp.get('spectruminput'))
	#print(peaks)

	xclusters=[]
	yclusters=[]
	for peak in peaks:
		found=False
		for xcluster in xclusters:
			#print(str(peak[2])+'  '+str(xcluster[0][2])+'  '+str(peak[1])+'  '+str(xcluster[0][1]))
			if peak[1]>xcluster[0][1]-C_LIMIT and peak[1]<xcluster[0][1]+C_LIMIT:
				xcluster.append(peak)
				found=True
				break;
		if not found:
			xclusters.append([peak])
		found=False
		for ycluster in yclusters:
			if peak[2]>ycluster[0][2]-H_LIMIT and peak[2]<ycluster[0][2]+H_LIMIT:
				ycluster.append(peak)
				found=True
				break;
		if not found:
			yclusters.append([peak])
	#print(xclusters)
	#print(yclusters)

	found=True
	while found:
		for yindex, ycluster in enumerate(yclusters):
			xclustersnew=[]
			donexclusters=[]
			found=False
			for index1, xcluster1 in enumerate(xclusters):
				for index2, xcluster2 in enumerate(xclusters):
					if index2>index1 and setofy(xcluster1).intersection(setofy(ycluster)) and setofy(xcluster2).intersection(setofy(ycluster)):
						xpeaks=xcluster1+xcluster2
						xclustersnew.append(xpeaks)
						donexclusters.append(index1)
						donexclusters.append(index2)
						found=True
						break
				if found:
					break
			for index, xcluster in enumerate(xclusters):
				if index not in donexclusters:
					xclustersnew.append(xcluster)
			xclusters=xclustersnew
	#print(xclusters)


	#for cluster in xclusters:
		#print('cluster')
	#	for peak1 in cluster:
	#		for peak2 in cluster:
	#			if(peak2[0]>peak1[0] and ((peak1[1]>peak2[1]-C_LIMIT and peak1[1]<peak2[1]+C_LIMIT) or (peak1[2]>peak2[2]-H_LIMIT and peak1[2]<peak2[2]+H_LIMIT))):
	#				print(str(peak1[0])+'->'+str(peak2[0]))


	f=open(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('clusteringoutput'),'w')
	for cluster in xclusters:
		for peak1 in cluster:
			for peak2 in cluster:
				if(peak2[0]>=peak1[0] and ((peak1[1]>=peak2[1]-C_LIMIT and peak1[1]<=peak2[1]+C_LIMIT) or (peak1[2]>=peak2[2]-H_LIMIT and peak1[2]<=peak2[2]+H_LIMIT))):
					f.write(str(peak1[0])+' '+str(peak2[0])+'\n')
	f.close()

