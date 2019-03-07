from scipy.optimize import linear_sum_assignment
import numpy as np
import csv
import math
import configparser
import sys

def Three_Column_List(file):
    with open(file) as input:
        mycsv = csv.reader(input, delimiter='\t', skipinitialspace=True)
        peaks = []
        for cols in mycsv:
            peaks.append([float(cols[1].strip()),float(cols[2].strip())])
    return peaks

cp = configparser.SafeConfigParser()
cp.readfp(open('nmrproc.properties'))

fp = open(cp.get('onesectiononly', 'predictionoutput'),'r')
spectra_simulated = []
spectrum_simulated= []
line=fp.readline().strip()
while line:
	if line is '/':
		spectra_simulated.append(spectrum_simulated)
		spectrum_simulated=[]
	else:
		peaks_string=line.split(",")
		peaks_float=[float(peaks_string[0]),float(peaks_string[1])]
		spectrum_simulated.append(peaks_float)
	line=fp.readline().strip()
#spectra_simulated.append(spectrum_simulated)
#print(spectra_simulated)
fp.close()

fp = open(cp.get('onesectiononly', 'louvainoutput'),'r')
clusters_real = []
cluster_real= []
line=fp.readline().strip()
while line:
	if line is '/':
		if len(cluster_real)>0:
			clusters_real.append(cluster_real)
		cluster_real=[]
	else:
		peaks_string=line.split(",")
		peaks_float=[float(peaks_string[1]),float(peaks_string[2])]
		#spectrum_real.append(peaks_float)
		cluster_real.append(peaks_float)
	line=fp.readline().strip()
clusters_real.append(cluster_real)
#print(clusters_real)
fp.close()

spectrum_real = Three_Column_List(cp.get('onesectiononly', 'spectruminput'))

costs = {}
stddevs = {}
costspercompound = {}
stddevspercompound = {}
s=0
noshifts= []
for spectrum_simulated in spectra_simulated:
	if len(spectrum_simulated)>0:
		#print(str(len(spectrum_simulated))+' b '+str(len(spectrum_real)))
		cost=np.zeros((len(spectrum_simulated),len(spectrum_real)))
		i=0
		#print(cost)
		for peak_simulated in spectrum_simulated:
			k=0
			#print(str(i)+' a '+str(k))
			for peak_real in spectrum_real:
				cost[i][k]=(abs(peak_real[0]-peak_simulated[0])+abs((peak_real[1]-peak_simulated[1])*10))*(abs(peak_real[0]-peak_simulated[0])+abs((peak_real[1]-peak_simulated[1]*10)))
				#if(cost[i][k]<90):
				#print(peak_real)
				#print(peak_simulated)
				#print(cost[i][k])
				k+=1
			i+=1
		#print(cost)
		row_ind, col_ind=linear_sum_assignment(cost)
		#print('cost: '+str(cost[row_ind,col_ind].sum())+'   '+str(len(spectrum_simulated)));
		#print('cost: '+str(cost[row_ind,col_ind].sum()/len(spectrum_simulated)));
		costs.setdefault(cost[row_ind,col_ind].sum()/len(spectrum_simulated), [])
		costs[cost[row_ind,col_ind].sum()/len(spectrum_simulated)].append(s)
		costspercompound[s]=cost[row_ind,col_ind].sum()/len(spectrum_simulated)
		#print("___________")
		#print(costspercompound[s])
		#print(row_ind)
		#print(col_ind)
		i=0
		#for row in row_ind:
			#print(str(row)+' '+str(col_ind[i]))
			#print('1_ '+str(spectrum_real[col_ind[i]])+' '+str(spectrum_simulated[row])+' '+str(cost[row][col_ind[i]]))
			#i+=1
		hits_clusters=[]
		for cluster_real in clusters_real:
			#print('cluster')
			number_of_peaks=0
			number_of_hits=0
			for peak in cluster_real:
				found=False
				number_of_peaks+=1
				i=0
				for row in row_ind:
					#print(str(peak[0])+' '+str(spectrum_real[col_ind[row]][0])+' '+str(peak[1])+' '+str(spectrum_real[col_ind[row]][1]))
					if peak[0]==spectrum_real[col_ind[i]][0] and peak[1]==spectrum_real[col_ind[i]][1] and cost[row][col_ind[i]]<90:
						#print('hit'+str(peak)+' '+str(spectrum_real[col_ind[i]]));
						found=True
						number_of_hits+=1
						break;
					i+=1
				#print(number_of_hits)
				#if not found:
					#print('no hit')
			hits_clusters.append(number_of_hits/number_of_peaks)
		#print(hits_clusters)
		stddevs.setdefault(np.std(hits_clusters), [])
		stddevs[np.std(hits_clusters)].append(s)
		stddevspercompound[s]=np.std(hits_clusters)
		#print('standard deviation '+str(np.std(hits_clusters)))
	else:
		noshifts.append(s)
	s+=1

#we have now got the costs in costs and the standard deviations of the cluster distributions in stddists, so we can do the normalisation
#print(costs)
#print(stddevs)
costsorder=[]
stddevsorder=[]
maxcost=0
mincost=sys.float_info.max
maxstddev=0
minstddev=sys.float_info.max
costsum=0
stdsum=0
for cost in costs:
	if cost<mincost:
		mincost=cost
	if cost>maxcost:
		maxcost=cost
	costsum=costsum+len(costs[cost])
for stddev in stddevs:
	if stddev<minstddev:
		minstddev=stddev
	if stddev>maxstddev:
		maxstddev=stddev
	stdsum=stdsum+len(stddevs[stddev])
#print(costsum, stdsum)
costspercompound_norm = {}
stddevspercompound_norm = {}
overallcosts={}
for i in costspercompound:
	costspercompound_norm[i]=(costspercompound[i]-mincost)/(maxcost-mincost)
	stddevspercompound_norm[i]=(stddevspercompound[i]-minstddev)/(maxstddev-minstddev)
	overallcosts.setdefault((costspercompound_norm[i]+(1-stddevspercompound_norm[i]))/2, [])
	overallcosts[(costspercompound_norm[i]+(1-stddevspercompound_norm[i]))/2].append(i)
	i+=1

fp = open(cp.get('onesectiononly', 'msmsinput'),'r')
line=fp.readline().strip()
smiles=[]
while line:
	smiles.append(line)
	line=fp.readline().strip()

#with open('testallnames.txt') as f:
#     linesnames = f.read().splitlines()
i=0
for cost in sorted(overallcosts):
	for position in overallcosts[cost]:
		print(str(i+1)+': '+str(smiles[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: '+"{0:.2f}".format(stddevspercompound_norm[position]))
		#print(str(i+1)+': '+str(smiles[position])+'/'+str(linesnames[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: '+"{0:.2f}".format(stddevspercompound_norm[position]))
		i+=1

for noshift in noshifts:
	print('no shifts were predicted for '+str(smiles[noshift])+' and we cannot say anything about it!')
