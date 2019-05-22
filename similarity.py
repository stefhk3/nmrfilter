from scipy.optimize import linear_sum_assignment
import numpy as np
import csv
import math
import configparser
import sys
import os
import matplotlib.pyplot as plt

def Two_Column_List(file):
    with open(file) as input:
        mycsv = csv.reader(input, delimiter='\t', skipinitialspace=True)
        peaks = []
        i=0
        for cols in mycsv:
            if len(cols)==2:
                peaks.append([float(cols[0].strip()),float(cols[1].strip())])
            i+=1
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
		peaks_float=[float(peaks_string[0]),float(peaks_string[1]),peaks_string[2]]
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
		cluster_real.append(peaks_float)
	line=fp.readline().strip()
clusters_real.append(cluster_real)
#print(clusters_real)
fp.close()

spectrum_real = Two_Column_List(cp.get('onesectiononly', 'spectruminput'))

costs = {}
stddevs = {}
costspercompound = {}
stddevspercompound = {}
s=0
noshifts= []
xreal=[]
yreal=[]
xsim=[]
ysim=[]
for spectrum_simulated in spectra_simulated:
	xreallocal=[]
	xreallocal.append([])
	xreallocal.append([])
	xreallocal.append([])
	yreallocal=[]
	yreallocal.append([])
	yreallocal.append([])
	yreallocal.append([])
	xsimlocal=[]
	xsimlocal.append([])
	xsimlocal.append([])
	xsimlocal.append([])
	ysimlocal=[]
	ysimlocal.append([])
	ysimlocal.append([])
	ysimlocal.append([])
	if len(spectrum_simulated)>0:
		#print(str(len(spectrum_simulated))+' b '+str(len(spectrum_real)))
		cost=np.zeros((len(spectrum_simulated),len(spectrum_real)))
		i=0
		#print(cost)
		for peak_simulated in spectrum_simulated:
			k=0
			#print(str(i)+' a '+str(k))
			type=2
			if peak_simulated[2]=='b':
				type=0
			elif peak_simulated[2]=='q':
				type=1
			xsimlocal[type].append(peak_simulated[0])
			ysimlocal[type].append(peak_simulated[1])
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
				mincost=sys.float_info.max
				indexmin=0
				rowmin=0
				number_of_peaks+=1
				i=0
				for row in row_ind:
					#print(str(peak[0])+' '+str(spectrum_real[col_ind[row]][0])+' '+str(peak[1])+' '+str(spectrum_real[col_ind[row]][1]))
					if peak[0]==spectrum_real[col_ind[i]][0] and peak[1]==spectrum_real[col_ind[i]][1] and cost[row][col_ind[i]]<mincost:
						#print('hit'+str(peak)+' '+str(spectrum_real[col_ind[i]]));
						mincost=cost[row][col_ind[i]]
						indexmin=i
						rowmin=row
					i+=1
				#print(number_of_hits)
				#if not found:
					#print('no hit')
				if mincost<90:
					number_of_hits+=1
					type=2
					if spectrum_simulated[rowmin][2]=='b':
						type=0
					elif spectrum_simulated[rowmin][2]=='q':
						type=1
					xreallocal[type].append(spectrum_real[col_ind[indexmin]][0])
					yreallocal[type].append(spectrum_real[col_ind[indexmin]][1])
			hits_clusters.append(number_of_hits/number_of_peaks)
		#print(hits_clusters)
		stddevs.setdefault(np.std(hits_clusters), [])
		stddevs[np.std(hits_clusters)].append(s)
		stddevspercompound[s]=np.std(hits_clusters)
		#print('standard deviation '+str(np.std(hits_clusters)))
	else:
		noshifts.append(s)
	s+=1
	xreal.append(xreallocal)
	yreal.append(yreallocal)
	xsim.append(xsimlocal)
	ysim.append(ysimlocal)
	

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
usehsqctocsy = cp.get('onesectiononly', 'usehsqctocsy')
debug = cp.get('onesectiononly', 'debug')
if debug=='true':
	with open('testallnames.txt') as f:
		linesnames = f.read().splitlines()
	#we make plot
	i=0
	for name in linesnames:
		fig = plt.figure(figsize=(30,10))
		ax = fig.add_subplot(1,3,1)
		ax.scatter(xreal[i][0], yreal[i][0], c='red', label='measured hmbc', alpha=0.3, edgecolors='none')
		ax.scatter(xsim[i][0], ysim[i][0], c='green', label='simulated hmbc', alpha=0.3, edgecolors='none')
		ax.legend()
		ax.grid(True)
		ax = fig.add_subplot(1,3,2)
		ax.scatter(xreal[i][1], yreal[i][1], c='red', label='measured hsqc', alpha=0.3, edgecolors='none')
		ax.scatter(xsim[i][1], ysim[i][1], c='green', label='simulated hsqc', alpha=0.3, edgecolors='none')
		ax.legend()
		ax.grid(True)
		if usehsqctocsy== 'true':
			ax = fig.add_subplot(1,3,3)
			ax.scatter(xreal[i][2], yreal[i][2], c='red', label='measured hsqctocsy', alpha=0.3, edgecolors='none')
			ax.scatter(xsim[i][2], ysim[i][2], c='green', label='simulated hsqc', alpha=0.3, edgecolors='none')
			ax.legend()
			ax.grid(True)
		fig.savefig('plots'+os.sep+name+'.png', transparent=False, dpi=80, bbox_inches="tight")
		i+=1
		plt.close()
        
i=0
for cost in sorted(overallcosts):
    for position in overallcosts[cost]:
        if debug=='true':
            print(str(i+1)+': '+str(smiles[position])+'/'+str(linesnames[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: '+"{0:.2f}".format(stddevspercompound_norm[position]))
        else:
            print(str(i+1)+': '+str(smiles[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: '+"{0:.2f}".format(stddevspercompound_norm[position]))
        i+=1

for noshift in noshifts:
	print('no shifts were predicted for '+str(smiles[noshift])+' and we cannot say anything about it!')
