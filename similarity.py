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
        type=""
        mycsv = csv.reader(input, delimiter='\t', skipinitialspace=True)
        peaks = []
        for cols in mycsv:
            if len(cols)==2:
                peaks.append([float(cols[0].strip()),float(cols[1].strip()),type])
            elif len(cols)==1:
                type=cols[0]
    return peaks

def similarity(cp, project):
	datapath=cp.get('datadir')
	fp = open(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('predictionoutput'),'r')
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

	fp = open(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('louvainoutput'),'r')
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

	spectrum_real = Two_Column_List(datapath+os.sep+project+os.sep+cp.get('spectruminput'))

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
	xrealunassigned=[]
	yrealunassigned=[]
	linesnames=[]
	#this is for calculating the optimal set
	spectraforcover=[]
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
		xreallocalunassigned=[]
		xreallocalunassigned.append([])
		xreallocalunassigned.append([])
		xreallocalunassigned.append([])
		yreallocalunassigned=[]
		yreallocalunassigned.append([])
		yreallocalunassigned.append([])
		yreallocalunassigned.append([])
		spectrumforcover=[]
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
					x=abs(peak_real[0]-peak_simulated[0])+abs((peak_real[1]-peak_simulated[1])*10)
					if (type==0 and "HMBC" in peak_real[2]) or (type==1 and "HSQC" in peak_real[2] and not "TOCSY" in peak_real[2]) or (type==2 and "HSQCTOCSY" in peak_real[2]):
						cost[i][k]=x*x
					else:
						cost[i][k]=sys.float_info.max/10000
					#if(peak_real[0]>69 and peak_real[0]<70 and peak_real[1]>5.3 and peak_real[1]<5.4):
					#	print(peak_real)
					#	print(peak_simulated)
					#	print(cost[i][k])
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
			#i=0
			#for row in row_ind:
			#	print(str(row)+' '+str(col_ind[i]))
			#	print('1_ '+str(spectrum_real[col_ind[i]])+' '+str(spectrum_simulated[row])+' '+str(cost[row][col_ind[i]]))
			#	i+=1
			hits_clusters=[]
			for cluster_real in clusters_real:
				#print('cluster')
				number_of_peaks=0
				number_of_hits=0
				for peak in cluster_real:
					mincost=sys.float_info.max
					indexmin=-1
					rowmin=-1
					number_of_peaks+=1
					i=0
					#print(str(peak[0])+'______ '+str(peak[1]))
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
					if indexmin>-1:
						#print(mincost)
						#print(str(spectrum_real[col_ind[indexmin]][0])+' '+str(spectrum_real[col_ind[indexmin]][1]))
						#print(str(spectrum_simulated[col_ind[rowmin]][0])+' '+str(spectrum_simulated[col_ind[rowmin]][1]))
						type=2
						if 'HMBC' in spectrum_real[col_ind[indexmin]][2]:
							type=0
						elif 'HSQC' in spectrum_real[col_ind[indexmin]][2] and not 'TOCSY' in spectrum_real[col_ind[indexmin]][2]:
							type=1
						if mincost<9:
							number_of_hits+=1
							xreallocal[type].append(spectrum_real[col_ind[indexmin]][0])
							yreallocal[type].append(spectrum_real[col_ind[indexmin]][1])
							spectrumforcover.append(peak)
						else:
							xreallocalunassigned[type].append(spectrum_real[col_ind[indexmin]][0])
							yreallocalunassigned[type].append(spectrum_real[col_ind[indexmin]][1])
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
		xrealunassigned.append(xreallocalunassigned)
		yrealunassigned.append(yreallocalunassigned)
		spectraforcover.append(spectrumforcover)
		
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
	#print(maxcost,costsum, stdsum)
	costspercompound_norm = {}
	stddevspercompound_norm = {}
	overallcosts={}
	for i in costspercompound:
		if maxcost-mincost!=0:
			costspercompound_norm[i]=(costspercompound[i]-mincost)/(maxcost-mincost)
		else:
			costspercompound_norm[i]=1
		if maxstddev-minstddev!=0:
			stddevspercompound_norm[i]=(stddevspercompound[i]-minstddev)/(maxstddev-minstddev)
			overallcosts.setdefault((costspercompound_norm[i]+(1-stddevspercompound_norm[i]))/2, [])
			overallcosts[(costspercompound_norm[i]+(1-stddevspercompound_norm[i]))/2].append(i)
		else:
			overallcosts.setdefault((costspercompound_norm[i]+1)/2, [])
			overallcosts[(costspercompound_norm[i]+1)/2].append(i)
		i+=1

	fp = open(datapath+os.sep+project+os.sep+cp.get('msmsinput'),'r')
	line=fp.readline().strip()
	smiles=[]
	while line:
		smiles.append(line)
		line=fp.readline().strip()
	usehsqctocsy = cp.get('usehsqctocsy')
	debug = cp.get('debug')
	usehmbc = cp.get('usehmbc')
	if os.path.exists(datapath+os.sep+project+os.sep+cp.get('msmsinput')[0:len(cp.get('msmsinput'))-4]+'names.txt'):
		with open(datapath+os.sep+project+os.sep+cp.get('msmsinput')[0:len(cp.get('msmsinput'))-4]+'names.txt') as f:
			linesnames = f.read().splitlines()
	#we make plot
	i=0
	for name in linesnames:
		#print(name)
		fig = plt.figure(figsize=(30,10))
		if usehmbc!= 'false':
			ax = fig.add_subplot(1,3,1)
			ax.invert_xaxis()
			ax.invert_yaxis()
			ax.scatter(ysim[i][0], xsim[i][0], c='grey', label='simulated hmbc ('+str(len(ysim[i][0]))+')', alpha=0.6, edgecolors='none', s=12)
			if len(xreal[i][0])>0:
				ax.scatter(yreal[i][0], xreal[i][0], c='green', label='measured assigned ('+str(len(yreal[i][0]))+')', alpha=1, edgecolors='none', s=12)
			ax.scatter(yrealunassigned[i][0], xrealunassigned[i][0], c='blue', label='measured unassigned closest shifts ('+str(len(yrealunassigned[i][0]))+')', alpha=0.6, edgecolors='none', s=12)
			if debug=='true':
				xrealrest=[]
				yrealrest=[]
				for peak_real in spectrum_real:
					if peak_real[2] == "HMBC":
						valuecontained=False
						if peak_real[0] in xreal[i][0] and peak_real[1] in yreal[i][0]:
							if xreal[i][0].index(peak_real[0])!=yreal[i][0].index(peak_real[1]):
								valuecontained=True
						if peak_real[0] in xrealunassigned[i][0] and peak_real[1] in yrealunassigned[i][0]:
							if xrealunassigned[i][0].index(peak_real[0])!=yrealunassigned[i][0].index(peak_real[1]):
								valuecontained=True
						if not valuecontained:
							xrealrest.append(peak_real[0])
							yrealrest.append(peak_real[1])
				if len(xrealrest)>0:
					ax.scatter(yrealrest, xrealrest, c='red', label='measured unused ('+str(len(yrealrest))+')', alpha=0.2, edgecolors='none', s=12)
			ax.legend()
			ax.grid(True)
		ax = fig.add_subplot(1,3,2)
		ax.invert_xaxis()
		ax.invert_yaxis()
		ax.scatter(ysim[i][1], xsim[i][1], c='grey', label='simulated hsqc ('+str(len(ysim[i][1]))+')', alpha=0.6, edgecolors='none', s=12)
		if len(xreal[i][1])>0:
			ax.scatter(yreal[i][1], xreal[i][1], c='green', label='measured assigned ('+str(len(yreal[i][1]))+')', alpha=1, edgecolors='none', s=12)
		ax.scatter(yrealunassigned[i][1], xrealunassigned[i][1], c='blue', label='measured unassigned closest shifts ('+str(len(yrealunassigned[i][1]))+')', alpha=0.6, edgecolors='none', s=12)
		if debug=='true':
			xrealrest=[]
			yrealrest=[]
			#print(xreal[i][1])
			#print(yreal[i][1])
			#print(xrealunassigned[i][1])
			#print(yrealunassigned[i][1])
			#print(xsim[i][1])
			#print(ysim[i][1])
			for peak_real in spectrum_real:
				if "HSQC" in peak_real[2] and not "TOCSY" in peak_real[2]:
					valuecontained=False
					#print(peak_real)
					#print(peak_real[0] in xrealunassigned[i][1])
					#print(peak_real[1] in yrealunassigned[i][1])
					if peak_real[0] in xreal[i][1] and peak_real[1] in yreal[i][1]:
						#if xreal[i][1].index(peak_real[0])!=yreal[i][1].index(peak_real[1]):
							valuecontained=True
					if peak_real[0] in xrealunassigned[i][1] and peak_real[1] in yrealunassigned[i][1]:
						#if xrealunassigned[i][1].index(peak_real[0])!=yrealunassigned[i][1].index(peak_real[1]):
							valuecontained=True
					if not valuecontained:
						xrealrest.append(peak_real[0])
						yrealrest.append(peak_real[1])
			if len(xrealrest)>0:
				ax.scatter(yrealrest, xrealrest, c='red', label='measured unused ('+str(len(yrealrest))+')', alpha=0.2, edgecolors='none', s=12)
				#print(xrealrest)
				#print(yrealrest)
		ax.legend()
		ax.grid(True)
		if usehsqctocsy== 'true':
			ax = fig.add_subplot(1,3,3)
			ax.invert_xaxis()
			ax.invert_yaxis()
			ax.scatter(ysim[i][2], xsim[i][2], c='grey', label='simulated hsqc ('+str(len(ysim[i][2]))+')', alpha=0.6, edgecolors='none', s=12)
			if len(xreal[i][2])>0:
				ax.scatter(yreal[i][2], xreal[i][2], c='green', label='measured assigned ('+str(len(yreal[i][2]))+')', alpha=1, edgecolors='none', s=12)
			ax.scatter(yrealunassigned[i][2], xrealunassigned[i][2], c='blue', label='measured unassigned closest shifts ('+str(len(yrealunassigned[i][2]))+')', alpha=0.6, edgecolors='none', s=12)
			if debug=='true':
				xrealrest=[]
				yrealrest=[]
				for peak_real in spectrum_real:
					if "HSQCTOCSY" in peak_real[2]:
						valuecontained=False
						if peak_real[0] in xreal[i][2] and peak_real[1] in yreal[i][2]:
							if xreal[i][2].index(peak_real[0])!=yreal[i][2].index(peak_real[1]):
								valuecontained=True
						if peak_real[0] in xrealunassigned[i][2] and peak_real[1] in yrealunassigned[i][2]:
							if xrealunassigned[i][2].index(peak_real[0])!=yrealunassigned[i][2].index(peak_real[1]):
								valuecontained=True
						if not valuecontained:
							xrealrest.append(peak_real[0])
							yrealrest.append(peak_real[1])
				if len(xrealrest)>0:
					ax.scatter(yrealrest, xrealrest, c='red', label='measured unused ('+str(len(yrealrest))+')', alpha=0.2, edgecolors='none', s=12)
			ax.legend()
			ax.grid(True)
		fig.savefig(datapath+os.sep+project+os.sep+'plots'+os.sep+name+'.png', transparent=False, dpi=80, bbox_inches="tight")
		i+=1
		plt.close()
		
	i=0
	fp = open(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('result'),'w')
	fpcsv = open(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('result')+'.csv','w')
	if len(linesnames)>0:
		fpcsv.write('ID\tsmiles\tname\tdistance\tstandard deviation')
	else:
		fpcsv.write('ID\tsmiles\tdistance\tstandard deviation')
	if usehmbc!= 'false':
		fpcsv.write('\tmatching rate (HMBC)\t')
	fpcsv.write('\tmatching rate (HSQC)\t')
	if usehsqctocsy== 'true':
		fpcsv.write('\tmatching rate (HSQC-TOCSY)\t')
	fpcsv.write('\n')
	for cost in sorted(overallcosts):
		for position in overallcosts[cost]:
			matchingrate=''
			matchingratecsv=''
			if usehmbc!= 'false':
				rate='n/a'
				ratecsv='n/a'
				if len(ysim[position][0])>0:
					rate='{:.2%}'.format(len(yreal[position][0])/len(ysim[position][0]))
					ratecsv='{:.2%}'.format(len(yreal[position][0])/len(ysim[position][0]))
				matchingrate=matchingrate+', matching rate: '+str(len(yreal[position][0]))+'/'+str(len(ysim[position][0]))+', '+rate+' (HMBC)'
				matchingratecsv=matchingratecsv+'\t'+str(len(yreal[position][0]))+'/'+str(len(ysim[position][0]))+'\t'+ratecsv
			matchingrate=matchingrate+', matching rate: '+str(len(yreal[position][1]))+'/'+str(len(ysim[position][1]))+', '+'{:.2%}'.format(len(yreal[position][1])/len(ysim[position][1]))+' (HSQC)'
			matchingratecsv=matchingratecsv+'\t'+str(len(yreal[position][1]))+'/'+str(len(ysim[position][1]))+'\t'+'{:.2%}'.format(len(yreal[position][1])/len(ysim[position][1]))
			if usehsqctocsy== 'true':
				matchingrate=matchingrate+', matching rate: '+str(len(yreal[position][2]))+'/'+str(len(ysim[position][2]))+', '+'{:.2%}'.format(len(yreal[position][2])/len(ysim[i][2]))+' (HSQC-TOCSY)'
				matchingratecsv=matchingratecsv+'\t'+str(len(yreal[position][2]))+'/'+str(len(ysim[position][2]))+'\t'+'{:.2%}'.format(len(yreal[position][2])/len(ysim[i][2]))
			if len(linesnames)>0:
				if maxstddev-minstddev!=0:
					print(str(i+1)+': '+str(smiles[position])+'/'+str(linesnames[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: '+"{0:.2f}".format(stddevspercompound_norm[position])+matchingrate)
					fp.write(str(i+1)+': '+str(smiles[position])+'/'+str(linesnames[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: '+"{0:.2f}".format(stddevspercompound_norm[position])+matchingrate+'\n')
					fpcsv.write(str(i+1)+'\t'+str(smiles[position])+'\t'+str(linesnames[position])+'\t'+"{0:.2f}".format(costspercompound_norm[position])+'\t'+"{0:.2f}".format(stddevspercompound_norm[position])+matchingratecsv+'\n')
				else:
					print(str(i+1)+': '+str(smiles[position])+'/'+str(linesnames[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation:  n/a'+matchingrate)
					fp.write(str(i+1)+': '+str(smiles[position])+'/'+str(linesnames[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation:  n/a'+matchingrate+'\n')
					fpcsv.write(str(i+1)+'\t'+str(smiles[position])+'\t'+str(linesnames[position])+'\t'+"{0:.2f}".format(costspercompound_norm[position])+'\t'+matchingratecsv+'\n')
			else:
				if maxstddev-minstddev!=0:
					print(str(i+1)+': '+str(smiles[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: '+"{0:.2f}".format(stddevspercompound_norm[position])+matchingrate)
					fp.write(str(i+1)+': '+str(smiles[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: '+"{0:.2f}".format(stddevspercompound_norm[position])+matchingrate+'\n')
					fpcsv.write(str(i+1)+'\t'+str(smiles[position])+'\t'+"{0:.2f}".format(costspercompound_norm[position])+'\t'+"{0:.2f}".format(stddevspercompound_norm[position])+matchingratecsv+'\n')
				else:
					print(str(i+1)+': '+str(smiles[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: n/a'+matchingrate)
					fp(str(i+1)+': '+str(smiles[position])+', distance: '+"{0:.2f}".format(costspercompound_norm[position])+', standard deviation: n/a'+'\n')
					fpcsv(str(i+1)+'\t'+str(smiles[position])+'\t'+"{0:.2f}".format(costspercompound_norm[position])+'\tn/a'+'\n')
			i+=1

	for noshift in noshifts:
		print('no shifts were predicted for '+str(smiles[noshift])+' and we cannot say anything about it!')
	fp.close()
	fpcsv.close()

	#now we find the optimal set. We have spectraforcover containing the peaks covered by each spectrum, costspercompound_norm containing the distances, and stddevspercompound_norm containing the std devs
	#usedspectra = [False for i in range(len(spectraforcover))] 
	#addspectrum(usedspectra, spectraforcover,0, costspercompound_norm, stddevspercompound_norm)

best=0
bestspectrum=[]
count=0

def addspectrum(usedspectra, spectraforcover, start, costspercompound_norm, stddevspercompound_norm):
	global best
	global bestspectrum
	global count
	for x in range(start,len(usedspectra)):
		if count% 10000 == 0:
			print(count)
		count+=1
		newusedspectra=usedspectra.copy()
		if newusedspectra[x]==False:
			newusedspectra[x]=True
			usedpeaks=[]
			numberofpeaks=0
			numberofmultiplepeaks=0
			distance=0
			stddev=0
			for y in range(0,len(newusedspectra)):
				if newusedspectra[y]==True:
					distance+=costspercompound_norm[y]
					stddev+=stddevspercompound_norm[y]
					for peak in spectraforcover[y]:
						if peak in usedpeaks:
							numberofmultiplepeaks+=1
						else:
							numberofpeaks+=1
							usedpeaks.append(peak)
			if (numberofpeaks-numberofmultiplepeaks)>best:
				best=numberofpeaks-numberofmultiplepeaks
				bestspectrum=newusedspectra
				print("combination "+str(newusedspectra).strip('[]')+", best "+str(best)+" "+str((numberofpeaks-numberofmultiplepeaks))+", peaks: "+str(numberofpeaks)+", multiple peaks: "+str(numberofmultiplepeaks)+", distance: "+"{0:.2f}".format(distance)+", std dev: "+"{0:.2f}".format(stddev))
			addspectrum(newusedspectra, spectraforcover, x+1, costspercompound_norm, stddevspercompound_norm)
