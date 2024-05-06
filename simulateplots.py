import csv
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from nmrutil import *


 


def make_plots(cp, project):
    datapath=cp.get('datadir')
    fp = open(datapath+os.sep+project+os.sep+'result'+os.sep+cp.get('predictionoutput'),'r')
    
    spectra_simulated_dicts = []
    spectrum_simulated_dict = {}
    line=fp.readline().strip()
    while line:
        if line == '/':
            spectra_simulated_dicts.append(spectrum_simulated_dict)
            #spectrum_simulated=[]
            spectrum_simulated_dict = {}
        else:
            peaks_string=line.split(",")
            
            peak_y = float(peaks_string[0])
            peak_x = float(peaks_string[1])
            peak_type = peaks_string[2]
            #peak_c = int(peaks_string[3])
            #peak_h = int(peaks_string[4]) q: [], [] || m: [], [] || t: [], y: []

            if peak_type in spectrum_simulated_dict:
                spectrum_simulated_dict[peak_type][0].append(peak_x)
                spectrum_simulated_dict[peak_type][1].append(peak_y)
            else:
                spectrum_simulated_dict[peak_type] = [[peak_x], [peak_y]]
        
        line=fp.readline().strip()
	#spectra_simulated.append(spectrum_simulated)
	#print(spectra_simulated)
    fp.close()

    fp = open(datapath+os.sep+project+os.sep+cp.get('msmsinput'),'r')
    line=fp.readline().strip()
    smiles=[]
    while line:
        smiles.append(line)
        line = fp.readline().strip()
    
    usehsqctocsy = cp.get('usehsqctocsy')
    usehmbc = cp.get('usehmbc')

    if os.path.exists(datapath+os.sep+project+os.sep+cp.get('msmsinput')[0:len(cp.get('msmsinput'))-4]+'names.txt'):
        with open(datapath+os.sep+project+os.sep+cp.get('msmsinput')[0:len(cp.get('msmsinput'))-4]+'names.txt') as f:
            linesnames = f.read().splitlines()
    i = 0
    for name in linesnames:
        plt_w = 15
        plt_h = 30
        mosaic = """A;Q"""

        if usehmbc == 'true':
            if usehsqctocsy == 'true':
                mosaic = "AAA;BQC"
                plt_w = 45
                
            else:  
                mosaic = "AA;BQ"
                plt_w = 30
        elif usehmbc == 'false' and usehsqctocsy == 'true':
            mosaic = "AA;QC"
            plt_w = 30

        fig = plt.figure(figsize=(plt_w, plt_h))
        ax_dict = fig.subplot_mosaic(mosaic)

        try:
            skeletal_structure = plt.imread(datapath+os.sep+project+os.sep+"reports"+os.sep+name+".jpg")			
        except:
            print("Structure image not found for " + name)
            pass
		
		
        ax_dict['A'].axis('off')
        ax_dict['A'].imshow(skeletal_structure)

        ax_dict['Q'].set_xlim([10, 0])
        ax_dict['Q'].set_ylim([200, 0])
        ax_dict['Q'].scatter(spectra_simulated_dicts[i]['q'][0], spectra_simulated_dicts[i]['q'][1], c='blue', label='Simulated HSQC', alpha=0.6, edgecolors='none', s=50)
        ax_dict['Q'].legend()
        ax_dict['Q'].grid(True)

        if usehmbc != 'false':
            ax_dict['B'].set_xlim([10, 0])
            ax_dict['B'].set_ylim([200, 0])
            ax_dict['B'].scatter(spectra_simulated_dicts[i]['b'][0], spectra_simulated_dicts[i]['b'][1], c='blue', label='Simulated HMBC', alpha=0.6, edgecolors='none', s=50)
            ax_dict['B'].legend()
            ax_dict['B'].grid(True)


        if usehsqctocsy != 'false':
            ax_dict['C'].set_xlim([10, 0])
            ax_dict['C'].set_ylim([200, 0])
            ax_dict['C'].scatter(spectra_simulated_dicts[i]['t'][0], spectra_simulated_dicts[i]['t'][1], c='blue', label='Simulated HSQCTOCSY', alpha=0.6, edgecolors='none', s=50)
            ax_dict['C'].legend()
            ax_dict['C'].grid(True)

        save_path = datapath+os.sep+project+os.sep+'sim_plots'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path+os.sep+name+'.png', transparent=False, dpi=80, bbox_inches="tight")
		
        i+=1
        plt.close()

if __name__ == '__main__':
    project = sys.argv[1]
    cp = readprops(project)
    print("Plotting spectra..")
    make_plots(cp, project)
