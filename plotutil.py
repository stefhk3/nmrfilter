import nmrplot.core as npl
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

left, upper, right, lower = 242, 175, 1727, 1281

def generateBackgrounds(cp, project):
    datapath = cp.get('datadir')

    if 'hmbcbruker' in cp.keys():

        try:
            hmbc_bruker_ar = cp.get('hmbcbruker').split(",")
            hmbc_bruker_fp = datapath+os.sep+project+os.sep+hmbc_bruker_ar[0]
            hmbc_bruker_out = datapath+os.sep+project+os.sep+"plots"+os.sep+"hmbc_spectrum.png"
        
            spectrum_hmbc = npl.Spectrum(hmbc_bruker_fp, pdata=hmbc_bruker_ar[1]);
            fig_hmbc, ax_hmbc = spectrum_hmbc.plot_spectrum(linewidth=2.5, xlims=(10, 0), ylims=(200, 0), factor=1.2)
            fig_hmbc.savefig(hmbc_bruker_out, dpi=300)
        
            hmbc_img = Image.open(hmbc_bruker_out)
            hmbc_cropped_image = hmbc_img.crop((left, upper, right, lower))
            hmbc_cropped_image.save(hmbc_bruker_out)
        except:
            print("HMBC Bruker path or format is misconfigured, background image unavailable")
    
    if 'hsqcbruker' in cp.keys():
        
        try:
            hsqc_bruker_ar = cp.get('hsqcbruker').split(",")
            hsqc_bruker_fp = datapath+os.sep+project+os.sep+hsqc_bruker_ar[0]
            hsqc_bruker_out = datapath+os.sep+project+os.sep+"plots"+os.sep+"hsqc_spectrum.png"

            spectrum_hsqc = npl.Spectrum(hsqc_bruker_fp, pdata=hsqc_bruker_ar[1]);
            fig_hsqc, ax_hsqc = spectrum_hsqc.plot_spectrum(linewidth=2.5, xlims=(10, 0), ylims=(200, 0), factor=1.2)
            fig_hsqc.savefig(hsqc_bruker_out, dpi=300)

            hsqc_img = Image.open(hsqc_bruker_out)
            hsqc_cropped_image = hsqc_img.crop((left, upper, right, lower))
            hsqc_cropped_image.save(hsqc_bruker_out)
        except:
            print("HSQC Bruker path or format is misconfigred, background image unavailable")

    if 'hsqctocsybruker' in cp.keys():

        try:

            hsqctocsy_bruker_ar = cp.get('hsqctocsybruker').split(",")
            hsqctocsy_bruker_fp = datapath+os.sep+project+os.sep+hsqctocsy_bruker_ar[0]
            hsqctocsy_bruker_out = datapath+os.sep+project+os.sep+"plots"+os.sep+"hsqctocsy_spectrum.png"

            spectrum_hsqctocsy = npl.Spectrum(hsqctocsy_bruker_fp, pdata=hsqctocsy_bruker_ar[1]);
            fig_hsqctocsy, ax_hsqc = spectrum_hsqctocsy.plot_spectrum(linewidth=2.5, xlims=(10, 0), ylims=(200, 0), factor=1.2)
            fig_hsqctocsy.savefig(hsqctocsy_bruker_out, dpi=300)

            hsqctocsy_img = Image.open(hsqctocsy_bruker_out)
            hsqctocsy_cropped_image = hsqctocsy_img.crop((left, upper, right, lower))
            hsqctocsy_cropped_image.save(hsqctocsy_bruker_out)

        except:
            print("HSQCTOCSY Bruker path or format is misconfigured, background image unavailable.")
    