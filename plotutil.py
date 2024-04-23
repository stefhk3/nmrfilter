import nmrplot.core as npl
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image

left, upper, right, lower = 242, 175, 1727, 1281

def generateBackgrounds(cp, project):
    datapath = cp.get('datadir')

    hmbc_bruker_ar = cp.get('hmbcbruker').split(",")
    hmbc_bruker_fp = datapath+os.sep+project+os.sep+hmbc_bruker_ar[0]
    hmbc_bruker_out = datapath+os.sep+project+os.sep+"plots"+os.sep+"hmbc_spectrum.png"
	
    hsqc_bruker_ar = cp.get('hsqcbruker').split(",")
    hsqc_bruker_fp = datapath+os.sep+project+os.sep+hsqc_bruker_ar[0]
    hsqc_bruker_out = datapath+os.sep+project+os.sep+"plots"+os.sep+"hsqc_spectrum.png"

    spectrum_hmbc = npl.Spectrum(hmbc_bruker_fp, pdata=1);
    fig_hmbc, ax_hmbc = spectrum_hmbc.plot_spectrum(linewidth=2.5, xlims=(10, 0), ylims=(200, 0), factor=1.2)
    fig_hmbc.savefig(hmbc_bruker_out, dpi=300)

    spectrum_hsqc = npl.Spectrum(hsqc_bruker_fp, pdata=1);
    fig_hsqc, ax_hsqc = spectrum_hsqc.plot_spectrum(linewidth=2.5, xlims=(10, 0), ylims=(200, 0), factor=1.2)
    fig_hsqc.savefig(hsqc_bruker_out, dpi=300)


    hmbc_img = Image.open(hmbc_bruker_out)
    hmbc_cropped_image = hmbc_img.crop((left, upper, right, lower))
    hmbc_cropped_image.save(hmbc_bruker_out)

    hsqc_img = Image.open(hsqc_bruker_out)
    hsqc_cropped_image = hsqc_img.crop((left, upper, right, lower))
    hsqc_cropped_image.save(hsqc_bruker_out)

    