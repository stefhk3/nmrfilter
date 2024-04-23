import nmrplot.core as npl
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image


hmbc_bruker_fp = "/home/karl/nmrfilterprojects/artificial/15"
#left, right, upper, bottom = 241, 1727, 1280, 174
left, upper, right, lower = 242, 175, 1727, 1281

#def identify_axes(axd):
#    for ax, label in axd.items():
#        ax.text(0.5, 0.5, label, va='center', ha='center')


#fig = plt.figure(figsize=(30,10))
#ax = fig.add_subplot(1, 3, 3)
#ax.set_xlim([10, 0])
#ax.set_ylim([200, 0])
#ax.plot([0, 25], [24, 64], color='red', linestyle='--')

#plt.subplot(1, 3, 3)
#mosaic = """AB;CD"""


spectrum_hmbc = npl.Spectrum(hmbc_bruker_fp, pdata=1);
fig_hmbc, ax_hmbc = spectrum_hmbc.plot_spectrum(linewidth=2.5, xlims=(10, 0), ylims=(200, 0), factor=1.2)
fig_hmbc.savefig("plswork.png", dpi=300)

image = Image.open("plswork.png")
cropped_image = image.crop((left, upper, right, lower))

cropped_image.save("plswork_cropped.png")


fig1, ax1 = plt.subplots()
bg_img = plt.imread('plswork_cropped.png')
ax1.set_xlim(10, 0)
ax1.set_ylim(200, 0)
#ax1.set_aspect('auto')
ax1.imshow(bg_img, extent=[10, 0, 200, 0], aspect='auto')
x = [1, 20, 30, 40, 50]
y = [2, 30, 40, 50, 60]
ax1.scatter(x, y, color='red')

fig1.savefig("parem_nime.png", dpi=300)