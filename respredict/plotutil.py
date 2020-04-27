import numpy as np
from rdkit.Chem import Draw
from rdkit import Chem
import py3Dmol
from matplotlib.collections import LineCollection
import PIL
import cairosvg
import io

def draw_vlines(xpoints, ymin=0, ymax=1, ax=None, **kwargs):
    ls = [[(x, ymin), (x,ymax)] for x in xpoints]
    if ax is None:
        ax = pylab.gca()
    
    line_segments = LineCollection(ls, **kwargs)
    ax.add_collection(line_segments)
    return line_segments



def jupyter_draw_conf(m,p=None,confId=-1, style='stick', extrastyle={}):
    """
    remember to
    from rdkit.Chem.Draw import IPythonConsole
    """
    mb = Chem.MolToMolBlock(m,confId=confId)

    if p is None:
        p = py3Dmol.view(width=400,height=400)
    p.removeAllModels()
    p.addModel(mb,'sdf')
    p.setStyle({style:extrastyle})
    p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    return p.show()



def mol_to_image(mol, w, h, embed_2D = True, **kwargs):
    mol = Chem.Mol(mol)
    if embed_2D:
        Chem.AllChem.Compute2DCoords(mol)

    svgdraw = Chem.Draw.MolDraw2DSVG(w, h)
    svgdraw.DrawMolecule(mol,**kwargs)
    svgdraw.FinishDrawing()
    svg_text = svgdraw.GetDrawingText()

    png_bytes = cairosvg.svg2png(bytestring=svg_text)

    mol_img = PIL.Image.open(io.BytesIO(png_bytes))
    return mol_img


def plot_err_vs_ppm(tgt_df, nuc='13C', max_shift=None, 
                    max_shift_scale = 7.0):


    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1],  )

    gs.update(hspace=0.05)
    
    if max_shift is None:
        if nuc == '13C' :
            max_shift = 200.0
        else:
            max_shift = 9.0

    ppm_bins = np.linspace(0, max_shift, 52)
    tgt_df['bin'] = pd.cut(tgt_df.value, bins=ppm_bins, labels=False)

    a = tgt_df.groupby('bin').agg({'delta' : ['mean', 'std', 'count', 
                                              'min', 'max', 'quantile']}).reset_index()
    y_min = a[('delta', 'min')]
    y_max = a[('delta', 'max')]
    y_std = a[('delta', 'std')]
    y_mean = a[('delta', 'mean')]
    y_count = a[('delta', 'count')]

    fig = pylab.figure()
    ax = fig.add_subplot(gs[0])
    #print(ppm_bins.shape, y_mean.shape)
    #print(y_mean)
    x_label = y_mean.index.values
    ax.plot(ppm_bins[x_label], y_mean, label='error mean')

    ax.fill_between(ppm_bins[x_label], y_mean - y_std, y_mean + y_std, alpha=0.2, label='error std')

    ax.plot(ppm_bins[x_label], y_min, c='k', linewidth=0.5, label='error range')
    ax.plot(ppm_bins[x_label], np.array(y_max), c='k', linewidth=0.5, label=None)

    #tgt_df.bin.value_counts()
    ax2 = fig.add_subplot(gs[1])

    ax2.plot(ppm_bins[x_label], y_count/np.sum(y_count), c='r')
    ax2.set_ylabel('fraction\n of data', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_ylim(0, 0.12)
    ax2.set_yticks([0.0, 0.05, 0.1])

    ax2.set_xlabel("true shift (ppm)")

    ax.axhline(0, c='k', zorder=-1, linewidth=0.5, alpha=0.5, label=None)
    #pylab.xlim(0, max_shift)
    
    ax.set_ylim(-max_shift*max_shift_scale, max_shift*max_shift_scale)
    if nuc == '13C':
        ax.set_title("$^{{13}}$C chemical shift prediction")
        
    else:
        ax.set_title("$^{{1}}$H chemical shift prediction")
        
    ax.set_ylabel("error (ppm)")
    ax.set_xticks([])
    ax.legend(loc='lower left', ncol=3)

    fig.tight_layout()

    #fig.savefig(f"model_validate.{checkpoint_i}.pdf")

    return fig, ax, ax2
