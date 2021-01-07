import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import  mapping_career_causeways

useful_paths = mapping_career_causeways.Paths()

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    Reference:
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def export_figure(figure_name, figure_folder = useful_paths.figure_dir, png=True, svg=True):

    export_params = {'dpi': 200, 'bbox_inches': 'tight', 'transparent': True}

    if png:
        plt.savefig(f'{figure_folder}{figure_name}.png', **export_params)
    if svg:
        plt.savefig(f'{figure_folder}svg/{figure_name}.svg', **export_params)

def fix_heatmaps(ax):
    """
    Fix for mpl bug that cuts off top/bottom of seaborn viz
    Reference:
    """
    b, t = ax.get_ylim()
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_ylim(b, t)
