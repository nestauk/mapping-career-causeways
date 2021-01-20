import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import seaborn as sns
import  mapping_career_causeways

useful_paths = mapping_career_causeways.Paths()

# Colours for visualisations
colour_pal = ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
             '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
             '#008080', '#e6beff', '#9a6324', '#800000', '#808000','#ffe119',
             '#aaffc3', '#000000', '#ffd8b1', '#808000', '#000075', '#DCDCDC']

colour_map = {
    'Low risk': np.array([52, 146, 235, 255])/255, # blue
    'Other': np.array([191, 191, 191, 255])/255, # gray
    'High risk': np.array([222, 51, 9, 255])/255, # red
}

def set_font_sizes(SMALL_SIZE = 13, MEDIUM_SIZE=14):
    """ Sets up font sizes for plots """
    plt.rc('font', size=SMALL_SIZE, family='Arial')
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
set_font_sizes()

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    Reference: http://chris35wills.github.io/matplotlib_diverging_colorbar/
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
    """ Utility function for easy exports of figures """

    export_params = {'dpi': 200, 'bbox_inches': 'tight', 'transparent': True}

    if png:
        plt.savefig(f'{figure_folder}{figure_name}.png', **export_params)
    if svg:
        plt.savefig(f'{figure_folder}svg/{figure_name}.svg', **export_params)

def fix_heatmaps(ax):
    """
    Fix for mpl bug that cuts off top/bottom of seaborn viz
    Reference: https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    """
    b, t = ax.get_ylim()
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_ylim(b, t)

def remove_right_axis(ax):
    for x in ['right', 'top']:
        right_side = ax.spines[x]
        right_side.set_visible(False)

def plot_heatmap(mat, x_labels, y_labels=None, cmap=None,
                 figsize=(10,10), fix_heatmap=False, limits = (None, None),
                 annot=True, shorten_xlabel=True,
                 new_order=None,
                 include_rows=None, include_cols=None):

    """
    Utility function for easy plotting of heatmaps and, if necessary,
    reordering of rows and columns (used for displaying transition matrices between sectors)
    """

    f, ax = plt.subplots(figsize=figsize)

    if y_labels is None:
        y_labels = x_labels

    # Re-order columns
    if type(new_order) != type(None):
        mat = mat[new_order,:]
        mat = mat[:, new_order]
        x_labels = np.array(x_labels)[new_order]
        y_labels = np.array(y_labels)[new_order]

        map_old_to_new_order = dict(zip(new_order, range(len(new_order))))
        if type(include_rows) != type(None):
            include_rows = [map_old_to_new_order[x] for x in include_rows]
        if type(include_cols) != type(None):
            include_cols = [map_old_to_new_order[x] for x in include_cols]

    # Select a subsection of the matrix
    if type(include_rows) != type(None):
        mat = mat[include_rows,:]
        y_labels = y_labels[include_rows]
    if type(include_cols) != type(None):
        mat = mat[:, include_cols]
        x_labels = x_labels[include_cols]

    if type(cmap) == type(None):
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

    ax = sns.heatmap(
        mat,
        annot=annot,
        cmap=cmap,
        vmin = limits[0],
        vmax = limits[1],
        cbar_kws={"shrink": 0.5},
        center=0, square=True, linewidths=.1)

    if fix_heatmap:
        fix_heatmaps(ax)

    if shorten_xlabel == True:
        x_labels = [x.split(' ')[0]+'..' for x in x_labels]
    plt.yticks(ticks=np.array(list(range(len(y_labels))))+0.5, labels=y_labels, rotation=0)
    plt.xticks(ticks=np.array(list(range(len(x_labels))))+0.5, labels=x_labels, rotation=90)
    # ax.tick_params(axis='both', which='major', labelsize=9)

    return ax
