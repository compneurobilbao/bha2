import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list
import numpy as np
from scipy.stats import pearsonr


def plot_hline_dend(position, text, c="gray", s=":", w=2):
    plt.axhline(y=position, color=c, ls=s, lw=w)
    plt.annotate(
        text,
        (0, position),
        textcoords="offset points",
        xytext=(-6, 0),
        ha="right",
        color="k",
    )


def plot_vline_dend(x, height_bf, height_af, text, c="k", s=":", w=3):
    plt.axvline(x=x, ymin=height_af, ymax=height_bf, color=c, ls=s, lw=w)
    plt.annotate(
        text,
        (x, (height_bf + height_af) / 2),
        textcoords="offset points",
        xytext=(2, 0),
        ha="left",
        color="k",
    )


def plot_module_size_dend(Z, module_rois, den, c="black", s=":", w=3):
    leaves = leaves_list(Z)
    loc_in_tree = np.where(np.in1d(leaves, module_rois))[0]
    x1 = den["icoord"][loc_in_tree[0]][0]
    x2 = den["icoord"][loc_in_tree[-1]][0]
    # draw a horizontal line in the bottom of the dendrogram from the first to the last leaf of the module
    plt.axhline(
        y=0,
        xmin=loc_in_tree[0] / len(leaves),
        xmax=loc_in_tree[-1] / len(leaves),
        color=c,
        ls=s,
        lw=w,
    )
    plt.annotate(
        "Module size",
        ((x1 + x2) / 2, 0),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        rotation=0,
    )


def plot_corr(x, y, lx=0.85, ly=0.15, hue=None, ax=None, fsize=14, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, p = pearsonr(x, y)
    ax = ax or plt.gca()
    plt.annotate(
        f"r = {r:.2f}",
        xy=(lx, ly + 0.1),
        xycoords=ax.transAxes,
        ha="center",
        fontsize=fsize,
    )
    ax.annotate(
        f"p = {p:.2f}",
        xy=(lx, ly),
        xycoords=ax.transAxes,
        ha="center",
        fontsize=fsize,
    )
