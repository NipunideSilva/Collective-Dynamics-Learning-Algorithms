# defines research-paper-quality defaults

from __future__ import annotations
import matplotlib.pyplot as plt


OKABE_ITO = [
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#000000",  # black
    "#715531",  # mocha
]


# def apply_paper_style() -> None:
#     """
#     Global matplotlib style for research-paper-quality figures.
#     """
#     plt.rcParams.update({
#         "figure.figsize": (7.5, 4.8),
#         "figure.dpi": 140,
#         "savefig.dpi": 300,
#         "savefig.bbox": "tight",

#         "font.family": "serif",
#         "font.size": 12,

#         "axes.labelsize": 14,
#         "axes.titlesize": 14,
#         "axes.titlepad": 10,
#         "axes.linewidth": 1.0,
#         "axes.grid": True,

#         "xtick.labelsize": 12,
#         "ytick.labelsize": 12,
#         "xtick.major.width": 1.0,
#         "ytick.major.width": 1.0,
#         "xtick.direction": "in",
#         "ytick.direction": "in",

#         "legend.fontsize": 11,
#         "legend.frameon": False,
#         "legend.edgecolor": "black",

#         "grid.linestyle": "--",
#         "grid.alpha": 0.45,
#         "grid.linewidth": 0.5,

#         "lines.linewidth": 2.0,
#         "lines.markersize": 5,

#         "mathtext.fontset": "cm",
#     })

def apply_paper_style() -> None:
    """
    Global matplotlib style for research-paper-quality figures.
    """
    plt.rcParams.update({
        # Figure
        "figure.figsize": (7.5, 4.8),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Fonts
        "font.family": "serif",
        "font.size": 13,
        "font.weight": "normal",

        # Axes
        "axes.labelsize": 16,
        "axes.labelweight": "bold",

        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,

        "axes.linewidth": 1.2,
        "axes.grid": True,

        # Ticks
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,

        "xtick.major.size": 5,
        "ytick.major.size": 5,

        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,

        "xtick.direction": "in",
        "ytick.direction": "in",

        # Legend
        "legend.fontsize": 15,
        "legend.frameon": True,
        "legend.framealpha": 0.98,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.handlelength": 2.5,
        "legend.handletextpad": 0.8,
        "legend.borderpad": 0.5,

        # Grid
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,

        # Lines
        "lines.linewidth": 2.2,
        "lines.markersize": 6,

        # Math
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        # "mathtext.fontset": "cm",
    })