"""Shared publication-sized overrides for style.mplstyle."""

# Scale the shared stylesheet for single-column figures and a 7.1-inch panel array.
PAPER_STYLE = {"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
               "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .9,
               "xtick.major.size": 3.5, "ytick.major.size": 3.5,
               "xtick.major.width": .8, "ytick.major.width": .8,
               "xtick.minor.size": 2, "ytick.minor.size": 2,
               "xtick.minor.width": .5, "ytick.minor.width": .5,
               "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42}
