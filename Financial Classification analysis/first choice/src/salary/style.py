"""Shared matplotlib style for the report figures."""
import matplotlib.pyplot as plt

COLORS = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "ink": "#0b0b0b", "muted": "#52514e", "grid": "#d9d8d4", "surface": "#fcfcfb",
}


def setup():
    plt.rcParams.update({
        "figure.facecolor": COLORS["surface"], "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"], "axes.labelcolor": COLORS["muted"],
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.6,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": COLORS["ink"],
        "font.size": 10, "legend.fontsize": 9, "figure.dpi": 100,
    })


def save(fig, path):
    """Save a figure to reports/ (the caller decides when to close it)."""
    fig.tight_layout()
    fig.savefig(path, dpi=200)
