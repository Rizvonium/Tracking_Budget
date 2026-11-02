import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
for f in os.listdir(FD):
    if f.endswith(".ttf"): fm.fontManager.addfont(os.path.join(FD, f))
plt.rcParams.update({
    "font.family": "Inter", "font.size": 10, "axes.edgecolor": "#c9c7c0",
    "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold",
    "axes.titlesize": 12, "axes.titlecolor": "#0b0b0b", "text.color": "#0b0b0b",
    "figure.dpi": 100, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    "axes.grid": False, "grid.color": "#e6e4de", "grid.linewidth": 0.8,
})
# validated reference categorical palette (light)
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948")
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a887f", "#fcfcfb", "#e6e4de"
BRAND = dict(indigo="#1E2A78", tea="#2F7D4F", basalt="#3d3d3d", sand="#efe9dc", amber="#c98500")
BLUE_RAMP = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7","#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"]
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
def save(fig, name):
    fig.savefig(os.path.join(FIG, name), facecolor="white"); plt.close(fig); print("saved", name)
