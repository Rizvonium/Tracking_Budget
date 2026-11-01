import sys, os, json, numpy as np
sys.path.insert(0, os.getcwd())
from common import *
from scipy import stats
rows = json.load(open("rs/dated_rows.json"))
labels = {"hill_farm": "Hill farmland & tea\n(crop/grass/shrub, 500–2300 m)", "hill_tree": "Hill tree cover incl. shaded tea\n(500–2300 m)", "plains_crop": "Plains cropland incl. rice\n(<300 m)"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
summary = {}
for ax, k in zip(axes, labels):
    pts = [(r["year"], r[f"{k}_IN"] - r[f"{k}_NP"]) for r in rows if r[f"{k}_IN"] is not None and r[f"{k}_NP"] is not None]
    yrs = np.array([p[0] for p in pts]); d = np.array([p[1] for p in pts])
    pre, post = d[yrs <= 2023], d[yrs >= 2024]
    t, p = stats.ttest_ind(post, pre, equal_var=False); u = stats.mannwhitneyu(post, pre).pvalue
    # placebo: pretend rollout in 2021 using only pre-period
    pp = stats.ttest_ind(d[(yrs >= 2021) & (yrs <= 2023)], d[yrs <= 2020], equal_var=False).pvalue
    summary[k] = (len(pre), len(post), pre.mean(), post.mean(), post.mean() - pre.mean(), p, u, pp)
    jit = (np.random.default_rng(1).random(len(yrs)) - 0.5) * 0.35
    ax.scatter(yrs + jit, d, s=22, color=[C["aqua"] if y >= 2024 else "#9a988f" for y in yrs], zorder=3, edgecolor="white", lw=0.6)
    ym = sorted(set(yrs)); ax.plot(ym, [d[yrs == y].mean() for y in ym], color=INK, lw=1.4, zorder=2)
    ax.axhline(pre.mean(), color="#9a988f", lw=1, ls="--"); ax.axhline(0, color="#c9c7c0", lw=0.8)
    ax.axvspan(2023.5, 2025.5, color=C["orange"], alpha=0.08)
    ax.set_title(labels[k], loc="left", fontsize=9.5); ax.set_xticks(ym); ax.set_xticklabels([str(y)[2:] for y in ym], fontsize=8); ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.text(0.02, 0.04, f"post − pre = {post.mean()-pre.mean():+.3f}\nWelch p = {p:.2f} · placebo p = {pp:.2f}", transform=ax.transAxes, fontsize=8, color=INK2, bbox=dict(fc="white", ec="none", alpha=.8))
axes[0].set_ylabel("NDVI difference, India − Nepal\n(same overpass date)")
fig.text(0.01, -0.07, "Each dot = one Sentinel-2 overpass date (Oct–Dec) with both sides imaged. Grey = before rollout (2018–23), green = 2024–25. Dashed = pre-rollout mean.\nPlacebo = same test with a fake 2021 rollout, using pre-rollout years only; a real effect should beat the placebo.", fontsize=7.2, color=MUTED)
save(fig, "rs_paired.png")
for k, v in summary.items(): print(k, "n_pre=%d n_post=%d pre=%+.4f post=%+.4f diff=%+.4f welch_p=%.3f mwu_p=%.3f placebo_p=%.3f" % v)
# seam check map
YRS = range(2018, 2026); N = {y: np.load(f"rs/dated_ndvi_{y}.npy") for y in YRS}
dem, wc, cty = np.load("rs/dem.npy"), np.load("rs/wc.npy"), np.load("rs/cty.npy")
anom = np.nanmean([N[2024], N[2025]], axis=0) - np.nanmean([N[y] for y in range(2018, 2024)], axis=0)
anom[~np.isin(wc, [10, 20, 30, 40])] = np.nan
x0, y1, RES, W, H = np.load("rs/grid.npy"); ext = (x0, x0 + W * RES, y1 - H * RES, y1)
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from pyproj import Transformer
cm = LinearSegmentedColormap.from_list("d", ["#b4521f", "#eb6834", "#f4c2a6", "#f2f1ed", "#a6dcc6", "#1baf7a", "#0d6b49"])
fig, ax = plt.subplots(figsize=(9, 7.8))
im = ax.imshow(anom, extent=ext, cmap=cm, norm=TwoSlopeNorm(0, -0.12, 0.12), interpolation="nearest")
ax.contour(np.flipud(cty), levels=[1.5], extent=ext, colors="#333", linewidths=1.2, origin="lower")
tr = Transformer.from_crs(4326, 32645, always_xy=True)
for n, (lo, la) in {"Darjeeling": (88.26, 27.04), "Kurseong": (88.28, 26.88), "Mirik": (88.18, 26.89), "Gayabari (near Selim Hill pilot)": (88.312, 26.859), "Bagdogra (D-CAL)": (88.32, 26.70), "Siliguri": (88.43, 26.71), "Ilam (Nepal)": (87.92, 26.91)}.items():
    x, y = tr.transform(lo, la); ax.plot(x, y, "o", ms=6, mfc="white", mec=INK); ax.text(x + 900, y + 600, n, fontsize=8.5, weight="semibold", bbox=dict(boxstyle="round,pad=0.15", fc=(1,1,1,.8), ec="none"))
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("NDVI change: 2024–25 mean minus 2018–23 mean", fontsize=8.5); cb.outline.set_visible(False)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Vegetation change after the rollout (date-matched composites)", loc="left", fontsize=11)
fig.text(0.02, 0.06, "Line = India–Nepal border. Vegetated pixels only. A basalt effect would show as Indian-side patches with estate-shaped edges;\nregion-wide shifts on both sides of the border are weather or season-timing effects.", fontsize=7.4, color=MUTED)
save(fig, "rs_anomaly_map2.png")
