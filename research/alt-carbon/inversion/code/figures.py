import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
OUT = sys.argv[2]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
res = json.load(open(f"{D}/results.json"))
scal = json.load(open(f"{D}/scaling.json"))
nx, ny, nz = np.load(f"{D}/mesh_shape.npy")
h = 25.0
x = (np.arange(nx) - nx / 2 + 0.5) * h
z = -(np.arange(nz)[::-1] + 0.5) * h  # origin CCN: top at 0


def section(m):
    M = m.reshape((nx, ny, nz), order="F")
    return M[:, ny // 2, :].T


def true_chi():
    t = np.zeros((nx, ny, nz))
    xc = x; yc = x; zc = -(nz - np.arange(nz) - 0.5) * h
    XX, YY, ZZ = np.meshgrid(xc, yc, zc, indexing="ij")
    A = (XX > -200) & (XX < -75) & (YY > -100) & (YY < 100) & (ZZ > -175) & (ZZ < -50)
    B = (XX > 75) & (XX < 175) & (YY > -100) & (YY < 100) & (ZZ > -150) & (ZZ < -50)
    t[A] = 0.05; t[B] = 0.03
    return t[:, ny // 2, :].T


zc = -(nz - np.arange(nz) - 0.5) * h
ext = [x[0] - h / 2, x[-1] + h / 2, zc[0] - h / 2, zc[-1] + h / 2]

# Fig 1: remanence test
panels = [("True susceptibility\n(body B also has reversed remanence, Q = 3)", true_chi()),
          ("Scalar inversion, no bounds", section(np.load(f"{D}/model_remanent_b1000_free.npy"))),
          ("Scalar inversion, χ ≥ 0", section(np.load(f"{D}/model_remanent_b1000_pos.npy"))),
          ("Control: no remanence, no bounds", section(np.load(f"{D}/model_induced_b1000_free.npy")))]
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
v = 0.05
for ax, (t, s) in zip(axs.ravel(), panels):
    im = ax.imshow(s, extent=ext, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_title(t, fontsize=9.5)
for ax in axs[:, 0]: ax.set_ylabel("Depth (m)")
for ax in axs[1]: ax.set_xlabel("Easting (m)")
fig.colorbar(im, ax=axs, shrink=0.8, label="Susceptibility (SI)")
fig.savefig(f"{OUT}/fig_remanence.png"); plt.close(fig)

# Fig 2: beta sweep
sw = sorted([r for k, r in res.items() if k.startswith("induced_b") and k.endswith("free")], key=lambda r: r["beta"])
b = np.array([r["beta"] for r in sw])
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
axs[0].loglog(b, [r["phi_train_per_datum"] for r in sw], "o-", color="#2a6f97")
axs[0].axhline(1, color="#c0392b", ls="--", lw=1); axs[0].text(b[0], 1.15, "target χ²/N = 1", color="#c0392b", fontsize=8)
axs[0].set_title("Misfit on training lines"); axs[0].set_ylabel("φ_d / N")
axs[1].semilogx(b, [r["rms_test_vs_clean_nT"] for r in sw], "o-", color="#2a6f97")
axs[1].set_title("Error on hold-out lines vs noise-free field"); axs[1].set_ylabel("RMS (nT)")
axs[2].semilogx(b, [r["model_rmse"] for r in sw], "o-", color="#2a6f97", label="model RMSE")
ax2 = axs[2].twinx(); ax2.semilogx(b, [r["frac_neg_cells"] * 100 for r in sw], "s--", color="#888", label="% cells χ < −0.002")
ax2.set_ylabel("% negative cells", color="#666"); ax2.spines["right"].set_visible(True)
axs[2].set_title("Model error and negative cells"); axs[2].set_ylabel("RMSE (SI)")
for a in axs: a.set_xlabel("β (regularisation weight)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_beta_sweep.png"); plt.close(fig)

# Fig 3: memory scaling
c = np.array([s["cells"] for s in scal]); d = np.array([s["data"] for s in scal])
mb = np.array([s["G_MB"] for s in scal]); sec = np.array([s["seconds"] for s in scal])
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
axs[0].loglog(c * d, mb, "o", color="#2a6f97", label="measured (float32)")
xx = np.logspace(5, 16, 50)
axs[0].loglog(xx, xx * 4 / 1e6, "-", color="#999", lw=1, label="N × M × 4 bytes")
for lab, nd, ncell in [("This test", 576, 6912), ("10 km block\n50 m cells", 2.0e5, 3.2e6),
                       ("100 km region\n50 m cells", 4.0e6, 1.6e8)]:
    axs[0].plot(nd * ncell, nd * ncell * 4 / 1e6, "D", color="#c0392b")
    axs[0].annotate(lab, (nd * ncell, nd * ncell * 4 / 1e6), textcoords="offset points", xytext=(6, -14), fontsize=7.5)
for yv, lab in [(1.6e4, "16 GB laptop"), (1e9, "1 PB")]:
    axs[0].axhline(yv, color="#c0392b", ls=":", lw=0.8); axs[0].text(2e5, yv * 1.4, lab, fontsize=7.5, color="#c0392b")
axs[0].set_xlabel("data × cells"); axs[0].set_ylabel("Dense sensitivity size (MB)"); axs[0].legend(fontsize=8, loc="lower right")
axs[0].set_title("Why memory tricks matter")
axs[1].loglog(c * d, sec, "o-", color="#2a6f97")
axs[1].set_xlabel("data × cells"); axs[1].set_ylabel("Seconds to build G (4 CPU)"); axs[1].set_title("Build time")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scaling.png"); plt.close(fig)

# Fig 4: regional error
keys = [("induced_b1000_free", "Correct regional"), ("induced_offset-25_free", "−25 nT offset left in"),
        ("induced_trend_free", "0.1 nT/m trend left in")]
fig, axs = plt.subplots(1, 3, figsize=(12, 2.9), sharey=True)
for ax, (k, t) in zip(axs, keys):
    im = ax.imshow(section(np.load(f"{D}/model_{k}.npy")), extent=ext, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    r = res[k]
    ax.set_title(f"{t}\nmin χ = {r['min_chi']:.3f}, {100*r['frac_neg_cells']:.0f}% cells < −0.002", fontsize=9)
    ax.set_xlabel("Easting (m)")
axs[0].set_ylabel("Depth (m)")
fig.colorbar(im, ax=axs, shrink=0.9, label="χ (SI)")
fig.savefig(f"{OUT}/fig_regional.png"); plt.close(fig)
print("ok")
