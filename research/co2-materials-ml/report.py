import sys, os, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
H = os.path.dirname(os.path.abspath(__file__)); OUT = sys.argv[1]
f = sorted(glob.glob(f"{H}/results/results_*.json"))[-1]; R = json.load(open(f))
M1, M5 = "MatterSim-v1.0.0-1M.pth", "MatterSim-v1.0.0-5M.pth"; r1, r5 = R["models"][M1]["reactions"], R["models"][M5]["reactions"]
names = [k for k in r5 if k in r1]
# Fig 1: parity
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(wspace=0.3))
for mod, rx, col, lab in [(M1, r1, "#9a988f", "MatterSim 1M"), (M5, r5, C["blue"], "MatterSim 5M")]:
    x = [rx[k]["dH_exp"] for k in names]; y = [rx[k]["dE_relaxed"] for k in names]
    mae = np.mean(np.abs(np.array(x) - y)); ax[0].scatter(x, y, s=46, color=col, edgecolor="white", label=f"{lab}  (MAE {mae:.1f} kJ/mol)", zorder=3)
lim = [-190, -40]; ax[0].plot(lim, lim, color=INK, lw=1); ax[0].fill_between(lim, [l - 10 for l in lim], [l + 10 for l in lim], color="#e6e4de", alpha=0.6, label="±10 kJ/mol")
for k in names:
    short = k.split(" →")[0].split(" (")[0]; ax[0].annotate(short, (r5[k]["dH_exp"], r5[k]["dE_relaxed"]), (4, -9), textcoords="offset points", fontsize=7.5, color=INK2)
ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_xlabel("Experimental ΔH (kJ per mol CO₂)"); ax[0].set_ylabel("MatterSim ΔE (kJ per mol CO₂)")
ax[0].set_title("Exp. A: does the AI model reproduce real thermodynamics?", loc="left", fontsize=11); ax[0].legend(fontsize=8, frameon=False, loc="upper left"); ax[0].grid(True, color="#eeece6")
xr = np.arange(len(names)); w = 0.38
eu = [abs(r5[k]["dE_unrelaxed"] - r5[k]["dH_exp"]) for k in names]; er = [abs(r5[k]["dE_relaxed"] - r5[k]["dH_exp"]) for k in names]
ax[1].barh(xr + w/2, eu, height=w, color="#f4c2a6", label=f"experimental geometry (MAE {np.mean(eu):.0f})"); ax[1].barh(xr - w/2, er, height=w, color=C["blue"], label=f"after AI relaxation (MAE {np.mean(er):.0f})")
ax[1].set_yticks(xr); ax[1].set_yticklabels([k.replace(" → ", "→").replace(" (", "\n(") for k in names], fontsize=7.2); ax[1].invert_yaxis()
ax[1].set_xlabel("absolute error vs experiment (kJ/mol)"); ax[1].set_title("Exp. B: relaxing structures matters", loc="left", fontsize=11); ax[1].legend(fontsize=8, frameon=False, loc="lower right"); ax[1].xaxis.grid(True)
save(fig, OUT + "_validation.png")
# Fig 2: ranking
order = sorted(names, key=lambda k: r5[k]["dE_relaxed"])
fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(wspace=0.65))
cat = lambda k: C["orange"] if any(s in k for s in ["lime", "periclase", "portlandite", "brucite", "larnite"]) else C["aqua"]
y = np.arange(len(order))
ax[0].barh(y, [-r5[k]["dE_relaxed"] for k in order], color=[cat(k) for k in order], height=0.62)
for yi, k in zip(y, order): ax[0].text(-r5[k]["dE_relaxed"] + 2, yi, f"{r5[k]['dE_relaxed']:.0f}", va="center", fontsize=8, color=INK2)
ax[0].set_yticks(y); ax[0].set_yticklabels([k.replace(" → ", " → ") for k in order], fontsize=7.8); ax[0].invert_yaxis()
ax[0].set_xlabel("Driving force −ΔE (kJ per mol CO₂, MatterSim 5M)"); ax[0].set_title("How strongly each mineral 'wants' CO₂", loc="left", fontsize=11); ax[0].xaxis.grid(True)
ax[1].barh(y, [r5[k]["kgCO2_per_t"] for k in order], color=[cat(k) for k in order], height=0.62)
for yi, k in zip(y, order): ax[1].text(r5[k]["kgCO2_per_t"] + 10, yi, f"{r5[k]['kgCO2_per_t']:.0f}", va="center", fontsize=8, color=INK2)
ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis(); ax[1].set_xlabel("Max CO₂ captured (kg per tonne of mineral)")
ax[1].set_title("…and how much it can hold", loc="left", fontsize=11); ax[1].xaxis.grid(True)
ax[1].text(0.98, 0.97, "orange = oxides/hydroxides/slag (industrial wastes)\ngreen = rock-forming silicates (basalt minerals)", transform=ax[1].transAxes, ha="right", va="top", fontsize=7.8, color=INK2)
save(fig, OUT + "_ranking.png")
# Fig 3: basalt
b = R.get("basalt", {}); c = b.get("contribution_kg_per_t", {})
if c:
    fig, ax = plt.subplots(figsize=(10, 2.9)); left = 0; cols = [C["blue"], C["aqua"], C["green"], C["orange"], C["yellow"]]
    for (m, v), col in zip(c.items(), cols):
        ax.barh(0, v, left=left, color=col, height=0.5, edgecolor="white", lw=2); ax.text(left + v/2, 0, (f"{m}\n{v:.0f}" if v > 25 else f"{m[:3]}.\n{v:.0f}"), ha="center", va="center", fontsize=8, color="white" if v > 30 else INK)
        left += v
    ax.barh(-0.75, 250, color="#9a988f", height=0.4); ax.text(255, -0.75, "Alt Carbon's claim: 1 t CO₂ per 4 t basalt = 250 kg/t", va="center", fontsize=8.5, color=INK2)
    ax.text(left + 30, 0, f"total ≈ {left:.0f} kg CO₂ per t basalt (theoretical max)", va="center", fontsize=9, weight="bold")
    ax.set_yticks([]); ax.set_xlim(0, 520); ax.set_xlabel("kg CO₂ per tonne of rock"); ax.set_title("Exp. D: theoretical CO₂ ceiling of a typical basalt (50% plagioclase, 35% pyroxene, 10% olivine)", loc="left", fontsize=10.5)
    ax.spines["left"].set_visible(False); save(fig, OUT + "_basalt.png")
print("used", f)
