"""Delivered cost of CO2-capturing feedstocks in India, per tonne of CO2 (theoretical and realistic).
All inputs are explicit assumptions (INR/t of material) — edit and re-run."""
import csv, os, sys
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "."); from common import *  # common.py = shared plot style from the dossier scripts
import numpy as np
FX = 88.0          # INR per USD (assumption, Nov 2026)
TRUCK = 2.0        # INR per tonne-km (Fretron benchmark 1.8–2.8; rail avg ~1.9)
KM = 300           # haul distance assumed for all feedstocks (Alt Carbon: Pakur→Darjeeling ~350 km)
SPREAD = 200       # INR/t spreading & handling (assumption)
# name: material INR/t (low, mid, high), grinding INR/t, theoretical t CO2 per t (solid-carbonate route), realistic fraction reacting in ~5 yr (low, high), notes
F = {
 "Basalt quarry fines (Alt Carbon's choice)": ((400, 700, 1300), 0,   0.25, (0.10, 0.40), "already fine; widely available; slow"),
 "Steel slag (BF/SMS)":                        ((150, 300, 500),  400, 0.35, (0.30, 0.70), "fast-reacting; 19+ Mt/yr; heavy-metal & regulatory checks"),
 "Dunite / olivine rock":                      ((800, 1500, 2500), 800, 0.50, (0.20, 0.50), "high capacity; Ni/Cr risk; limited deposits"),
 "Wollastonite":                               ((900, 1500, 2500), 800, 0.38, (0.40, 0.80), "fast; India is a producer (Rajasthan); competes with industrial uses"),
 "Fly ash (Class F)":                          ((0, 600, 1000),   0,   0.04, (0.30, 0.70), "abundant but almost no Ca/Mg in Indian Class F ash"),
}
rows = []
for n, (mat, grind, cap, (fl, fh), note) in F.items():
    deliv = [m + grind + TRUCK * KM + SPREAD for m in mat]
    theo = [d / cap / FX for d in deliv]                      # USD per t CO2 if fully reacted
    real = (deliv[1] / (cap * fh) / FX, deliv[1] / (cap * fl) / FX)
    rows.append(dict(feedstock=n, material_inr=f"{mat[0]}–{mat[2]}", grind_inr=grind, delivered_inr_mid=deliv[1], t_co2_per_t=cap,
                     usd_per_tco2_theoretical_low=round(theo[0]), usd_per_tco2_theoretical_mid=round(theo[1]), usd_per_tco2_theoretical_high=round(theo[2]),
                     usd_per_tco2_realistic_best=round(real[0]), usd_per_tco2_realistic_worst=round(real[1]), notes=note))
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['feedstock']:44s} delivered ₹{r['delivered_inr_mid']:>5} /t  cap {r['t_co2_per_t']:.2f}  theoretical ${r['usd_per_tco2_theoretical_low']}–{r['usd_per_tco2_theoretical_high']} (mid {r['usd_per_tco2_theoretical_mid']})  realistic ${r['usd_per_tco2_realistic_best']}–{r['usd_per_tco2_realistic_worst']}")
# chart
order = sorted(rows, key=lambda r: r["usd_per_tco2_realistic_best"])
fig, ax = plt.subplots(figsize=(11, 4.6)); y = np.arange(len(order))[::-1]
for yi, r in zip(y, order):
    ax.plot([r["usd_per_tco2_theoretical_low"], r["usd_per_tco2_theoretical_high"]], [yi + 0.15] * 2, color="#9ec5f4", lw=7, solid_capstyle="round")
    ax.plot([r["usd_per_tco2_realistic_best"], min(r["usd_per_tco2_realistic_worst"], 1400)], [yi - 0.15] * 2, color=C["blue"], lw=7, solid_capstyle="round")
    ax.text(min(r["usd_per_tco2_realistic_worst"], 1400) + 15, yi - 0.15, f"${r['usd_per_tco2_realistic_best']}–{r['usd_per_tco2_realistic_worst']:,}" + (" (off scale)" if r["usd_per_tco2_realistic_worst"] > 1400 else ""), va="center", fontsize=8.3, color=INK2)
ax.axvline(270, color=C["orange"], ls="--", lw=1.3); ax.text(285, 3.5, "Alt Carbon's credit price ≈ $270/t\n(includes MRV, labour, margin)", fontsize=8, color=C["orange"], va="center")
ax.set_yticks(y); ax.set_yticklabels([r["feedstock"] for r in order], fontsize=9); ax.set_xlim(0, 1550)
ax.set_xlabel("USD per tonne of CO₂ (material + grinding + 300 km truck + spreading only)"); ax.xaxis.grid(True); ax.set_axisbelow(True)
ax.plot([], [], color="#9ec5f4", lw=7, label="if fully reacted (theoretical, low–high price)"); ax.plot([], [], color=C["blue"], lw=7, label="realistic fraction reacting in ~5 yr (mid price)")
ax.legend(frameon=False, fontsize=8.5, loc="upper right"); ax.set_title("Feedstock cost per tonne of CO₂ captured, India", loc="left", fontsize=12)
fig.text(0.01, -0.06, "Prices: IndiaMART/TradeIndia listings 2025–26, IBM average sale prices (dunite, wollastonite; inflated from 2012–16), CemNet (slag). Freight ₹2/t-km (Fretron; rail avg ₹1.9). ₹88/$.\nCapacities: solid-carbonate ceiling (MatterSim run + mineral chemistry). Reacting fractions are assumptions. Excludes MRV, land access, margins. Quicklime excluded: making it emits the CO₂ it later absorbs.", fontsize=7.2, color=MUTED)
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedstock_costs.png"), dpi=200, bbox_inches="tight", facecolor="white")
