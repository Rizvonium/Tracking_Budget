"""Automated CO2-mineralisation screening with MatterSim.
Experiments: A) relaxed energies with two model sizes; B) unrelaxed (experimental geometry) vs relaxed;
C) empirical CO2 correction fitted on simple oxides; D) whole-basalt CO2 potential. All results logged to results/."""
import os, json, time, datetime, warnings, numpy as np
from collections import Counter
from ase.io import read
from ase.build import molecule
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from mattersim.forcefield import MatterSimCalculator
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__)); EV = 96.485
RUN = datetime.datetime.now().strftime("%Y%m%d-%H%M%S"); LOG = open(f"{H}/results/log_{RUN}.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
FU = {"forsterite": {"Mg": 2, "Si": 1, "O": 4}, "fayalite": {"Fe": 2, "Si": 1, "O": 4}, "enstatite": {"Mg": 1, "Si": 1, "O": 3},
      "diopside": {"Ca": 1, "Mg": 1, "Si": 2, "O": 6}, "wollastonite": {"Ca": 1, "Si": 1, "O": 3}, "larnite": {"Ca": 2, "Si": 1, "O": 4},
      "anorthite": {"Ca": 1, "Al": 2, "Si": 2, "O": 8}, "lime": {"Ca": 1, "O": 1}, "periclase": {"Mg": 1, "O": 1},
      "brucite": {"Mg": 1, "O": 2, "H": 2}, "portlandite": {"Ca": 1, "O": 2, "H": 2}, "quartz": {"Si": 1, "O": 2},
      "calcite": {"Ca": 1, "C": 1, "O": 3}, "magnesite": {"Mg": 1, "C": 1, "O": 3}, "siderite": {"Fe": 1, "C": 1, "O": 3},
      "dolomite": {"Ca": 1, "Mg": 1, "C": 2, "O": 6}, "kaolinite": {"Al": 2, "Si": 2, "O": 9, "H": 4}}
# Approximate standard enthalpies of formation at 298 K (kJ/mol), Robie & Hemingway-type tabulations; verify before publication.
DHF = {"CO2": -393.5, "H2O_l": -285.8, "forsterite": -2174.0, "fayalite": -1478.2, "enstatite": -1545.0, "diopside": -3202.3,
       "wollastonite": -1634.9, "larnite": -2307.5, "anorthite": -4227.9, "lime": -635.1, "periclase": -601.6, "brucite": -924.5,
       "portlandite": -985.2, "quartz": -910.7, "calcite": -1207.4, "magnesite": -1111.7, "siderite": -755.9, "dolomite": -2326.3, "kaolinite": -4119.6}
# reactions: (reactants, products), CO2 count taken from reactants
RX = {
 "lime → calcite":                 ({"lime": 1, "CO2": 1}, {"calcite": 1}),
 "periclase → magnesite":          ({"periclase": 1, "CO2": 1}, {"magnesite": 1}),
 "portlandite → calcite":          ({"portlandite": 1, "CO2": 1}, {"calcite": 1, "H2O_l": 1}),
 "brucite → magnesite":            ({"brucite": 1, "CO2": 1}, {"magnesite": 1, "H2O_l": 1}),
 "larnite (slag) → calcite":       ({"larnite": 1, "CO2": 2}, {"calcite": 2, "quartz": 1}),
 "wollastonite → calcite":         ({"wollastonite": 1, "CO2": 1}, {"calcite": 1, "quartz": 1}),
 "forsterite (olivine) → magnesite": ({"forsterite": 1, "CO2": 2}, {"magnesite": 2, "quartz": 1}),
 "fayalite (olivine) → siderite":  ({"fayalite": 1, "CO2": 2}, {"siderite": 2, "quartz": 1}),
 "enstatite (pyroxene) → magnesite": ({"enstatite": 1, "CO2": 1}, {"magnesite": 1, "quartz": 1}),
 "diopside (pyroxene) → calcite+magnesite": ({"diopside": 1, "CO2": 2}, {"calcite": 1, "magnesite": 1, "quartz": 2}),
 "diopside (pyroxene) → dolomite": ({"diopside": 1, "CO2": 2}, {"dolomite": 1, "quartz": 2}),
 "anorthite (plagioclase) → calcite+kaolinite": ({"anorthite": 1, "CO2": 1, "H2O_l": 2}, {"calcite": 1, "kaolinite": 1}),
}
MASS = {"H": 1.008, "C": 12.011, "O": 15.999, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "Ca": 40.078, "Fe": 55.845}
def nfu(atoms, fu):
    c = Counter(atoms.get_chemical_symbols()); k = [c[e] / n for e, n in fu.items()]
    if set(c) != set(fu) or max(k) - min(k) > 1e-6: return None, dict(c)
    return k[0], dict(c)
def relax(atoms, calc, fmax=0.02, steps=600, cell=True):
    atoms = atoms.copy(); atoms.calc = calc; t = time.time()
    e0 = atoms.get_potential_energy(); v0 = atoms.get_volume()
    opt = FIRE(FrechetCellFilter(atoms) if cell else atoms, logfile=None); conv = opt.run(fmax=fmax, steps=steps)
    return dict(E_unrelaxed=float(e0), E=float(atoms.get_potential_energy()), V0=float(v0), V=float(atoms.get_volume()),
                steps=int(opt.nsteps), converged=bool(conv), seconds=round(time.time() - t, 1))
def mol(name, calc):
    m = molecule(name); m.center(vacuum=8.0); m.pbc = True
    r = relax(m, calc, cell=False); return r
results = {"run": RUN, "models": {}, "sources": json.load(open(f"{H}/structures/sources.json"))}
for model in ["MatterSim-v1.0.0-1M.pth", "MatterSim-v1.0.0-5M.pth"]:
    log(f"\n=== model {model} ==="); calc = MatterSimCalculator(load_path=model, device="cpu"); E = {}; Eu = {}; meta = {}
    for name, fu in FU.items():
        a = read(f"{H}/structures/{name}.cif"); n, comp = nfu(a, fu)
        if n is None: log(f"  {name:13s} SKIP composition mismatch {comp}"); continue
        r = relax(a, calc); E[name] = r["E"] / n; Eu[name] = r["E_unrelaxed"] / n
        meta[name] = dict(r, atoms=len(a), fu=n, dV_pct=round(100 * (r["V"] / r["V0"] - 1), 2))
        log(f"  {name:13s} atoms={len(a):3d} fu={n:g} E/fu={E[name]:10.4f} eV  ΔV={meta[name]['dV_pct']:+6.2f}%  steps={r['steps']:3d} conv={r['converged']}  {r['seconds']}s")
    for m_ in ["CO2", "H2O"]:
        r = mol(m_, calc); E[m_] = Eu[m_] = r["E"]; meta[m_] = r; log(f"  {m_:13s} E={r['E']:.4f} eV")
    E["H2O_l"] = E["H2O"] - 44.0 / EV; Eu["H2O_l"] = E["H2O_l"]   # liquid ≈ gas − ΔH_vap (44 kJ/mol)
    rx = {}
    for rn, (re_, pr) in RX.items():
        if not all(k in E for k in list(re_) + list(pr)): continue
        nco2 = re_["CO2"]
        dE = (sum(v * E[k] for k, v in pr.items()) - sum(v * E[k] for k, v in re_.items())) * EV / nco2
        dEu = (sum(v * Eu[k] for k, v in pr.items()) - sum(v * Eu[k] for k, v in re_.items())) * EV / nco2
        dH = (sum(v * DHF[k] for k, v in pr.items()) - sum(v * DHF[k] for k, v in re_.items())) / nco2
        mineral = [k for k in re_ if k not in ("CO2", "H2O_l")][0]
        mw = sum(MASS[e] * n for e, n in FU[mineral].items())
        cap = nco2 * 44.01 / mw * 1000  # kg CO2 per tonne mineral
        rx[rn] = dict(dE_relaxed=round(dE, 1), dE_unrelaxed=round(dEu, 1), dH_exp=round(dH, 1), kgCO2_per_t=round(cap, 0), mineral=mineral)
    # Experiment C: constant per-CO2 correction fitted on lime + periclase only
    fit = [k for k in rx if k.startswith(("lime", "periclase"))]
    corr = float(np.mean([rx[k]["dH_exp"] - rx[k]["dE_relaxed"] for k in fit]))
    for k in rx: rx[k]["dE_corrected"] = round(rx[k]["dE_relaxed"] + corr, 1)
    test = [k for k in rx if k not in fit]
    mae_raw = np.mean([abs(rx[k]["dE_relaxed"] - rx[k]["dH_exp"]) for k in test]); mae_cor = np.mean([abs(rx[k]["dE_corrected"] - rx[k]["dH_exp"]) for k in test])
    log(f"\n  {'reaction':44s} {'ΔE relaxed':>11s} {'unrelaxed':>10s} {'corrected':>10s} {'expt ΔH':>8s} {'kgCO2/t':>8s}")
    for k, v in rx.items(): log(f"  {k:44s} {v['dE_relaxed']:11.1f} {v['dE_unrelaxed']:10.1f} {v['dE_corrected']:10.1f} {v['dH_exp']:8.1f} {v['kgCO2_per_t']:8.0f}")
    log(f"  CO2 correction (fit on lime+periclase) = {corr:+.1f} kJ/mol; MAE on {len(test)} held-out reactions: raw {mae_raw:.1f} → corrected {mae_cor:.1f} kJ/mol")
    results["models"][model] = dict(energies=E, meta=meta, reactions=rx, co2_correction=corr, mae_raw=mae_raw, mae_corrected=mae_cor)
json.dump(results, open(f"{H}/results/results_{RUN}.json", "w"), indent=1, default=float)
# Experiment D: typical tholeiitic basalt (illustrative modal mineralogy, wt %)
BASALT = {"anorthite": 0.50, "diopside": 0.25, "enstatite": 0.10, "forsterite": 0.07, "fayalite": 0.03}  # remainder: glass/oxides ignored
rx = results["models"]["MatterSim-v1.0.0-5M.pth"]["reactions"]
pick = {"anorthite": "anorthite (plagioclase) → calcite+kaolinite", "diopside": "diopside (pyroxene) → calcite+magnesite",
        "enstatite": "enstatite (pyroxene) → magnesite", "forsterite": "forsterite (olivine) → magnesite", "fayalite": "fayalite (olivine) → siderite"}
contrib = {m: w * rx[pick[m]]["kgCO2_per_t"] for m, w in BASALT.items() if pick[m] in rx}
results["basalt"] = dict(modes=BASALT, contribution_kg_per_t=contrib, total_kg_per_t=sum(contrib.values()))
log(f"\nExperiment D: illustrative basalt (97% of mass as 5 end-members) max CO2 uptake ≈ {sum(contrib.values()):.0f} kg CO2 per tonne → 1 t CO2 per {1000/sum(contrib.values()):.1f} t rock")
log("  (Fe-bearing siderite pathway is unlikely in oxidising soils; Alt Carbon claims 1 t CO2 : 4 t basalt)")
json.dump(results, open(f"{H}/results/results_{RUN}.json", "w"), indent=1, default=float)
log(f"\nSaved results/results_{RUN}.json")
