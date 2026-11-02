# Screening CO2-capturing minerals with Microsoft MatterSim (automated pipeline)

Run 2 Nov 2026, CPU only, about 25 min.

## What the pipeline does
1. `fetch.py`: downloads experimental crystal structures from the Crystallography Open Database.
   It keeps only ambient-pressure phases, verifies each file's composition, and records the COD IDs in
   `structures/sources.json`.
2. `run_experiments.py`: relaxes every crystal (atoms and cell) with MatterSim v1.0.0, using the 1M and 5M
   models. It computes carbonation reaction energies per mol CO2, compares them with experimental
   enthalpies, and writes `results/results_<run>.json` and `results/log_<run>.txt`.
3. `report.py`: generates the figures in `figures/`.

Re-run:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install mattersim && pip uninstall -y torchvision
python fetch.py && python run_experiments.py && python report.py figures/mattersim
```

## Experiments and results
| Exp. | Question | Result |
|---|---|---|
| A | Does MatterSim reproduce experimental carbonation enthalpies? | Yes. 12 reactions, mean absolute error **7.1 kJ/mol (5M model)** and 9.6 (1M). The bigger model is better. |
| B | Do we need to relax structures? | Yes. Single-point energies at experimental geometry give MAE 24 kJ/mol. Fe-bearing fayalite/siderite is off by 124 kJ/mol until relaxed. |
| C | Does a constant CO2-molecule correction (fitted on CaO and MgO) help? | Inconsistently. It improved 1M (9.1 → 7.7) but worsened 5M (7.4 → 8.5). The errors are not a simple CO2 offset. |
| D | What is the CO2 ceiling of a typical basalt? | About 280 kg CO2/t for an idealised mix (50 % anorthite, 35 % pyroxene, 10 % olivine; solid-carbonate route). |

## Rankings (MatterSim 5M, kJ per mol CO2; more negative = stronger driving force)
- lime −175
- larnite (steel slag) −122
- anorthite −114
- periclase −110
- portlandite −108
- wollastonite −94
- pyroxenes −86 to −88
- forsterite −84
- fayalite −76
- brucite −70

Capacity per tonne:
- periclase 1,092 kg/t
- lime 785
- forsterite 626
- portlandite 594
- larnite 511
- pyroxenes 406–438
- wollastonite 379
- plagioclase (anorthite) 158

## What it means for enhanced weathering
- **Every basalt mineral carbonates spontaneously.** Thermodynamics is not the bottleneck. Field weathering is
  limited by **kinetics**: dissolution rates, water, surface area. DFT and ML potentials alone cannot capture this.
- **Industrial wastes are stronger and denser sinks than basalt.** Lime, steel slag (larnite) and portlandite
  in cement and concrete waste beat basalt. India produces large volumes of steel slag, so this deserves a look.
- **Alt Carbon's "1 t CO2 per 4 t basalt" (250 kg/t) is close to this idealised ceiling (~280 kg/t).**
  It implies near-complete reaction of the rock. That is a theoretical potential, not a field rate.
  (The bicarbonate route stores up to 2 CO2 per Ca/Mg, but natural plagioclase contains Na and less Ca
  than end-member anorthite, so the true ceiling depends on the actual rock chemistry.)

## Caveats
- These are 0 K static energies, compared with 298 K enthalpies; vibrational and thermal terms are ignored.
- Fe magnetism is handled only implicitly. Siderite expands 28 % on relaxation, a known weak spot.
  Brucite expands 16 % because layered hydroxides need van der Waals corrections.
- The H2O(l) energy is approximated as the gas-phase energy minus 44 kJ/mol.
- Experimental ΔH values come from standard formation enthalpies (Robie & Hemingway-type tables). Verify them before publishing.
- Validate final numbers with real DFT (Quantum ESPRESSO / GPAW, PBEsol or r2SCAN) on a university cluster.

## Next experiments
1. Mineral surfaces: water and CO2 adsorption energies on forsterite (010), diopside and labradorite slabs
   (rough proxies for dissolution).
2. Real feedstocks: use XRD/XRF of Rajmahal and Panjal basalt instead of end-member minerals.
3. Na/K-bearing plagioclase (albite, labradorite) and basaltic glass, which dominate real basalt.
4. Steel slag phases (larnite, merwinite, brownmillerite) for an Indian industrial-waste pathway.
5. MatterGen: ask the generative model for new silicates with high CO2 driving force.
