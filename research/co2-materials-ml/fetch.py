"""Fetch reference crystal structures from the Crystallography Open Database (COD)."""
import json, re, urllib.request, os, sys, time
def get(u):
    for i in range(5):
        try: return urllib.request.urlopen(u, timeout=90).read()
        except Exception as e: print("  retry", i, e); time.sleep(3 * (i + 1))
    raise RuntimeError(u)
from collections import Counter
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "structures")
TARGETS = {  # name: (search text, reduced formula {el: n})
 "forsterite": ("forsterite", {"Mg": 2, "Si": 1, "O": 4}), "fayalite": ("fayalite", {"Fe": 2, "Si": 1, "O": 4}),
 "enstatite": ("enstatite", {"Mg": 1, "Si": 1, "O": 3}), "diopside": ("diopside", {"Ca": 1, "Mg": 1, "Si": 2, "O": 6}),
 "wollastonite": ("wollastonite", {"Ca": 1, "Si": 1, "O": 3}), "larnite": ("larnite", {"Ca": 2, "Si": 1, "O": 4}),
 "anorthite": ("anorthite", {"Ca": 1, "Al": 2, "Si": 2, "O": 8}), "lime": ("lime", {"Ca": 1, "O": 1}),
 "periclase": ("periclase", {"Mg": 1, "O": 1}), "brucite": ("brucite", {"Mg": 1, "O": 2, "H": 2}),
 "portlandite": ("portlandite", {"Ca": 1, "O": 2, "H": 2}), "quartz": ("quartz", {"Si": 1, "O": 2}),
 "calcite": ("calcite", {"Ca": 1, "C": 1, "O": 3}), "magnesite": ("magnesite", {"Mg": 1, "C": 1, "O": 3}),
 "siderite": ("siderite", {"Fe": 1, "C": 1, "O": 3}), "dolomite": ("dolomite", {"Ca": 1, "Mg": 1, "C": 2, "O": 6}),
 "kaolinite": ("kaolinite", {"Al": 2, "Si": 2, "O": 9, "H": 4}),
}
def parse(f):
    c = Counter()
    for tok in f.strip("- ").split():
        m = re.match(r"([A-Z][a-z]?)([\d.]*)", tok); c[m.group(1)] += float(m.group(2) or 1)
    return c
def ratio_ok(c, want):
    if set(c) != set(want): return False
    k = c[next(iter(want))] / want[next(iter(want))]
    return all(abs(c[e] - k * n) < 1e-6 for e, n in want.items())
SG = {"forsterite": ["Pbnm", "Pnma"], "fayalite": ["Pbnm", "Pnma"], "enstatite": ["Pbca"], "diopside": ["C12/c1", "C2/c"],
      "wollastonite": ["P-1"], "larnite": ["P121/n1", "P21/n", "P121/c1"], "anorthite": ["P-1", "I-1"], "lime": ["Fm-3m"], "periclase": ["Fm-3m"],
      "brucite": ["P-3m1"], "portlandite": ["P-3m1"], "quartz": ["P3121", "P3221"], "calcite": ["R-3c", "R-3c:H", "R-3cRS"], "magnesite": ["R-3c:H", "R-3c"],
      "siderite": ["R-3c:H", "R-3c"], "dolomite": ["R-3:H", "R-3"], "kaolinite": ["C1", "P1"]}
norm = lambda g: (g or "").replace(" ", "")
log = []
for name, (q, want) in TARGETS.items():
    d = json.loads(get(f"https://www.crystallography.net/cod/result?text={q}&format=json"))
    cand = [x for x in d if ratio_ok(parse(x.get("formula", "")), want) and q in (x.get("mineral") or x.get("commonname") or x.get("title") or "").lower()]
    if not cand: cand = [x for x in d if ratio_ok(parse(x.get("formula", "")), want)]
    # prefer ambient-condition, smallest cell, recent refinement
    amb = [x for x in cand if norm(x.get("sg")) in SG[name]]
    cand = amb or cand
    # ambient phase, then median-volume entry (avoids high-P / high-T outliers)
    cand.sort(key=lambda x: float(x.get("vol") or 0)); cand = cand[len(cand)//2:] + cand[:len(cand)//2]
    if not cand: print("NO MATCH", name); log.append({"name": name, "cod": None}); continue
    from ase.io import read as aread
    x = None
    for c in cand:
        cif = get(f"https://www.crystallography.net/cod/{c['file']}.cif"); p = f"{OUT}/{name}.cif"; open(p, "wb").write(cif)
        try:
            cc = Counter(aread(p).get_chemical_symbols())
            if ratio_ok(Counter({k: float(v) for k, v in cc.items()}), want): x = c; break
            print(f"  {name}: COD {c['file']} parsed as {dict(cc)} - trying next")
        except Exception as e: print(f"  {name}: COD {c['file']} unreadable ({str(e)[:50]})")
    if x is None: print("NO VALID STRUCTURE", name); log.append({"name": name, "cod": None}); os.remove(p); continue
    log.append({"name": name, "cod_id": x["file"], "formula": x["formula"], "sg": x.get("sg"), "a": x.get("a"), "vol": x.get("vol"), "year": x.get("year"), "n_candidates": len(cand)})
    print(f"{name:13s} COD {x['file']}  {x['formula']:<22s} {x.get('sg')}  ({len(cand)} candidates)")
json.dump(log, open(f"{OUT}/sources.json", "w"), indent=1)
