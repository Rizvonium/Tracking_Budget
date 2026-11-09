# Proposed timeline and deliverables

| Week | Deliverable | Depends on |
|---|---|---|
| 1 | Hakku access. Parameter checklist of the paper. Literature database (Zotero) with this report's references verified. | Paper PDF, Hakku login |
| 2 | Task 1 v2: ranking table checked against the papers, with 1-page summaries of the top 10 papers. Task 3 dataset inventory for the exact study area (coverage maps, QC statistics). | Study area coordinates |
| 3–4 | Task 2: dense reference vs the paper's method on one tile, a full reproduction, and stress tests E1–E4 on real data. Technical note. | Data download |
| 5–8 | Task 3: 2–3 improvements tested on the same data (for example a sampled lithology model and joint TV / parallel level sets). Whitepaper draft. | Task 2 baseline |
| 9+ | Task 4 with the Alt Carbon team: a 2D prototype of the latent-lithology formulation with DC/IP, then 3D. | Permitted description of the current method |

# Method and reproducibility

- **Synthetic tests:** SimPEG 0.26.0, discretize 0.13.0 and choclo on Python 3, on a 4-CPU and 15 GB cloud machine. The random seed is 42. The code is in `research/alt-carbon/inversion/` (`experiments.py`, `figures.py`). The results are in `results.json` and `scaling.json`.
- **Literature:** The references come from web searches by AI research agents. Each reference has a DOI or arXiv ID that appeared in a search result. Items tagged UNVERIFIED need a manual check before citation.
- **Not done:** We did not read the full text of every paper. The Elicit research tool was not connected to this session. A systematic search in Elicit, Scopus or Web of Science is the next step for Task 1.
