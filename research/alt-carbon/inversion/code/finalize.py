import sys, pathlib
D = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
head = (D / "page_head.html").read_text()
body = (D / "body.html").read_text()
mast = """<header class="mast">
<div class="eyebrow">Alt Carbon × Ashoka University · Inversion assignment</div>
<h1>Joint and probabilistic inversion: first report on Tasks 1–4</h1>
<p>A response to Shrey Agarwal's email of 3 November 2026: a literature ranking, a critique protocol for the paper we were sent with synthetic stress tests, a review of Finnish datasets and ideas from other fields, and formulations for adding a third physical property.</p>
<p>Rizwan Mir, PhD scholar, Ashoka University · Supervisor: Prof. Aalok Thakkar · 9 November 2026 · Draft v1</p>
<div class="field">B₀ = 52 000 nT · I = 75° · D = 10° · synthetic tests: SimPEG 0.26, 6 912 cells, 576 TMI data, seed 42</div>
</header>"""
page = head + '\n<div class="wrap">\n' + mast + "\n" + body + "\n</div>\n"
(OUT / "inversion_report.html").write_text(page)
full = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + \
       head + "</head><body>" + page[len(head):] + "</body></html>"
(D / "full.html").write_text(full)

print("html ok")
