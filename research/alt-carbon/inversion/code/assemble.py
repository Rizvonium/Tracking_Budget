"""Assemble report markdown from parts, render HTML fragment (artifact) + full HTML (for PDF)."""
import base64, re, subprocess, sys, pathlib

D = pathlib.Path(sys.argv[1])
lines = lambda p: (D / p).read_text().splitlines()


def section(src, start_pat, end_pat=None):
    L = lines(src)
    s = next(i for i, l in enumerate(L) if re.match(start_pat, l))
    e = len(L) if end_pat is None else next(i for i, l in enumerate(L) if i > s and re.match(end_pat, l))
    return L[s:e]


def shift(L, delta):
    out = []
    in_code = False
    for l in L:
        if l.startswith("```"):
            in_code = not in_code
        m = re.match(r"^(#+) (.*)", l)
        if m and not in_code:
            n = max(1, min(6, len(m.group(1)) + delta))
            l = "#" * n + " " + m.group(2)
        out.append(l)
    return out


# Task 1
t1 = lines("task1_review.md")
first_h1 = next(i for i, l in enumerate(t1) if l.startswith("# "))
t1 = ["# Task 1: Literature review of joint and probabilistic inversion", ""] + t1[first_h1 + 1:]
t1 = [l for l in t1]
# demote any further H1 to H2
t1 = [("#" + l) if (l.startswith("# ") and i > 0) else l for i, l in enumerate(t1)]

# Task 3
inv = section("task2_3_paper_data.md", r"^## JOB 2", r"^## Sources")
inv = inv[1:]  # drop JOB 2 heading
inv = shift(inv, 1)
srcs = section("task2_3_paper_data.md", r"^## Sources")[1:]
partA = section("task3_4_ideas.md", r"^## Part A", r"^## Part B")[1:]
t3 = (["# Task 3: Dataset report and ideas for better inversion", "",
       "## 3.1 Datasets for the Finnish study area", ""] + inv +
      ["", "### Sources for the dataset inventory", ""] + srcs +
      ["", "## 3.2 Ideas from other inverse-problem fields", ""] + partA)

# Task 4
partB = section("task3_4_ideas.md", r"^## Part B", r"^## References")[1:]
partB = shift(partB, -1)
refs34 = section("task3_4_ideas.md", r"^## References")[1:]
t4 = (["# Task 4: Adding a third physical property", ""] + partB +
      ["", "## References for Tasks 3 and 4", ""] + refs34)

md = "\n".join(lines("r00_summary.md") + [""] + t1 + [""] + lines("r20_task2.md") + [""] +
               t3 + [""] + t4 + [""] + lines("r50_plan.md")) + "\n"
(D / "report.md").write_text(md)

(D / "tpl.html").write_text('<nav id="toc">\n<h2>Contents</h2>\n$table-of-contents$\n</nav>\n<main>\n$body$\n</main>\n')
frag = subprocess.run(["pandoc", str(D / "report.md"), "-f", "markdown+tex_math_dollars+pipe_tables",
                       "-t", "html5", "--mathml", "--toc", "--toc-depth=2", "-s",
                       "--template", str(D / "tpl.html"), "--resource-path", str(D)],
                      capture_output=True, text=True, check=True).stdout


def embed(m):
    p = D / m.group(1)
    b = base64.b64encode(p.read_bytes()).decode()
    return f'src="data:image/png;base64,{b}"'


frag = re.sub(r'src="(fig/[^"]+\.png)"', embed, frag)
frag = frag.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
(D / "body.html").write_text(frag)
print(len(md), len(frag))
