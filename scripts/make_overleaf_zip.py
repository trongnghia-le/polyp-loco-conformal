"""Package the manuscript for Overleaf with only what compilation needs.

Collects paper/main.tex (and paper/supplementary.tex if present), every \\input/\\include'd .tex file reachable from
them, the bibliography (.bib) and only the figure files actually referenced by \\includegraphics (PDF preferred, as
pdflatex uses it; EPS/PNG copies are left out). Figure paths are rewritten to figures/<name>.pdf so the zip is
self-contained. Then a test compile (pdflatex + bibtex, twice) is run on the packaged copy, and the zip is written
with forward-slash entry names: outputs/overleaf/polyp-paper-overleaf.zip.  Excludes: BRIEF.md, README.md,
citation_check.md, build files, EPS/PNG, anything not referenced.
usage: python -I scripts/make_overleaf_zip.py [--no-compile]"""
import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "outputs" / "overleaf"
STAGE = OUT / "stage"

ap = argparse.ArgumentParser()
ap.add_argument("--no-compile", action="store_true")
a = ap.parse_args()

if STAGE.exists():
    shutil.rmtree(STAGE)
(STAGE / "figures").mkdir(parents=True)

INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPH = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})")
BIB = re.compile(r"\\bibliography\{([^}]+)\}")
strip_comments = lambda s: re.sub(r"(?<!\\)%.*", "", s)


def resolve_fig(ref: str, base: Path) -> Path:
    """Find the referenced figure; prefer a PDF next to it or in results/figures."""
    p = Path(ref)
    cands = [(base / p), (ROOT / "results" / "figures" / p.name)]
    for c in cands:
        for ext in ([c.suffix] if c.suffix else []) + [".pdf", ".png", ".jpg"]:
            q = c.with_suffix(ext) if ext else c
            if q.suffix.lower() == ".pdf" and q.exists():
                return q
    for c in cands:
        for ext in (".png", ".jpg", ".jpeg"):
            if c.with_suffix(ext).exists():
                return c.with_suffix(ext)
    sys.exit(f"MISSING FIGURE: {ref}")


copied, figs, bibs = [], {}, set()


def process(tex: Path, rel: Path):
    src = tex.read_text(encoding="utf-8")
    live = strip_comments(src)

    def fig_sub(m):
        f = resolve_fig(m.group(2), tex.parent)
        figs[f.name] = f
        return f"{m.group(1)}figures/{f.name}{m.group(3)}"

    out = GRAPH.sub(fig_sub, src)
    dst = STAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(out, encoding="utf-8")
    copied.append(rel.as_posix())
    for b in BIB.findall(live):
        for name in b.split(","):
            bibs.add(name.strip())
    for inc in INPUT.findall(live):
        r = Path(inc if inc.endswith(".tex") else inc + ".tex")
        f = (tex.parent / r) if (tex.parent / r).exists() else (PAPER / r)
        if not f.exists():
            sys.exit(f"MISSING INPUT: {inc} (from {rel})")
        process(f, f.relative_to(PAPER))


roots = [p for p in (PAPER / "main.tex", PAPER / "supplementary.tex") if p.exists()]
if not roots:
    sys.exit("paper/main.tex not found")
for r in roots:
    process(r, r.relative_to(PAPER))
for b in sorted(bibs):
    f = PAPER / (b if b.endswith(".bib") else b + ".bib")
    if not f.exists():
        sys.exit(f"MISSING BIB: {f}")
    shutil.copy2(f, STAGE / f.name); copied.append(f.name)
for name, f in figs.items():
    shutil.copy2(f, STAGE / "figures" / name); copied.append(f"figures/{name}")
for f in sorted(PAPER.glob("*.cls")) + sorted(PAPER.glob("*.bst")):   # journal template class and reference style
    shutil.copy2(f, STAGE / f.name); copied.append(f.name)

if not a.no_compile and shutil.which("pdflatex"):
    for r in roots:
        stem = r.stem
        for cmd in (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"],
                    ["bibtex", stem], ["pdflatex", "-interaction=nonstopmode", stem + ".tex"],
                    ["pdflatex", "-interaction=nonstopmode", stem + ".tex"]):
            if cmd[0] == "bibtex" and not bibs:
                continue
            res = subprocess.run(cmd, cwd=STAGE, capture_output=True, text=True, errors="replace")
            if res.returncode != 0 and cmd[0] == "pdflatex" and "-halt-on-error" in cmd:
                print(res.stdout[-3000:]); sys.exit(f"COMPILE FAILED: {stem}")
        log = (STAGE / f"{stem}.log").read_text(errors="replace")
        undef = sorted(set(re.findall(r"Citation `([^']+)' .*undefined", log)))
        refs = len(re.findall(r"Reference `[^']+' .*undefined", log))
        pages = re.search(r"Output written on .*?\((\d+) pages", log)
        print(f"{stem}.pdf: {pages.group(1) if pages else '?'} pages; undefined citations: {undef or 'none'}; "
              f"undefined refs: {refs}")
        shutil.copy2(STAGE / f"{stem}.pdf", OUT / f"{stem}.pdf")

OUT.mkdir(parents=True, exist_ok=True)
zp = OUT / "polyp-paper-overleaf.zip"
zp.unlink(missing_ok=True)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in sorted(set(copied)):
        z.write(STAGE / rel, rel)
names = zipfile.ZipFile(zp).namelist()
assert not any("\\" in n for n in names)
print(f"{zp}  ({zp.stat().st_size / 1e6:.2f} MB, {len(names)} files)")
for n in names:
    print("  ", n)
