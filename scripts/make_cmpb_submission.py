"""Build the Elsevier (Editorial Manager) LaTeX submission from paper/main.tex.

Editorial Manager flattens uploads into one directory, so the package is: one manuscript.tex with every \\input
inlined and the compiled bibliography (.bbl) pasted in place of \\bibliography, figures renamed Figure_<n>.pdf in
order of appearance, plus the compiled manuscript PDF and the supplementary PDF. The flattened source is compiled in
an empty directory (pdflatex x2) and must give the same page count as the regular build with no errors or warnings.
Run scripts/make_overleaf_zip.py first (it compiles paper/ and leaves main.bbl in outputs/overleaf/stage).
usage: python -I scripts/make_cmpb_submission.py   -> outputs/cmpb_submission/
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER, STAGE = ROOT / "paper", ROOT / "outputs" / "overleaf" / "stage"
OUT = ROOT / "outputs" / "cmpb_submission"
SRC = OUT / "latex_source"
INPUT = re.compile(r"\\input\{([^}]+)\}")


def inline(tex: str) -> str:
    def rep(m):
        f = PAPER / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return inline(f.read_text(encoding="utf-8")).rstrip("\n")
    return INPUT.sub(rep, tex)   # a function replacement inserts the text literally (no backslash processing)


def main():
    bbl = STAGE / "main.bbl"
    if not bbl.exists():
        sys.exit("run scripts/make_overleaf_zip.py first (no outputs/overleaf/stage/main.bbl)")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.iterdir():                     # regenerate everything except the Word files just built by
        if f.suffix.lower() != ".docx":         # paper/build_submission_docs.py (run that first)
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    SRC.mkdir()

    tex = inline((PAPER / "main.tex").read_text(encoding="utf-8"))
    tex = re.sub(r"^\\graphicspath\{.*\}\s*$\n?", "", tex, flags=re.M)
    figs = []

    def fig(m):
        name = Path(m.group(2)).stem
        if name not in figs:
            figs.append(name)
        return f"{m.group(1)}Figure_{figs.index(name) + 1}.pdf{m.group(3)}"
    tex = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})", fig, tex)
    n_bib = len(re.findall(r"^\\bibliography\{[^}]*\}\s*$", tex, flags=re.M))
    assert n_bib == 1, n_bib
    tex = re.sub(r"^\\bibliographystyle\{[^}]*\}\s*$\n?", "", tex, flags=re.M)
    bbl_text = bbl.read_text(encoding="utf-8")
    tex = re.sub(r"^\\bibliography\{[^}]*\}\s*$", lambda m: bbl_text.strip(), tex, flags=re.M)
    assert "\\input{" not in tex and "\\bibliography{" not in tex
    (SRC / "manuscript.tex").write_text(tex, encoding="utf-8")
    for i, name in enumerate(figs, 1):
        for ext in (".pdf", ".eps"):
            f = ROOT / "results" / "figures" / (name + ext)
            if f.exists():
                shutil.copy2(f, (SRC if ext == ".pdf" else OUT) / f"Figure_{i}{ext}")

    # compile the flattened source in isolation
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "manuscript.tex"],
                           cwd=SRC, capture_output=True, text=True, errors="replace")
        if r.returncode:
            print(r.stdout[-2500:]); sys.exit("flattened source does not compile")
    log = (SRC / "manuscript.log").read_text(errors="replace")
    pages = int(re.search(r"Output written on .*?\((\d+) pages", log).group(1))
    ref_pages = int(re.search(r"Output written on .*?\((\d+) pages", (STAGE / "main.log").read_text(errors="replace")).group(1))
    warn = re.findall(r"^(?:! .*|LaTeX Warning.*|Overfull.*|Package \S+ Warning.*)$", log, flags=re.M)
    assert pages == ref_pages, (pages, ref_pages)
    assert not warn, warn[:5]
    shutil.copy2(SRC / "manuscript.pdf", OUT / "Manuscript.pdf")
    shutil.copy2(STAGE / "supplementary.pdf", OUT / "Supplementary_material.pdf")
    for f in SRC.glob("manuscript.*"):
        if f.suffix not in (".tex",):
            f.unlink()

    zp = OUT / "CMPB_LaTeX_source.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SRC.iterdir()):
            z.write(f, f.name)
    print(f"manuscript.tex: {pages} pages (regular build {ref_pages}), 0 warnings; figures: "
          + ", ".join(f"Figure_{i}={n}" for i, n in enumerate(figs, 1)))
    print(zp, f"{zp.stat().st_size / 1e6:.2f} MB:", [n for n in zipfile.ZipFile(zp).namelist()])

    # one package with every upload item, in the order Editorial Manager asks for them
    items = [("Cover_letter_CMPB.docx", "1_Cover_letter_CMPB.docx"), ("Title_page.docx", "2_Title_page.docx"),
             ("Highlights.docx", "3_Highlights.docx"), ("Manuscript.pdf", "4_Manuscript.pdf"),
             ("Supplementary_material.pdf", "7_Supplementary_material.pdf")]
    pkg = OUT / "CMPB_submission_package.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in items:
            if (OUT / src).exists():
                z.write(OUT / src, arc)
            else:
                print("NOTE: missing", src, "(run paper/build_submission_docs.py for the Word files)")
        for f in sorted(SRC.iterdir()):
            z.write(f, f"5_LaTeX_source/{f.name}")
        for f in sorted(OUT.glob("Figure_*.eps")) + sorted(SRC.glob("Figure_*.pdf")):
            z.write(f, f"6_Figures/{f.name}")
    print(pkg, f"{pkg.stat().st_size / 1e6:.2f} MB")
    for n in zipfile.ZipFile(pkg).namelist():
        print("  ", n)
    # leave only the package: every loose file is inside it and is rebuilt on the next run
    for f in OUT.iterdir():
        if f != pkg:
            shutil.rmtree(f) if f.is_dir() else f.unlink()


if __name__ == "__main__":
    main()
