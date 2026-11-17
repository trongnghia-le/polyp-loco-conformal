"""Generate Fig. 1: study design, conformal calibration pipeline and evaluation (schematic).

Outputs: results/figures/fig_pipeline.{pdf,eps,png}
Sci Rep figure rules applied: one typeface and one size (7 pt, configs/paper.mplstyle) for all lettering,
sentence-case labels, no thousands separator below five digits, lines >= 1 pt. Card heights follow their
content; the script refuses to save if any text runs outside its card (checked after drawing).
Numbers: docs/data_audit.md (8037/3121/4916 frames), data/clusters.csv (1272), revision_R5_detection_floor.csv
(floors), revision2_R6a_weighted.csv (pooled > alpha at 5/6 frame, 4/6 video), revision_R4_degenerate_fa.csv
(cannot certify 4-37%), revision2_alarm.csv (47-82%), docs/notes/2026-11-08_clinicdb-b1.md (ClinicDB).
"""
from __future__ import annotations

import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
OUT = ROOT / "results" / "figures"

FS = 7.0            # the one font size used in every paper figure
STEP = 3.25         # mm between text lines at 7 pt
HDR = 5.6           # header strip height, mm
PAD = 1.6           # inner padding, mm
LW = 1.15           # box edge width, pt (>= 1 pt)

BLUE, BLUE_BG = "#2a78d6", "#f1f6fc"
PURPLE, PURPLE_BG = "#6250d6", "#f4f2fc"
ORANGE, ORANGE_BG = "#eb6834", "#fdf4ef"
TEAL, TEAL_BG = "#138f60", "#eff9f5"
AMBER, AMBER_BG = "#c98300", "#fdf8ec"
SLATE, SLATE_BG = "#2d3748", "#f4f6f8"
CYAN, CYAN_BG = "#0e7490", "#eef6f9"
MUTED = "#52514e"

CHECKS: list[tuple[mpl.text.Text, float, float]] = []   # (text, x_left, x_right) bounds in mm


def _box(ax, x, y, w, h, fc, ec, lw=LW, r=1.6, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def _text(ax, x, y, s, bounds, color=MUTED, bold=False, **kw):
    t = ax.text(x, y, s, color=color, fontsize=FS, fontweight="bold" if bold else "normal", zorder=4, **kw)
    CHECKS.append((t, *bounds))
    return t


def _arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8.5, linewidth=lw,
                                 color=color, zorder=5, shrinkA=0, shrinkB=0))


def card_height(lines, extra=0.0):
    return HDR + PAD + len(lines) * STEP + PAD + extra


def card(ax, x, y_top, w, h, title, lines, color, bg, extra=0.0):
    """Card hanging from y_top with a coloured header strip; lines = (text, bold, colour)."""
    y = y_top - h
    _box(ax, x, y, w, h, bg, color)
    ax.add_patch(FancyBboxPatch((x, y_top - HDR), w, HDR, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=color, edgecolor="none", zorder=2.5))
    ax.add_patch(Rectangle((x, y_top - HDR), w, HDR * 0.45, facecolor=color, edgecolor="none", zorder=2.6))
    _text(ax, x + PAD, y_top - HDR / 2, title, (x, x + w), color="white", bold=True, va="center", ha="left")
    for i, (s, bold, col) in enumerate(lines):
        _text(ax, x + PAD, y_top - HDR - PAD - extra - i * STEP, s, (x, x + w), color=col, bold=bold,
              va="top", ha="left")
    return y


def build_figure() -> plt.Figure:
    W = 183.0
    cols = [(1.5, 40.0), (46.5, 40.5), (92.5, 43.5), (141.0, 40.5)]
    stages = [("1  Audit and video clusters", BLUE), ("2  LOCO folds and detector", PURPLE),
              ("3  Calibration (α = 0.10)", SLATE), ("4  Evaluation and cost", SLATE)]
    top = [
        ("PolypGen v3 audit", BLUE, BLUE_BG, [
            ("8037 frames, centres C1–C6", True, SLATE),
            ("3121 positive, 4916 negative", False, MUTED),
            ("Audit decisions:", True, SLATE),
            ("• seq5 reassigned to C2", False, MUTED),
            ("• seq11 confirmed as C1", False, MUTED),
            ("• One official box = one polyp", False, MUTED)]),
        ("LOCO folds and training", PURPLE, PURPLE_BG, [
            ("Hold out one whole centre", True, SLATE),
            ("Source videos split by role:", False, MUTED),
            ("70% train, 10% val, 20% cal", True, PURPLE),
            ("YOLO11s, 3 seeds × 6 folds", True, SLATE),
            ("YOLO11m size check", False, MUTED),
            ("No target frame in training", False, MUTED)]),
        ("Source-centre transfer", ORANGE, ORANGE_BG, [
            ("λ from the 5 source centres:", True, SLATE),
            ("• Pooled (frames or videos)", False, MUTED),
            ("• Centre-balanced weights", False, MUTED),
            ("• Per-source, capped and", False, MUTED),
            ("   leave-one-out variants", False, MUTED),
            ("Fixed λ on whole target centre", True, ORANGE)]),
        ("Estimands and uncertainty", SLATE, SLATE_BG, [
            ("Two miss-rate scales:", True, SLATE),
            ("• Frame-averaged", False, MUTED),
            ("• Video-averaged", False, MUTED),
            ("Target-video bootstrap and", False, MUTED),
            ("block bootstrap of singles", False, MUTED),
            ("Pooled > α at 5 of 6 (frame)", True, ORANGE),
            ("and 4 of 6 (video) centres", True, ORANGE)]),
    ]
    PILL_ROW = STEP + 2.2   # room for the outcome pills in the last bottom card
    bottom = [
        ("Video clusters", BLUE, BLUE_BG, [
            ("1272 clusters (videos)", True, SLATE),
            ("From folders, file names", False, MUTED),
            ("and near-duplicate frames", False, MUTED),
            ("No video spans two roles", True, SLATE)], 0.0),
        ("Box export and miss floor", PURPLE, PURPLE_BG, [
            ("All boxes, confidence ≥ 0.001", True, SLATE),
            ("Hit: one-to-one, IoU ≥ 0.1", False, MUTED),
            ("Miss floor, IoU ≥ 0.1:", True, SLATE),
            ("frame 1.5–10.5%", True, PURPLE),
            ("video 1.5–5.4%", True, PURPLE),
            ("IoU ≥ 0.5: 2.7–24.2%", False, MUTED)], 0.0),
        ("Target recalibration", TEAL, TEAL_BG, [
            ("k = 5–40 target videos", True, SLATE),
            ("500 draws per k:", False, MUTED),
            ("• Video-level CRC, all frames", False, MUTED),
            ("• One frame per video", False, MUTED),
            ("• Frame-level CRC, all frames", False, MUTED),
            ("λ tested on undrawn videos", True, TEAL)], 0.0),
        ("Certification and alarms", AMBER, AMBER_BG, [
            ("k = 20, one frame per video:", True, SLATE),
            ("no threshold certified in 4–37%", False, MUTED),
            ("Alarms on 47–82% of", True, SLATE),
            ("polyp-free frames (C1, C4–C6)", False, MUTED)], PILL_ROW),
    ]

    pill_h, gap_pill, gap_rows, band_h = 5.0, 3.0, 6.0, 14.0
    h_top = max(card_height(c[3]) for c in top)
    h_bot = max(card_height(c[3], c[4]) for c in bottom)
    H = 1.5 + pill_h + gap_pill + h_top + gap_rows + h_bot + gap_rows + band_h + 1.5
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    y_pill = H - 1.5 - pill_h
    for (x, w), (label, col) in zip(cols, stages):
        _box(ax, x, y_pill, w, pill_h, "white", col, lw=1.05, r=1.4, z=3)
        _text(ax, x + w / 2, y_pill + pill_h / 2, label, (x, x + w), color=col, bold=True, ha="center", va="center")

    y_top1 = y_pill - gap_pill
    y_top2 = y_top1 - h_top - gap_rows
    for (x, w), (title, col, bg, lines) in zip(cols, top):
        card(ax, x, y_top1, w, h_top, title, lines, col, bg)
    for (x, w), (title, col, bg, lines, extra) in zip(cols, bottom):
        card(ax, x, y_top2, w, h_bot, title, lines, col, bg, extra)

    # outcome pills inside the last bottom card
    x4, w4 = cols[3]
    py = y_top2 - HDR - PAD - 4.2
    px = x4 + PAD
    for label, fc, ec, tc in (("Held", "#dcfce7", "#15803d", "#166534"), ("Violated", "#fee2e2", "#b91c1c", "#991b1b"),
                              ("No cert.", "#fef3c7", "#b45309", "#92400e")):
        t = ax.text(0, 0, label, fontsize=FS, fontweight="bold")
        fig.canvas.draw()
        tw = t.get_window_extent().width / fig.dpi * 25.4
        t.remove()
        _box(ax, px, py, tw + 2.0, 4.2, fc, ec, lw=1.0, r=1.0, z=3)
        _text(ax, px + (tw + 2.0) / 2, py + 2.1, label, (x4, x4 + w4), color=tc, bold=True, ha="center", va="center")
        px += tw + 3.0

    # arrows: within columns (top -> bottom card) and across columns
    ym1, ym2 = y_top1 - h_top / 2, y_top2 - h_bot / 2
    for (x, w), col in ((cols[0], BLUE), (cols[1], PURPLE)):
        _arrow(ax, x + w / 2, y_top1 - h_top, x + w / 2, y_top2, color=col)
    (x1, w1), (x2, w2), (x3, w3), (x4, w4) = cols
    _arrow(ax, x1 + w1, ym1, x2, ym1)
    _arrow(ax, x1 + w1, ym2, x2, ym2)
    fork = x2 + w2 + 2.4                                   # exported scores feed both calibration branches
    ax.plot([x2 + w2, fork], [ym2, ym2], color=MUTED, lw=1.3, zorder=4)
    ax.plot([fork, fork], [ym2, ym1], color=MUTED, lw=1.3, zorder=4)
    _arrow(ax, fork, ym1, x3, ym1, color=ORANGE)
    _arrow(ax, fork, ym2, x3, ym2, color=TEAL)
    _arrow(ax, x3 + w3, ym1, x4, ym1, color=ORANGE)
    _arrow(ax, x3 + w3, ym2, x4, ym2, color=TEAL)

    # external check band
    by = 1.5
    _box(ax, x1, by, W - 3.0, band_h, CYAN_BG, CYAN)
    _box(ax, x1, by, 35.0, band_h, CYAN, CYAN, lw=1.0, z=2.5)
    ax.add_patch(Rectangle((x1 + 31.0, by), 4.0, band_h, facecolor=CYAN, edgecolor="none", zorder=2.6))
    _text(ax, x1 + 17.5, by + band_h / 2, "External check\nCVC-ClinicDB", (x1, x1 + 35.0), color="white", bold=True,
          ha="center", va="center", linespacing=1.3)
    band = (x1 + 35.0, x1 + W - 3.0)
    for i, (s, bold, col) in enumerate((
            ("612 positive frames in 29 sequences; all 18 PolypGen-trained YOLO11s detectors, no retraining", True, SLATE),
            ("• Frame versus whole-video splits within ClinicDB (about 310 calibration frames)", False, MUTED),
            ("• Pooled PolypGen threshold versus recalibration on one frame from each of 20 videos", False, MUTED))):
        _text(ax, band[0] + 2.0, by + band_h - 2.6 - i * 4.0, s, band, color=col, bold=bold, ha="left", va="center")
    for x, w in (cols[1], cols[2]):
        ax.plot([x + w / 2] * 2, [y_top2 - h_bot, by + band_h + 1.5], ls="--", lw=1.1, color=CYAN, zorder=1)
        _arrow(ax, x + w / 2, by + band_h + 2.0, x + w / 2, by + band_h, color=CYAN, lw=1.1)

    # refuse to save if any text leaves its card (0.6 mm inner margin)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for t, xl, xr in CHECKS:
        bb = t.get_window_extent()
        (l, _), (r, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if l < xl + 0.6 or r > xr - 0.6:
            bad.append(f"{t.get_text()!r}: {l:.1f}-{r:.1f} mm, card {xl:.1f}-{xr:.1f}")
    assert not bad, "text outside its card:\n" + "\n".join(bad)
    sizes = {t.get_fontsize() for t, *_ in CHECKS}
    assert sizes == {FS}, sizes
    return fig, H


if __name__ == "__main__":
    fig, H = build_figure()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        fig.savefig(OUT / f"fig_pipeline.{ext}", facecolor="white", **({"dpi": 600} if ext == "png" else {}))
    print(f"Saved fig_pipeline.{{pdf,eps,png}}: 183 x {H:.1f} mm, all text {FS} pt")
