"""Fig: target recalibration with the labelling budget counted in VIDEOS (clusters), IoU >= 0.1.
Columns = held-out centres. Row a: alpha = 0.10, mean video-averaged target miss rate minus alpha vs number of
labelled positive target videos k. Row b: alpha = 0.10, share of calibrations that could not certify a threshold
(degenerate: every box kept), which pulls the means in row a down at small k. Row c: as row a at alpha = 0.20.
Two designs: cluster-level CRC on all frames of k videos (video budget) and image-level CRC on one random positive
frame per video. The frame-budget design is no longer shown here (it controls neither estimand; Supplement).
Reads outputs/eval/conformal_cluster_summary.csv; writes results/figures/fig_cluster_budget.{pdf,eps,png}.
Points = mean over 500 draws x 3 seeds; error bars = range of the per-seed means (row b: of the per-seed shares).
Budgets whose k was capped (C4, k = 40 -> 24 so that ten videos remain for testing) are not drawn."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
s = pd.read_csv(ROOT / "outputs/eval/conformal_cluster_summary.csv")
s = s[s.iou == 0.1]
folds = [f"C{k}" for k in range(1, 7)]
designs = (("fs_b", "One frame × k videos (image-level CRC)", "#eb6834", "s"),
           ("fs_a", "Video budget (cluster-level CRC, all frames of k videos)", "#2a78d6", "o"))
eb = dict(elinewidth=1.0, capsize=1.5, capthick=1.0)
panels = ((0.10, "risk_cl_mean", "Mean video-averaged\nmiss rate $-$ $\\alpha$ ($\\alpha$ = 0.10)"),
          (0.10, "degenerate_frac", "Share that could not\ncertify ($\\alpha$ = 0.10)"),
          (0.20, "risk_cl_mean", "Mean video-averaged\nmiss rate $-$ $\\alpha$ ($\\alpha$ = 0.20)"))

fig, axs = plt.subplots(3, 6, figsize=(183 / 25.4, 120 / 25.4), sharex=True, sharey="row")
for r, (alpha, col, ylab) in enumerate(panels):
    for c, f in enumerate(folds):
        ax = axs[r, c]
        if col == "risk_cl_mean":
            ax.axhline(0, color="#52514e", lw=1.0, ls=(0, (2, 1)), zorder=1)
        for st, lab, colr, mk in designs:
            d = s[(s.strategy == st) & (s.alpha == alpha) & (s.fold == f)]
            a = d[d.seed == "all"].set_index("budget")
            a = a[a.k_eff == a.index]  # drop capped budgets (C4: k = 40 -> 24 overlaps k = 20); in Suppl. Table S15
            ps = d[d.seed != "all"].groupby("budget")[col]
            off = alpha if col == "risk_cl_mean" else 0.0
            m, lo, hi = a[col] - off, ps.min().reindex(a.index) - off, ps.max().reindex(a.index) - off
            ax.errorbar(a.k_eff, m, yerr=[m - lo, hi - m], color=colr, marker=mk, ms=4.5 if st == "fs_b" else 2.5,
                        lw=1.2, label=lab, zorder=3, mfc="white" if st == "fs_b" else colr, **eb)
        ax.set_xscale("log"); ax.set_xlim(3.5, 55); ax.set_xticks([5, 10, 20, 40]); ax.set_xticklabels(["5", "10", "20", "40"])
        ax.minorticks_off()
        if col == "degenerate_frac":
            ax.set_ylim(-0.03, 1.03)
        if r == 0:
            ax.set_title(f, fontweight="bold")
        if c == 0:
            ax.set_ylabel(ylab)
            ax.text(-0.75, 1.02, "abc"[r], transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")
fig.supxlabel("Labelled positive target videos, $k$ (logarithmic axis)", fontsize=7)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=2.2, columnspacing=1.5)
fig.tight_layout(h_pad=0.8, w_pad=0.6)
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"):
    fig.savefig(out / f"fig_cluster_budget.{ext}")
