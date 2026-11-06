"""Fig: few-shot target recalibration counted in labelled VIDEOS (clusters). Per held-out centre (columns) and
alpha = 0.10 / 0.20 (rows): mean cluster-averaged target risk minus alpha vs number of labelled positive target
clusters k, for (a) cluster-level CRC, (b) image CRC on one frame per video, and the frame-budget few-shot
(image CRC on 20/50/100 positive frames) placed at its median cluster count. IoU >= 0.1.
Reads outputs/eval/conformal_cluster_summary.csv; writes results/figures/fig_cluster_budget.{pdf,eps,png}.
Points = mean over 500 draws x 2 seeds; error bars = range of the per-seed means."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
s = pd.read_csv(ROOT / "outputs/eval/conformal_cluster_summary.csv")
s = s[s.iou == 0.1]
folds = [f"C{k}" for k in range(1, 7)]
designs = (("fs_b", "Image CRC, one frame per video", "#eb6834", "s", "-"),
           ("fs_a", "Cluster-level CRC (all frames of k videos)", "#2a78d6", "o", "-"),
           ("fs_frames", "Image CRC, 20/50/100 frames (at median k)", "#1baf7a", "^", (0, (2, 1))))
fig, axs = plt.subplots(2, 6, figsize=(183 / 25.4, 90 / 25.4), sharex=True, sharey="row")
for r, alpha in enumerate((0.10, 0.20)):
    for c, f in enumerate(folds):
        ax = axs[r, c]
        ax.axhline(0, color="#52514e", lw=1.0, ls=(0, (2, 1)), zorder=1)
        for st, lab, col, mk, ls in designs:
            d = s[(s.strategy == st) & (s.alpha == alpha) & (s.fold == f)]
            a = d[d.seed == "all"].sort_values("budget")
            ps = d[d.seed != "all"].groupby("budget").risk_cl_mean
            x = a.k_cal_median.to_numpy() if st == "fs_frames" else a.k_eff.to_numpy()
            y = (a.risk_cl_mean - alpha).to_numpy()
            lo, hi = ps.min().reindex(a.budget).to_numpy() - alpha, ps.max().reindex(a.budget).to_numpy() - alpha
            ax.errorbar(x, y, yerr=[y - lo, hi - y], color=col, marker=mk, ms=4.5 if st == "fs_b" else 2.5, ls=ls,
                        lw=1.2, elinewidth=1.0, capsize=1.5, capthick=1.0, label=lab, zorder=3,
                        mfc="white" if st == "fs_b" else col)
        ax.set_xscale("log"); ax.set_xlim(2.4, 140); ax.set_xticks([5, 20, 100]); ax.set_xticklabels(["5", "20", "100"])
        ax.minorticks_off()
        if r == 0:
            ax.set_title(f"{'abcdef'[c]}  {f}", loc="left", fontweight="bold")
        if r == 1:
            ax.set_xlabel("Labelled target videos")
    axs[r, 0].set_ylabel(f"Mean risk $-$ $\\alpha$\n($\\alpha$ = {alpha:.2f})")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=2.6, columnspacing=1.5)
fig.tight_layout(w_pad=0.6, h_pad=0.8)
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"):
    fig.savefig(out / f"fig_cluster_budget.{ext}")
