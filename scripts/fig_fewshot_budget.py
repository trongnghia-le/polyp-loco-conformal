"""Fig: violation frequency of frame-budget recalibration vs target budget (20/50/100 positive frames) per
held-out centre, alpha = 0.10 (a) and 0.20 (b), IoU >= 0.1. Black haloed segment + diamond = in-domain (video split) reference.
Reads outputs/eval/conformal_summary.csv; writes results/figures/fig_fewshot_budget.{pdf,eps,png}.
Bar = violation frequency over 500 draws x seeds; error bar = range over seeds."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
s = pd.read_csv(ROOT / "outputs/eval/conformal_summary.csv")
s = s[s.iou == 0.1]
budgets, colors = (20, 50, 100), ("#86b6ef", "#2a78d6", "#104281")  # ordinal blue ramp (validated --ordinal)
folds = [f"C{k}" for k in range(1, 7)]
fig, axs = plt.subplots(1, 2, figsize=(183 / 25.4, 62 / 25.4), sharey=True)
w = 0.26
for ax, alpha, tag in zip(axs, (0.10, 0.20), "ab"):
    for i, b in enumerate(budgets):
        d = s[(s.strategy == "few_shot") & (s.budget == b) & (s.alpha == alpha)]
        v = d[d.seed == "all"].set_index("fold").viol_freq.reindex(folds)
        ps = d[d.seed != "all"].groupby("fold").viol_freq
        x = np.arange(6) + (i - 1) * w
        ax.bar(x, v, width=w, color=colors[i], edgecolor="white", linewidth=1.0, label=f"{b} frames")
        ax.errorbar(x, v, yerr=[v - ps.min().reindex(folds), ps.max().reindex(folds) - v], fmt="none",
                    ecolor="#0b0b0b", elinewidth=1.0, capsize=1.5, capthick=1.0, zorder=6)  # above the reference line
    ind = s[(s.strategy == "in_domain") & (s.alpha == alpha) & (s.seed == "all")].set_index("fold").viol_freq.reindex(folds)
    halo = [pe.Stroke(linewidth=3.0, foreground="white"), pe.Normal()]  # contrast against all three bar shades
    ax.hlines(ind, np.arange(6) - 1.6 * w, np.arange(6) + 1.6 * w, color="#0b0b0b", lw=1.2, zorder=4, path_effects=halo)
    ax.plot(np.arange(6) + 1.6 * w, ind, ls="none", marker="D", ms=3.5, mfc="white", mec="#0b0b0b", mew=1.0, zorder=5)
    ax.plot([], [], color="#0b0b0b", lw=1.2, marker="D", ms=3.5, mfc="white", mec="#0b0b0b", mew=1.0,
            label="In-domain reference")
    ax.set_xticks(np.arange(6)); ax.set_xticklabels(folds); ax.set_xlabel("Held-out centre"); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.0); ax.set_title(f"{tag}  " + r"$\alpha$" + f" = {alpha:.2f}", loc="left", fontweight="bold")
axs[0].set_ylabel("Violation frequency\n" + r"(frame-averaged miss rate > $\alpha$)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.5, columnspacing=1.5)
fig.tight_layout(w_pad=1.5)
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"): fig.savefig(out / f"fig_fewshot_budget.{ext}")
