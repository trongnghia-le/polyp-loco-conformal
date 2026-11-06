"""Fig: realized target miss rate vs alpha per held-out centre (IoU >= 0.1), four calibration strategies.
Reads outputs/eval/conformal_summary.csv (scripts/conformal.py); writes results/figures/fig_risk_vs_alpha.{pdf,eps,png}.
Point = mean over seeds (in-domain: over 200 cluster splits; few-shot: over draws); error bar = range over seeds."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
s = pd.read_csv(ROOT / "outputs/eval/conformal_summary.csv")
s = s[(s.iou == 0.1)]
series = [("in_domain", 0, "In-domain (source split)", "#2a78d6", "o"),
          ("pooled", 0, "Pooled source", "#eb6834", "s"),
          ("worst_case", 0, "Per-source worst case", "#1baf7a", "^"),
          ("few_shot", 50, "Target few-shot (50)", "#eda100", "D")]
alphas = sorted(s.alpha.unique())
fig, axs = plt.subplots(2, 3, figsize=(183 / 25.4, 108 / 25.4), sharex=True, sharey=True)
for k, ax in enumerate(axs.flat):
    fold = f"C{k + 1}"
    ax.plot([0, 0.35], [0, 0.35], color="#8a8985", lw=1.0, ls=(0, (4, 2)), zorder=1, label=r"$y=\alpha$")
    for i, (st, b, lab, c, m) in enumerate(series):
        d = s[(s.fold == fold) & (s.strategy == st) & (s.budget == b)]
        mean = d[d.seed == "all"].set_index("alpha").risk_mean.reindex(alphas)
        ps = d[d.seed != "all"].groupby("alpha").risk_mean
        lo, hi = mean - ps.min().reindex(alphas), ps.max().reindex(alphas) - mean
        x = [a + (i - 1.5) * 0.005 for a in alphas]  # small dodge so error bars do not overlap
        ax.errorbar(x, mean, yerr=[lo, hi], color=c, marker=m, ms=3.5, lw=1.0, elinewidth=1.0, capsize=1.5,
                    capthick=1.0, label=lab, zorder=3)
    ax.set_title(f"{'abcdef'[k]}  Held-out {fold}", loc="left", fontweight="bold")
    ax.set_xlim(0.03, 0.32); ax.set_ylim(0, 0.7)
    ax.set_xticks(alphas); ax.set_xticklabels([f"{a:.2f}" for a in alphas])
    if k % 3 == 0: ax.set_ylabel("Realized miss rate (target)")
    if k >= 3: ax.set_xlabel(r"Target miss rate $\alpha$")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, handlelength=1.8, columnspacing=1.5)
fig.tight_layout(h_pad=1.0, w_pad=1.0)
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"): fig.savefig(out / f"fig_risk_vs_alpha.{ext}")
print("seeds:", s[s.seed == "all"].n_seeds.max())
