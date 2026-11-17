"""Fig: realised target miss rate vs alpha per held-out centre (IoU >= 0.1) for source calibration, two estimands.
Row a: frame-averaged miss rate, image-level CRC: in-domain reference (video splits of the source calibration set),
       pooled source (every calibration frame counts once) and centre-balanced pooled source (weighted CRC, every
       source centre carries equal weight; conservative test weight).
Row b: video-averaged miss rate, cluster-level CRC: in-domain (splits of positive source calibration videos), pooled
       source and centre-balanced (plug-in weights; the conservative version cannot certify alpha < 1/6 because two
       source centres have 1 and 4 positive calibration videos).
Reads outputs/eval/conformal_summary.csv, conformal_cluster_summary.csv and revision2_R6a_weighted.csv;
writes results/figures/fig_risk_vs_alpha.{pdf,eps,png}. Point = mean over the three training seeds (in-domain: over
200 splits per seed); error bar = range of the per-seed values."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
rd = lambda f: pd.read_csv(ROOT / "outputs/eval" / f, dtype={"seed": str})
cs, cc, w = rd("conformal_summary.csv"), rd("conformal_cluster_summary.csv"), rd("revision2_R6a_weighted.csv")
cs, cc, w = cs[cs.iou == 0.1], cc[cc.iou == 0.1], w[w.iou == 0.1]
alphas = sorted(cs.alpha.unique())
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
rows = [  # (row label, y column, series [(frame, label, colour, marker)])
    ("Frame-averaged miss rate", [
        (cs[(cs.strategy == "in_domain")].assign(v=lambda d: d.risk_mean), "In-domain (video splits of source calibration set)", BLUE, "o"),
        (w[(w.level == "frame") & (w.variant == "equal")].assign(v=lambda d: d.risk_fr), "Pooled source", ORANGE, "s"),
        (w[(w.level == "frame") & (w.variant == "balanced_conservative")].assign(v=lambda d: d.risk_fr), "Pooled source, centre-balanced", GREEN, "^")]),
    ("Video-averaged miss rate", [
        (cc[(cc.strategy == "in_domain_cl")].assign(v=lambda d: d.risk_cl_mean), None, BLUE, "o"),
        (w[(w.level == "cluster") & (w.variant == "equal")].assign(v=lambda d: d.risk_cl), None, ORANGE, "s"),
        (w[(w.level == "cluster") & (w.variant == "balanced_plugin")].assign(v=lambda d: d.risk_cl), None, GREEN, "^")])]
fig, axs = plt.subplots(2, 6, figsize=(183 / 25.4, 92 / 25.4), sharex=True, sharey=True)
for r, (ylab, series) in enumerate(rows):
    for c in range(6):
        ax, fold = axs[r, c], f"C{c + 1}"
        ax.plot([0, 0.35], [0, 0.35], color="#8a8985", lw=1.0, ls=(0, (4, 2)), zorder=1, label=r"$y=\alpha$" if r == 0 else None)
        for i, (d, lab, col, mk) in enumerate(series):
            d = d[d.fold == fold]
            mean = d[d.seed == "all"].set_index("alpha").v.reindex(alphas)
            ps = d[d.seed.isin(["0", "1", "2"])].groupby("alpha").v
            lo, hi = mean - ps.min().reindex(alphas), ps.max().reindex(alphas) - mean
            x = [a + (i - 1) * 0.006 for a in alphas]  # small dodge so error bars do not overlap
            ax.errorbar(x, mean, yerr=[lo, hi], color=col, marker=mk, ms=3.0, lw=1.0, elinewidth=1.0, capsize=1.5,
                        capthick=1.0, label=lab, zorder=3)
        ax.set_xlim(0.03, 0.32); ax.set_ylim(0, 0.7)
        ax.set_xticks([0.05, 0.10, 0.20, 0.30]); ax.set_xticklabels(["0.05", "0.10", "0.20", "0.30"], rotation=90)
        if r == 0:
            ax.set_title(fold, fontweight="bold")
        if c == 0:
            ax.set_ylabel(ylab)
            ax.text(-0.62, 1.02, "ab"[r], transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom")
fig.supxlabel(r"Target miss rate $\alpha$", fontsize=7)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1.8, columnspacing=1.2)
fig.tight_layout(h_pad=0.8, w_pad=0.6)
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"):
    fig.savefig(out / f"fig_risk_vs_alpha.{ext}")
print("seeds:", cs[cs.seed == "all"].n_seeds.max())
