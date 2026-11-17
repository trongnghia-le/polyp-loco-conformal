"""Fig: does the pooled-source CRC failure persist when no single video dominates the source calibration set? (R3, M1/M2)
Reads outputs/eval/revision_R3_calibration_composition.csv (scripts/revision_analyses.py); writes
results/figures/fig_calibration_composition.{pdf,eps,png}. IoU >= 0.1, alpha = 0.10. Point = mean over seeds (and over
200 random subsamples for the capped variants); error bar = range of the three per-seed means."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
ALPHA = 0.10
s = pd.read_csv(ROOT / "outputs/eval/revision_R3_calibration_composition.csv", dtype={"seed": str})
s = s[(s.iou == 0.1) & (s.alpha == ALPHA)]
series = [("all", "All calibration frames (main analysis)", "#2a78d6", "o"),
          ("cap20", "≤ 20 positive frames per video", "#eb6834", "s"),
          ("cap1", "1 positive frame per video", "#1baf7a", "^"),
          ("excl_C2+C4", "C2 and C4 removed from calibration", "#eda100", "D")]
folds = [f"C{i}" for i in range(1, 7)]
fig, ax = plt.subplots(figsize=(89 / 25.4, 62 / 25.4))
ax.axhline(ALPHA, color="#8a8985", lw=1.0, ls=(0, (4, 2)), zorder=1)
ax.text(5.45, ALPHA + 0.008, r"$\alpha$ = 0.10", ha="right", va="bottom", color="#52514e", fontsize=6.5)
for i, (v, lab, c, m) in enumerate(series):
    xs, ys, lo, hi = [], [], [], []
    for k, f in enumerate(folds):
        d = s[(s.fold == f) & (s.variant == v)]
        if d.empty:  # held-out centre is itself C2 or C4: variant not defined
            continue
        mean = d[d.seed == "all"].risk_fr_mean.iloc[0]
        per = d[d.seed != "all"].risk_fr_mean
        xs.append(k + (i - 1.5) * 0.17); ys.append(mean); lo.append(mean - per.min()); hi.append(per.max() - mean)
    ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=m, ms=3.5, lw=0, elinewidth=1.0, capsize=1.5, capthick=1.0,
                label=lab, zorder=3)
ax.set_xticks(range(6)); ax.set_xticklabels(folds); ax.set_xlim(-0.5, 5.5)
ax.set_ylim(0, 0.55)
ax.set_xlabel("Held-out centre")
ax.set_ylabel("Realised frame-averaged miss rate")
ax.legend(loc="upper right", handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)
fig.tight_layout()
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"):
    fig.savefig(out / f"fig_calibration_composition.{ext}")
print(s[s.seed == "all"].pivot(index="variant", columns="fold", values="risk_fr_mean").round(3).to_string())
