"""Fig: the six LOCO folds - images and positive frames per role (train / val / cal / target).
Reads splits/fold_C*.csv + data/manifest.csv; writes results/figures/fig_splits.{pdf,png}."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
pos = pd.read_csv(ROOT / "data/manifest.csv").set_index("path").n_gt.gt(0)
roles = ["train", "val", "cal", "target"]
colors = dict(zip(roles, ["#2a78d6", "#eda100", "#1baf7a", "#eb6834"]))
labels = {"train": "Train", "val": "Validation", "cal": "Calibration", "target": "Held-out target"}
folds = [f"C{k}" for k in range(1, 7)]
imgs, poss = {}, {}
for c in folds:
    f = pd.read_csv(ROOT / f"splits/fold_{c}.csv"); f["pos"] = f.path.map(pos)
    assert len(f) == 8037
    imgs[c] = f.role.value_counts().reindex(roles); poss[c] = f[f.pos].role.value_counts().reindex(roles).fillna(0)
imgs, poss = pd.DataFrame(imgs).T, pd.DataFrame(poss).T
ylab = [f"Hold out {c}" for c in folds][::-1]

fig, axs = plt.subplots(1, 2, figsize=(183 / 25.4, 52 / 25.4), sharey=True)
for ax, t, xl, title in zip(axs, (imgs, poss), ("Images", "Positive frames"), ("a  All frames", "b  Frames with a polyp")):
    t = t.loc[folds[::-1]]; left = 0; ax.set_xlim(0, 8037 if t.sum(axis=1).max() > 5000 else t.sum(axis=1).max())
    for r in roles:
        ax.barh(ylab, t[r], left=left, color=colors[r], height=0.64, edgecolor="white", linewidth=1.0, label=labels[r])
        for y, l, v in zip(ylab, left if r != "train" else [0] * 6, t[r]):
            if v >= 0.05 * ax.get_xlim()[1]:  # label only segments wide enough
                ax.text(l + v / 2, y, f"{int(v):,}", ha="center", va="center", fontsize=6.5,
                        color="white" if r != "val" else "#0b0b0b")
        left = left + t[r]
    ax.set_xlabel(xl); ax.set_title(title, loc="left", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}")); ax.tick_params(axis="y", length=0)
axs[0].set_xlim(0, 8037); axs[1].set_xlim(0, poss.sum(axis=1).max())
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, handlelength=1, columnspacing=1.5)
fig.tight_layout(w_pad=1.5)
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"): fig.savefig(out / f"fig_splits.{ext}")
print(imgs, poss, sep="\n")
