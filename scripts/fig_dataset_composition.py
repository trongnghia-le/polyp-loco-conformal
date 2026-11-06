"""Fig: PolypGen frames per center by source (single positive / positive sequence / negative sequence).
Reads data/manifest.csv; writes results/figures/fig_dataset_composition.{pdf,png}."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")

m = pd.read_csv(ROOT / "data" / "manifest.csv")
order = ["single", "seq_pos", "seq_neg"]
labels = {"single": "Single frames", "seq_pos": "Positive sequence frames", "seq_neg": "Negative sequence frames"}
t = m.pivot_table(index="center", columns="source", values="path", aggfunc="count", fill_value=0)[order]
t = t.loc[sorted(t.index, reverse=True)]  # C1 on top
assert t.values.sum() == 8037

fig, ax = plt.subplots(figsize=(89 / 25.4, 50 / 25.4))
left = 0
for src, color in zip(order, ["#2a78d6", "#eb6834", "#1baf7a"]):
    ax.barh(t.index, t[src], left=left, color=color, height=0.62, label=labels[src],
            edgecolor="white", linewidth=1.0)  # white gap between stacked segments
    left = left + t[src]
for y, total in zip(t.index, t.sum(axis=1)):
    ax.text(total + 30, y, f"{total:,}", va="center", ha="left", fontsize=6.5, color="#52514e")
ax.set_xlabel("Frames")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xlim(0, t.sum(axis=1).max() * 1.12)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, handlelength=1, columnspacing=0.8)

out = ROOT / "results" / "figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "eps", "png"):
    fig.savefig(out / f"fig_dataset_composition.{ext}")
print(t.assign(total=t.sum(axis=1)))
