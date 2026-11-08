"""Fig: CVC-ClinicDB (experiment B1), IoU >= 0.1, 18 PolypGen-trained YOLO11s detectors.
a  Within-ClinicDB image CRC, ~300 calibration frames: realized test risk (frame-averaged) vs alpha for a random FRAME
   split and a random VIDEO split with the same calibration size.
b  External validation: realized video-averaged ClinicDB risk of the pooled PolypGen-source threshold, and of few-shot
   recalibration on one frame from each of 20 ClinicDB videos (tested on the other 9 videos).
Points = mean over the 18 models of the per-model mean (a, b few-shot: over 500 draws); error bars = s.d. across the
18 models. Panel a shaded bands = 5-95 % range of the realized risk over the 500 draws x 18 models. Reads outputs/eval/clinicdb_summary.csv (scripts/clinicdb_crc.py); writes results/figures/fig_clinicdb.*"""
from pathlib import Path
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
s = pd.read_csv(ROOT / "outputs/eval/clinicdb_summary.csv")
s = s[(s.run == "all") & (s.iou == 0.1)]
panels = (("a  Within ClinicDB (~300 calibration frames)", "risk_fr", "Realized miss rate (frame-averaged)",
           [("frame_split", 300, "Random frame split", "#2a78d6", "o"),
            ("video_split", 300, "Random video split", "#eb6834", "s")]),
          ("b  External validation on ClinicDB", "risk_cl", "Realized miss rate (video-averaged)",
           [("pooled_source", 0, "PolypGen source calibration", "#1baf7a", "^"),
            ("fs_b", 20, "Recalibrated, 1 frame × 20 videos", "#6250d6", "D")]))
fig, axs = plt.subplots(1, 2, figsize=(183 / 25.4, 70 / 25.4))
for ax, (title, col, ylab, series) in zip(axs, panels):
    ax.plot([0, 0.35], [0, 0.35], color="#8a8985", lw=1.0, ls=(0, (4, 2)), zorder=1, label=r"$y=\alpha$")
    for i, (st, b, lab, c, mk) in enumerate(series):
        d = s[(s.strategy == st) & (s.budget == b)].sort_values("alpha")
        x = d.alpha + (i - 0.5) * 0.006  # small dodge
        if col == "risk_fr":  # panel a: 5-95 % range of the realized risk over draws (pooled over models)
            tint = mcolors.to_hex([1 - 0.18 * (1 - v) for v in mcolors.to_rgb(c)])  # opaque tint (EPS has no alpha)
            ax.fill_between(d.alpha, d.risk_fr_q05, d.risk_fr_q95, color=tint, lw=0, zorder=0.5 - i * 0.1)
        ax.errorbar(x, d[f"{col}_mean"], yerr=d[f"{col}_sd_models"], color=c, marker=mk, ms=3.5, lw=1.2,
                    elinewidth=1.0, capsize=1.5, capthick=1.0, label=lab, zorder=3)
    al = sorted(s.alpha.unique())
    ax.set_xticks(al); ax.set_xticklabels([f"{a:.2f}" for a in al])
    ax.set_xlim(0.03, 0.32); ax.set_ylim(0, 0.5)
    ax.set_xlabel(r"Target miss rate $\alpha$"); ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="upper left", handlelength=2.2)
fig.tight_layout(w_pad=2.0)
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"):
    fig.savefig(out / f"fig_clinicdb.{ext}")
print(s[s.strategy.isin(["frame_split", "video_split", "pooled_source", "fs_b"]) & s.budget.isin([0, 20, 300])]
      [["strategy", "budget", "alpha", "risk_fr_mean", "risk_fr_sd_models", "risk_cl_mean", "risk_cl_sd_models"]]
      .round(4).to_string(index=False))
