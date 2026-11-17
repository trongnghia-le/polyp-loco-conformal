"""Fig: CVC-ClinicDB (experiment B1), IoU >= 0.1, 18 PolypGen-trained YOLO11s detectors.
a  Within-ClinicDB image CRC, ~300 calibration frames, random FRAME split vs random VIDEO split of the same size.
   Plotted directly (referee minor 17) instead of overlapping bands: top, mean realised frame-averaged miss rate
   minus alpha (mean over the 18 models of the per-model mean over 500 draws; error bars = s.d. across the 18 models);
   bottom, s.d. of the realised frame-averaged miss rate over all draws x models.
b  External validation: realised video-averaged ClinicDB risk of the pooled PolypGen-source threshold, and of
   recalibration on one frame from each of 20 ClinicDB videos (tested on the other 9 videos); error bars = s.d.
   across the 18 models.
Reads outputs/eval/clinicdb_summary.csv (scripts/clinicdb_crc.py); writes results/figures/fig_clinicdb.*"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.style.use(ROOT / "configs" / "paper.mplstyle")
s = pd.read_csv(ROOT / "outputs/eval/clinicdb_summary.csv")
s = s[(s.run == "all") & (s.iou == 0.1)]
al = sorted(s.alpha.dropna().unique())  # detection rows carry no alpha
eb = dict(ms=3.5, lw=1.2, elinewidth=1.0, capsize=1.5, capthick=1.0, zorder=3)
within = [("frame_split", "Random frame split", "#2a78d6", "o"), ("video_split", "Random video split", "#eb6834", "s")]
external = [("pooled_source", 0, "PolypGen source calibration", "#1baf7a", "^"),
            ("fs_b", 20, "Recalibrated, 1 frame × 20 videos", "#6250d6", "D")]

fig, ax = plt.subplot_mosaic([["a1", "b"], ["a2", "b"]], figsize=(183 / 25.4, 74 / 25.4), layout="constrained",
                             gridspec_kw=dict(wspace=0.08))
for i, (st, lab, c, mk) in enumerate(within):
    d = s[(s.strategy == st) & (s.budget == 300)].sort_values("alpha")
    x = d.alpha + (i - 0.5) * 0.006  # small dodge
    ax["a1"].errorbar(x, d.risk_fr_mean - d.alpha, yerr=d.risk_fr_sd_models, color=c, marker=mk, label=lab, **eb)
    ax["a2"].plot(d.alpha, d.risk_fr_sd_draws, color=c, marker=mk, ms=3.5, lw=1.2, label=lab, zorder=3)
ax["a1"].axhline(0, color="#8a8985", lw=1.0, ls=(0, (4, 2)), zorder=1)
ax["a1"].set_ylim(-0.03, 0.03); ax["a1"].set_ylabel("Mean realised $-$ $\\alpha$\n(frame-averaged)")
ax["a1"].set_title("a  Within ClinicDB (~300 calibration frames)", loc="left", fontweight="bold")
ax["a1"].legend(loc="upper left", ncol=2, handlelength=2.2, columnspacing=1.2)
ax["a1"].tick_params(labelbottom=False)
ax["a2"].set_ylim(0, 0.12); ax["a2"].set_ylabel("S.d. over draws\n(frame-averaged)")
for k in ("a1", "a2"):
    ax[k].set_xlim(0.03, 0.32); ax[k].set_xticks(al)
ax["a2"].set_xticklabels([f"{a:.2f}" for a in al]); ax["a2"].set_xlabel(r"Target miss rate $\alpha$")

b = ax["b"]
b.plot([0, 0.35], [0, 0.35], color="#8a8985", lw=1.0, ls=(0, (4, 2)), zorder=1, label=r"$y=\alpha$")
for i, (st, bud, lab, c, mk) in enumerate(external):
    d = s[(s.strategy == st) & (s.budget == bud)].sort_values("alpha")
    b.errorbar(d.alpha + (i - 0.5) * 0.006, d.risk_cl_mean, yerr=d.risk_cl_sd_models, color=c, marker=mk, label=lab, **eb)
b.set_xticks(al); b.set_xticklabels([f"{a:.2f}" for a in al]); b.set_xlim(0.03, 0.32); b.set_ylim(0, 0.5)
b.set_xlabel(r"Target miss rate $\alpha$"); b.set_ylabel("Realised video-averaged miss rate")
b.set_title("b  External validation on ClinicDB", loc="left", fontweight="bold")
b.legend(loc="upper left", handlelength=2.2)
fig.align_ylabels([ax["a1"], ax["a2"]])
out = ROOT / "results" / "figures"
for ext in ("pdf", "eps", "png"):
    fig.savefig(out / f"fig_clinicdb.{ext}")
print(s[s.strategy.isin(["frame_split", "video_split"]) & (s.budget == 300)]
      [["strategy", "alpha", "risk_fr_mean", "risk_fr_sd_models", "risk_fr_sd_draws"]].round(4).to_string(index=False))
