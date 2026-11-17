"""Generate Supplementary Fig. S5: qualitative failure and trade-off examples from held-out PolypGen frames.

All frames, ground-truth boxes, YOLO11s seed-0 predictions, and calibrated thresholds
are loaded programmatically from the project manifest, stored prediction parquets, and
evaluation outputs at alpha = 0.10, IoU >= 0.1.

Outputs:
  results/figures/fig_qualitative.pdf
  results/figures/fig_qualitative.eps
  results/figures/fig_qualitative.png
"""
from __future__ import annotations

import ast
import pathlib
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.metrics import iou  # noqa: E402

plt.style.use(ROOT / "configs" / "paper.mplstyle")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

IMG_ROOT = ROOT / "data" / "raw" / "PolypGen2021_MultiCenterData_v3"
OUT = ROOT / "results" / "figures"

# High-contrast, colorblind-safe overlay palette over endoscopic mucosa
COL_GT = "#ffffff"        # Dashed white over dark casing for ground truth
COL_DET = "#00d4ff"       # Sky-cyan solid for prediction >= pooled threshold
COL_MISS = "#ff8c1a"      # Vivid orange solid for prediction < pooled threshold (recovered at recal)
COL_FA = "#ffd000"        # Bright gold solid for false alarms at recalibrated threshold
HALO = "#0b0f17"
CARD_BG = "#0f172a"


def _load_thresholds() -> dict[str, float]:
    w = pd.read_csv(ROOT / "outputs" / "eval" / "revision2_R6a_weighted.csv")
    c5_pool = w[
        (w["fold"] == "C5")
        & (w["seed"] == "0")
        & (w["alpha"].round(2) == 0.10)
        & (w["iou"].round(2) == 0.10)
        & (w["level"] == "frame")
        & (w["variant"] == "equal")
    ]["lam"].iloc[0]
    c5_bal = w[
        (w["fold"] == "C5")
        & (w["seed"] == "0")
        & (w["alpha"].round(2) == 0.10)
        & (w["iou"].round(2) == 0.10)
        & (w["level"] == "frame")
        & (w["variant"] == "balanced_conservative")
    ]["lam"].iloc[0]
    c6_pool = w[
        (w["fold"] == "C6")
        & (w["seed"] == "0")
        & (w["alpha"].round(2) == 0.10)
        & (w["iou"].round(2) == 0.10)
        & (w["level"] == "frame")
        & (w["variant"] == "equal")
    ]["lam"].iloc[0]

    c4_pool = w[
        (w["fold"] == "C4")
        & (w["seed"] == "0")
        & (w["alpha"].round(2) == 0.10)
        & (w["iou"].round(2) == 0.10)
        & (w["level"] == "frame")
        & (w["variant"] == "equal")
    ]["lam"].iloc[0]

    cl = pd.read_parquet(ROOT / "outputs" / "eval" / "conformal_cluster_long.parquet")
    recal_cert_c5 = cl[
        (cl["fold"] == "C5")
        & (cl["seed"] == 0)
        & (cl["iou"].round(2) == 0.10)
        & (cl["alpha"].round(2) == 0.10)
        & (cl["strategy"] == "fs_b")
        & (cl["budget"] == 20)
        & (~cl["degenerate"])
    ]["lam"].median()
    recal_cert_c4 = cl[
        (cl["fold"] == "C4")
        & (cl["seed"] == 0)
        & (cl["iou"].round(2) == 0.10)
        & (cl["alpha"].round(2) == 0.10)
        & (cl["strategy"] == "fs_b")
        & (cl["budget"] == 20)
        & (~cl["degenerate"])
    ]["lam"].median()

    return {
        "c5_pool": float(c5_pool),
        "c5_bal": float(c5_bal),
        "c6_pool": float(c6_pool),
        "c4_pool": float(c4_pool),
        "c5_recal_med": float(recal_cert_c5),
        "c4_recal_med": float(recal_cert_c4),
    }


def _draw_box(
    ax,
    box: list[float],
    *,
    color: str,
    ls: str | tuple = "-",
    lw: float = 1.6,
    zorder: int = 4,
) -> None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ax.add_patch(
        mpatches.Rectangle(
            (x1, y1), w, h, fill=False, edgecolor=HALO, linewidth=lw + 1.2, linestyle="-", zorder=zorder - 0.2
        )
    )
    ax.add_patch(
        mpatches.Rectangle(
            (x1, y1), w, h, fill=False, edgecolor=color, linewidth=lw, linestyle=ls, zorder=zorder
        )
    )


def _tag_pill(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    fg: str,
    va: str = "top",
    ha: str = "left",
    fontsize: float = 7.0,
) -> None:
    ax.text(
        x,
        y,
        text,
        color=fg,
        fontsize=fontsize,
        fontweight="bold",
        va=va,
        ha=ha,
        bbox={
            "boxstyle": "round,pad=0.24,rounding_size=0.18",
            "facecolor": CARD_BG,
            "edgecolor": fg,
            "linewidth": 1.0,
        },
        zorder=7,
    )


def _footer_bar(ax, x0_lim: float, x1_lim: float, h_img: int, h_bar: int, text: str) -> None:
    ax.add_patch(
        mpatches.Rectangle(
            (x0_lim, h_img),
            x1_lim - x0_lim,
            h_bar,
            facecolor=CARD_BG,
            edgecolor="none",
            zorder=6,
        )
    )
    ax.plot([x0_lim, x1_lim], [h_img, h_img], color="#475569", lw=1.0, zorder=6.5)
    ax.text(
        0.5 * (x0_lim + x1_lim),
        h_img + 0.5 * h_bar,
        text,
        color="#f1f5f9",
        fontsize=7.0,
        fontweight="bold",
        va="center",
        ha="center",
        zorder=7,
    )


def build_figure() -> plt.Figure:
    thr = _load_thresholds()
    assert abs(thr["c5_pool"] - 0.244) < 1e-6, thr
    assert abs(thr["c5_bal"] - 0.128) < 1e-6, thr
    assert abs(thr["c6_pool"] - 0.162) < 1e-6, thr
    assert abs(thr["c4_pool"] - 0.269) < 1e-6, thr
    assert abs(thr["c5_recal_med"] - 0.004) < 1e-6, thr
    assert abs(thr["c4_recal_med"] - 0.004) < 1e-6, thr

    manifest = pd.read_csv(ROOT / "data" / "manifest.csv").set_index("path")
    p4 = pd.read_parquet(ROOT / "outputs" / "kaggle" / "C4_seed0" / "preds_C4_seed0.parquet")
    p5 = pd.read_parquet(ROOT / "outputs" / "kaggle" / "C5_seed0" / "preds_C5_seed0.parquet")
    p6 = pd.read_parquet(ROOT / "outputs" / "kaggle" / "C6_seed0" / "preds_C6_seed0.parquet")

    panels = [
        {
            "letter": "a",
            "title": "Missed under pooled calibration (C5)",
            "path": "data_C5/images_C5/EndoCV2021_C5_50000282.jpg",
            "pdf": p5,
            "crop_x": (655, 1920),
            "xlim": (568, 2008),  # Uniform 1440 x 1080 (4:3) black pillarboxed frame
        },
        {
            "letter": "b",
            "title": "Below the detector miss floor (C6)",
            "path": "data_C6/images_C6/EndoCV2021_C6_0100023.jpg",
            "pdf": p6,
            "crop_x": (480, 1920),
            "xlim": (480, 1920),  # 1440 x 1080 (4:3)
        },
        {
            "letter": "c",
            "title": "Hit at IoU ≥ 0.1, missed at IoU ≥ 0.5 (C5)",
            "path": "data_C5/images_C5/EndoCV2021_C5_50000193.jpg",
            "pdf": p5,
            "crop_x": (655, 1920),
            "xlim": (568, 2008),
        },
        {
            "letter": "d",
            "title": "False alarms at certified recalibration (C4 neg)",
            "path": "sequenceData/negativeOnly/seq8_neg/EndoCV2021_seq8_neg_45_endocv2021_negative_866.jpg",
            "pdf": p4,
            "crop_x": (550, 1920),
            "xlim": (515, 1955),  # Clean 1440 x 1080 (4:3) around full colon lumen octagon
        },
    ]

    h_bar = 96  # pixels in y-coordinates for dedicated non-overlapping footer bar below each image
    w_mm, h_mm = 183.0, 159.0
    fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4), dpi=300)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.015,
        right=0.985,
        top=0.895,
        bottom=0.012,
        wspace=0.045,
        hspace=0.135,
    )

    for idx, spec in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        rel = spec["path"]
        img = np.asarray(Image.open(IMG_ROOT / rel).convert("RGB"))
        h, w = img.shape[:2]
        assert (w, h) == (1920, 1080), (rel, w, h)
        if rel.endswith("EndoCV2021_C6_0100023.jpg"):   # burned-in examination date/time: mask (de-identification)
            img = img.copy()
            stamp = img[960:1025, 1475:1885]
            assert (stamp.max(axis=2) > 200).sum() > 500, "timestamp not where expected"
            img[960:1025, 1475:1885] = 0

        x0, x1 = spec["crop_x"]
        cropped = img[:, x0:x1]
        ax.imshow(cropped, extent=[x0, x1, h, 0], interpolation="bilinear")
        ax.set_facecolor("#030406")
        ax.set_xlim(spec["xlim"])
        ax.set_ylim(h + h_bar, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#475569")
            sp.set_linewidth(1.0)

        gt_boxes = ast.literal_eval(manifest.loc[rel, "gt_boxes"])
        sub = spec["pdf"][(spec["pdf"]["orig_path"] == rel) & (spec["pdf"]["role"] == "target")].sort_values(
            "score", ascending=False
        )

        # Draw ground-truth boxes first
        for g in gt_boxes:
            assert g[0] >= x0 and g[2] <= x1, (rel, g)
            _draw_box(ax, g, color=COL_GT, ls=(0, (4.0, 2.2)), lw=1.65, zorder=5)

        if spec["letter"] == "a":
            top = sub.iloc[0]
            box = [float(top[k]) for k in ("x1", "y1", "x2", "y2")]
            score = float(top["score"])
            box_iou = float(iou([box], gt_boxes)[0, 0])
            assert abs(score - 0.200) < 0.001 and abs(box_iou - 0.887) < 0.002
            assert box[0] >= x0 and score < thr["c5_pool"]
            _draw_box(ax, box, color=COL_MISS, ls="-", lw=1.55, zorder=4)
            _tag_pill(
                ax,
                box[2] + 24,
                box[1] + 20,
                f"Top pred: {score:.3f}\nIoU = {box_iou:.2f}",
                fg=COL_MISS,
                va="top",
                ha="left",
            )
            footer = f"Missed at pooled λ {thr['c5_pool']:.3f}; kept at {thr['c5_bal']:.3f} and {thr['c5_recal_med']:.3f}"

        elif spec["letter"] == "b":
            assert len(sub) == 0
            g = gt_boxes[0]
            _tag_pill(
                ax,
                g[0] - 24,
                g[1] + 24,
                "GT polyp\n0 exports ≥ 0.001",
                fg=COL_GT,
                va="top",
                ha="right",
            )
            footer = f"Pooled λ {thr['c6_pool']:.3f}; no box exported at ≥ 0.001"

        elif spec["letter"] == "c":
            top = sub.iloc[0]
            box = [float(top[k]) for k in ("x1", "y1", "x2", "y2")]
            score = float(top["score"])
            box_iou = float(iou([box], gt_boxes)[0, 0])
            assert abs(score - 0.6426) < 0.002 and abs(box_iou - 0.292) < 0.002
            assert score >= thr["c5_pool"]
            _draw_box(ax, box, color=COL_DET, ls="-", lw=1.6, zorder=4)
            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            ax.plot(cx, cy, marker="+", color=HALO, markersize=8.5, markeredgewidth=3.4, zorder=5.7)
            ax.plot(cx, cy, marker="+", color=COL_DET, markersize=7.5, markeredgewidth=1.8, zorder=6.0)
            _tag_pill(
                ax,
                box[2] + 22,
                box[1] + 10,
                f"Pred: {score:.3f}\nIoU = {box_iou:.2f}",
                fg=COL_DET,
                va="top",
                ha="left",
            )
            footer = f"Pooled λ {thr['c5_pool']:.3f}: hit at IoU ≥ 0.1, miss at IoU ≥ 0.5"

        elif spec["letter"] == "d":
            assert len(gt_boxes) == 0
            raised_pool = sub[sub["score"] >= thr["c4_pool"]]
            raised_recal = sub[sub["score"] >= thr["c4_recal_med"]]
            assert len(raised_pool) == 0 and len(raised_recal) == 2
            bx_arr = raised_recal[["x1", "y1", "x2", "y2"]].to_numpy()
            assert iou(bx_arr[[0]], bx_arr[[1]])[0, 0] == 0.0
            for r_i, (_, r) in enumerate(raised_recal.iterrows()):
                box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
                assert box[0] >= x0
                _draw_box(ax, box, color=COL_FA, ls="-", lw=1.6, zorder=4)
                _tag_pill(
                    ax,
                    box[0] + 18,
                    box[1] + 18,
                    f"FA {r_i + 1}: {float(r['score']):.3f}",
                    fg=COL_FA,
                    va="top",
                    ha="left",
                )
            footer = f"No polyp; 0 boxes at {thr['c4_pool']:.3f}, 2 false alarms at {thr['c4_recal_med']:.3f}"

        _footer_bar(ax, spec["xlim"][0], spec["xlim"][1], h, h_bar, footer)

        ax.set_title(
            f"{spec['letter']}   {spec['title']}",
            loc="left",
            fontsize=7.0,
            fontweight="bold",
            color="#1e293b",
            pad=4.0,
        )

    legend_handles = [
        mlines.Line2D([], [], color=COL_GT, ls="--", lw=1.6, label="Ground-truth polyp"),
        mlines.Line2D([], [], color=COL_DET, ls="-", lw=1.6, label=r"Detected (score $\geq \lambda_{\mathrm{pool}}$)"),
        mlines.Line2D([], [], color=COL_MISS, ls="-", lw=1.6, label=r"Missed by pooled (score $< \lambda_{\mathrm{pool}}$)"),
        mlines.Line2D([], [], color=COL_FA, ls="-", lw=1.6, label=r"False alarm at certified $\lambda_{\mathrm{recal}}$"),
        mlines.Line2D([], [], color=COL_DET, ls="none", marker="+", ms=7.0, mew=1.8, label="Predicted box centre"),
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        frameon=True,
        framealpha=1.0,
        facecolor=CARD_BG,
        edgecolor="#475569",
        fontsize=7.0,
        handlelength=1.7,
        columnspacing=1.05,
    )
    leg.get_frame().set_linewidth(1.0)
    for txt in leg.get_texts():
        txt.set_color("#f8fafc")

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps", "png"):
        kw = {"dpi": 600} if ext == "png" else {}
        fig.savefig(OUT / f"fig_qualitative.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("Saved fig_qualitative.{pdf,eps,png}")


if __name__ == "__main__":
    build_figure()
