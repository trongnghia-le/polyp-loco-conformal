"""Detection quality per LOCO run, in-domain (val = held-out clusters of the source centres) vs
held-out centre (target). Polyp-level matching at IoU >= 0.5, one-to-one, greedy by score (as docs/metric.md).
Reads outputs/kaggle/<fold>_seed<k>/{preds_*.parquet,index_*.csv} + data/manifest.csv (gt_boxes);
writes outputs/eval/detection_summary.csv and prints a fold x seed table."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import iou  # noqa: E402

clean = lambda p: re.sub(r"[^A-Za-z0-9_./-]", "", p)  # same sanitizing as make_kaggle_upload.py
man = pd.read_csv(ROOT / "data" / "manifest.csv")
man["key"] = man.path.map(clean)
GT = {k: np.asarray(json.loads(g), float).reshape(-1, 4) for k, g in zip(man.key, man.gt_boxes)}


def match(gt, boxes, scores, thr=0.5):
    """TP flag per detection (descending score), one-to-one greedy."""
    order = np.argsort(-scores, kind="stable")
    tp = np.zeros(len(order), bool)
    if len(gt) and len(order):
        m = iou(boxes[order], gt)
        free = np.ones(len(gt), bool)
        for i, row in enumerate(m):
            row = np.where(free, row, -1.0)
            j = row.argmax()
            if row[j] >= thr:
                free[j], tp[i] = False, True
    return scores[order], tp


def evaluate(paths, preds):
    n_gt, sc, tps, neg_scores = 0, [], [], []
    g = {k: v for k, v in preds.groupby("orig_path")}
    for p in paths:
        gt = GT[p]
        d = g.get(p)
        b = d[["x1", "y1", "x2", "y2"]].to_numpy(float) if d is not None else np.zeros((0, 4))
        s = d.score.to_numpy(float) if d is not None else np.zeros(0)
        n_gt += len(gt)
        if len(gt) == 0:
            neg_scores.append(s)
        s_sorted, tp = match(gt, b, s)
        sc.append(s_sorted); tps.append(tp)
    sc, tp = np.concatenate(sc), np.concatenate(tps)
    o = np.argsort(-sc, kind="stable"); sc, tp = sc[o], tp[o]
    ctp = np.cumsum(tp); prec = ctp / np.arange(1, len(tp) + 1); rec = ctp / max(n_gt, 1)
    # all-point interpolated AP (VOC 2010+)
    mprec = np.concatenate([[0], prec, [0]]); mrec = np.concatenate([[0], rec, [rec[-1] if len(rec) else 0]])
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    ap50 = float(np.sum((mrec[1:] - mrec[:-1]) * mprec[1:]))
    r = lambda thr: float(tp[sc >= thr].sum() / max(n_gt, 1))
    n_neg = len(neg_scores)
    fp_neg = float(np.mean([(s >= 0.25).sum() for s in neg_scores])) if n_neg else np.nan
    return dict(n_img=len(paths), n_polyps=n_gt, AP50=ap50, recall_at_0p25=r(0.25), max_recall=r(0.0),
                miss_at_0p25=1 - r(0.25), n_neg=n_neg, fp_per_neg_frame_0p25=fp_neg)


rows = []
for d in sorted((ROOT / "outputs" / "kaggle").glob("C*_seed*")):
    fold, seed = d.name.split("_seed")
    pq = next(d.glob("preds_*.parquet"), None)
    if pq is None:
        continue
    preds, idx = pd.read_parquet(pq), pd.read_csv(next(d.glob("index_*.csv")))
    for role in ("val", "target"):
        paths = idx.loc[idx.role == role, "orig_path"].tolist()
        rows.append(dict(fold=fold, seed=int(seed), role=role, **evaluate(paths, preds[preds.role == role])))
res = pd.DataFrame(rows)
out = ROOT / "outputs" / "eval"; out.mkdir(parents=True, exist_ok=True)
res.to_csv(out / "detection_summary.csv", index=False)

pd.set_option("display.width", 200)
t = res.pivot_table(index="fold", columns="role", values=["AP50", "miss_at_0p25"], aggfunc=["mean", "min", "max"])
print(res.groupby(["fold", "role"])[["AP50", "recall_at_0p25", "miss_at_0p25", "max_recall", "fp_per_neg_frame_0p25"]]
      .mean().round(3).to_string())
print("\nseeds per fold:", res.groupby("fold").seed.nunique().to_dict())

# self-check: our val AP50 should be close to Ultralytics' val mAP50 stored in run_meta (same images, same IoU 0.5)
for d in sorted((ROOT / "outputs" / "kaggle").glob("C*_seed*")):
    meta = next(d.glob("run_meta_*.json"), None)
    if meta:
        u = json.loads(meta.read_text())["val"]["mAP50"]
        fold, seed = d.name.split("_seed")
        ours = res[(res.fold == fold) & (res.seed == int(seed)) & (res.role == "val")].AP50.iloc[0]
        assert abs(ours - u) < 0.06, (d.name, ours, u)
print("val AP50 matches Ultralytics mAP50 within 0.06 for every run")
