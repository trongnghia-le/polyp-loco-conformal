"""YOLO11m vs YOLO11s on the SAME seed(s): does a bigger detector lower the miss floor on the held-out centre?
Per fold and role (val = source clusters, target = held-out centre): AP50, max recall at IoU 0.1 / 0.5 keeping every box
(conf >= 0.001; miss floor = 1 - max recall), recall at conf 0.25.  Matching: one-to-one greedy by score (docs/metric.md).
usage: python -I scripts/compare_models.py [--seeds 0]   -> outputs/eval/model_comparison.csv + printed table"""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import iou  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
a = ap.parse_args()

clean = lambda p: re.sub(r"[^A-Za-z0-9_./-]", "", p)
man = pd.read_csv(ROOT / "data" / "manifest.csv")
GT = {clean(p): np.asarray(json.loads(g), float).reshape(-1, 4) for p, g in zip(man.path, man.gt_boxes)}


def match(gt, boxes, scores, thr):
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
    g = {k: v for k, v in preds.groupby("orig_path")}
    out = {}
    for thr in (0.1, 0.5):
        n_gt, S, T = 0, [], []
        for p in paths:
            gt = GT[p]
            if not len(gt):
                continue
            d = g.get(p)
            b = d[["x1", "y1", "x2", "y2"]].to_numpy(float) if d is not None else np.zeros((0, 4))
            s = d.score.to_numpy(float) if d is not None else np.zeros(0)
            n_gt += len(gt)
            s2, tp = match(gt, b, s, thr)
            S.append(s2); T.append(tp)
        s, tp = np.concatenate(S), np.concatenate(T)
        o = np.argsort(-s, kind="stable"); s, tp = s[o], tp[o]
        ctp = np.cumsum(tp); prec = ctp / np.arange(1, len(tp) + 1); rec = ctp / n_gt
        mp = np.concatenate([[0], prec, [0]]); mr = np.concatenate([[0], rec, [rec[-1]]])
        for i in range(len(mp) - 2, -1, -1):
            mp[i] = max(mp[i], mp[i + 1])
        out[f"AP@{thr}"] = float(np.sum((mr[1:] - mr[:-1]) * mp[1:]))
        out[f"maxrecall@{thr}"] = float(tp.sum() / n_gt)
        out[f"recall_conf0.25@{thr}"] = float(tp[s >= 0.25].sum() / n_gt)
    return out


rows = []
for model, root in (("YOLO11s", ROOT / "outputs" / "kaggle"), ("YOLO11m", ROOT / "outputs" / "kaggle_m")):
    for seed in a.seeds:
        for fold in ("C1", "C2", "C3", "C4", "C5", "C6"):
            d = root / f"{fold}_seed{seed}"
            pq = next(d.glob("preds_*.parquet"), None) if d.exists() else None
            if pq is None:
                continue
            preds, idx = pd.read_parquet(pq), pd.read_csv(next(d.glob("index_*.csv")))
            for role in ("val", "target"):
                paths = idx.loc[idx.role == role, "orig_path"].tolist()
                rows.append(dict(model=model, seed=seed, fold=fold, role=role,
                                 **evaluate(paths, preds[preds.role == role])))
df = pd.DataFrame(rows)
(ROOT / "outputs" / "eval").mkdir(exist_ok=True)
df.to_csv(ROOT / "outputs" / "eval" / "model_comparison.csv", index=False)
pd.set_option("display.width", 220)
for role in ("target", "val"):
    t = df[df.role == role].pivot_table(index="fold", columns="model",
                                        values=["AP@0.5", "maxrecall@0.1", "maxrecall@0.5"], aggfunc="mean").round(3)
    print(f"\n== {role} (seeds {a.seeds}) ==\n{t.to_string()}")
print("\nmiss floor change (YOLO11m - YOLO11s), target, IoU 0.1 / 0.5:")
t = df[df.role == "target"].pivot_table(index="fold", columns="model", values=["maxrecall@0.1", "maxrecall@0.5"])
for k in ("maxrecall@0.1", "maxrecall@0.5"):
    print(k, (-(t[k]["YOLO11m"] - t[k]["YOLO11s"])).round(3).to_dict(), " (negative = lower floor with YOLO11m)")
