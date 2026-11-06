"""Train YOLO11 on one LOCO fold, then export ALL predictions (conf 0.001) for val/cal/target.

Kaggle cell:  !python train_fold.py --fold C4 --seed 0 --images-root /kaggle/input/<ds>/PolypGen2021_MultiCenterData_v3 --meta-root /kaggle/input/<ds>
Needs export_yolo.py next to this file or in ../scripts (it is imported).
Outputs in --workdir: preds_{fold}_seed{seed}.parquet (csv.gz if pyarrow missing), run_meta.json,
best_{fold}_seed{seed}.pt, index_{fold}.csv.
"""
import argparse, gc, json, platform, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
from export_yolo import export_fold  # noqa: E402

import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--fold", required=True)
ap.add_argument("--images-root", required=True)
ap.add_argument("--meta-root", default=str(HERE.parent), help="dir with splits/ and manifest.csv (or data/manifest.csv)")
ap.add_argument("--workdir", default="/kaggle/working" if Path("/kaggle/working").is_dir() else "work")
ap.add_argument("--size", default="s", choices=list("nsmlx"))
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--pred-batch", type=int, default=4, help="batch for val/predict (small: the trainer keeps GPU memory)")
ap.add_argument("--skip-train", action="store_true", help="reuse runs/<tag>/weights/best.pt + results.csv from --workdir, no retraining")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking (default: link on Linux)")
ap.add_argument("--smoke", action="store_true", help="1 epoch, 64 train / 16 other images, pipeline test")
a = ap.parse_args()

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

work = Path(a.workdir).resolve()
work.mkdir(parents=True, exist_ok=True)
ds = work / f"yolo_{a.fold}"
tag = f"{a.fold}_seed{a.seed}"
cuda = torch.cuda.is_available()
device = 0 if cuda else "cpu"
if a.smoke:
    a.epochs, a.patience, a.batch, a.workers = 1, 1, 8, 0
    limits = dict(train=64, val=16, cal=16, target=16)
else:
    limits = None

t0 = time.time()
mode = "copy" if (a.copy or platform.system() == "Windows") else "link"
export_fold(a.fold, a.images_root, ds, mode, a.meta_root, limits, a.seed)
t_export = time.time() - t0
idx = pd.read_csv(ds / "index.csv")
shutil.copy(ds / "index.csv", work / f"index_{a.fold}.csv")

best = work / "runs" / tag / "weights" / "best.pt"
if a.skip_train:
    assert best.exists(), f"--skip-train needs {best}"
    t_train = float(pd.read_csv(work / "runs" / tag / "results.csv").rename(columns=str.strip)["time"].iloc[-1])
else:
    model = YOLO(f"yolo11{a.size}.pt")
    t0 = time.time()
    model.train(data=str(ds / "data.yaml"), epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch,
                workers=a.workers, seed=a.seed, amp=True, device=device, project=str(work / "runs"), name=tag,
                exist_ok=True, plots=False, verbose=False)
    t_train = time.time() - t0
    del model
gc.collect()
if cuda:
    torch.cuda.empty_cache()  # the trainer leaves ~12 GB reserved on a 15 GB GPU; free it before predicting
shutil.copy(best, work / f"best_{tag}.pt")

# best epoch = argmax fitness (0.1*mAP50 + 0.9*mAP50-95) in results.csv, as Ultralytics picks best.pt
res = pd.read_csv(work / "runs" / tag / "results.csv")
res.columns = [c.strip() for c in res.columns]
fit = 0.1 * res["metrics/mAP50(B)"] + 0.9 * res["metrics/mAP50-95(B)"]
ep_times = [round(x, 2) for x in res["time"].diff().fillna(res["time"]).tolist()]  # cumulative train+val seconds -> per epoch
for i, x in enumerate(ep_times, 1):
    print(f"[time] epoch {i} {x:.1f}s", flush=True)
best_epoch = int(res["epoch"].iloc[fit.values.argmax()])

model = YOLO(str(best))
v = model.val(data=str(ds / "data.yaml"), split="val", imgsz=a.imgsz, batch=a.pred_batch, workers=a.workers,
              device=device, plots=False, verbose=False)
val_map = dict(mAP50=float(v.box.map50), mAP50_95=float(v.box.map))

rows, t_pred = [], {}
for role in ["val", "cal", "target"]:
    sub = idx[idx.role == role].set_index("name")
    t0 = time.time()
    paths = [str(ds / "images" / role / n) for n in sub.index]
    for r in model.predict(paths, conf=0.001, iou=0.6, max_det=100, imgsz=a.imgsz, batch=a.pred_batch,
                           device=device, half=cuda, stream=True, verbose=False):
        name = Path(r.path).name
        # r.boxes.xyxy is in ORIGINAL image pixels (Ultralytics rescales from the letterboxed input)
        assert tuple(r.orig_shape) == (sub.loc[name, "height"], sub.loc[name, "width"]), name
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()):
            rows.append((sub.loc[name, "orig_path"], role, sub.loc[name, "center"], x1, y1, x2, y2, s))
    t_pred[role] = round(time.time() - t0, 2)
    print(f"[time] predict {role}: {len(sub)} images {t_pred[role]:.1f}s", flush=True)
df = pd.DataFrame(rows, columns=["orig_path", "role", "center", "x1", "y1", "x2", "y2", "score"])
try:
    pred_file = work / f"preds_{tag}.parquet"
    df.to_parquet(pred_file, index=False)
except ImportError:
    pred_file = work / f"preds_{tag}.csv.gz"
    df.to_csv(pred_file, index=False)

meta = dict(fold=a.fold, args=vars(a), versions=dict(python=platform.python_version(), torch=torch.__version__,
            ultralytics=ultralytics.__version__, pandas=pd.__version__),
            gpu=torch.cuda.get_device_name(0) if cuda else None, device=str(device),
            n_images=idx.role.value_counts().to_dict(), n_pred_boxes=df.role.value_counts().to_dict(),
            seconds=dict(export=round(t_export, 1), train_total=round(t_train, 1), epochs=ep_times,
                         mean_epoch=round(sum(ep_times) / max(1, len(ep_times)), 2), predict=t_pred),
            epochs_run=len(res), best_epoch=best_epoch, val=val_map, pred_file=pred_file.name)
for n in ("run_meta.json", f"run_meta_{tag}.json"):  # tagged copy survives running several seeds in one session
    (work / n).write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
