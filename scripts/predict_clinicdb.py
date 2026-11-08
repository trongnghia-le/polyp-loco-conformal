"""CPU inference of the 18 PolypGen-trained YOLO11s detectors on all 612 CVC-ClinicDB frames (never seen in training).
Prediction settings identical to kaggle/train_fold.py: imgsz 640, conf 0.001, NMS iou 0.6, max_det 100, small batches.
Boxes are in ORIGINAL pixels (Ultralytics rescales from the letterbox); orig_shape must be (288, 384).

Usage: .venv/Scripts/python.exe -I scripts/predict_clinicdb.py [--batch 8] [--threads 8]
Writes outputs/clinicdb/preds_<fold>_seed<k>.parquet (frame_id, sequence_id, x1, y1, x2, y2, score); resumable
(existing files are skipped). Timing per model is printed ([time] lines; the run log is outputs/clinicdb/predict.log)."""
import argparse
import time
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "cvc-clinicdb"

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
torch.set_num_threads(a.threads)

meta = pd.read_csv(DATA / "metadata.csv").sort_values("frame_id").reset_index(drop=True)
assert len(meta) == 612 and meta.frame_id.tolist() == list(range(1, 613)) and meta.sequence_id.nunique() == 29
seq = dict(zip(meta.frame_id, meta.sequence_id))
paths = [str(DATA / p) for p in meta.png_image_path]
out = ROOT / "outputs" / "clinicdb"; out.mkdir(parents=True, exist_ok=True)
t_all = time.time()
for w in sorted((ROOT / "outputs" / "kaggle").glob("C*_seed*/best_C*_seed*.pt")):
    run = w.stem.removeprefix("best_")
    dst = out / f"preds_{run}.parquet"
    if dst.exists():
        print(f"skip {run} (exists)", flush=True); continue
    model, rows, t0 = YOLO(str(w)), [], time.time()
    for i in range(0, len(paths), a.batch):
        for r in model.predict(paths[i:i + a.batch], conf=0.001, iou=0.6, max_det=100, imgsz=640, device="cpu",
                               verbose=False):  # fp32 (default); `half` is deprecated in this ultralytics
            assert tuple(r.orig_shape) == (288, 384), (r.path, r.orig_shape)
            fid = int(Path(r.path).stem)
            for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                rows.append((fid, seq[fid], x1, y1, x2, y2, s))
    df = pd.DataFrame(rows, columns=["frame_id", "sequence_id", "x1", "y1", "x2", "y2", "score"])
    df.to_parquet(dst, index=False)
    dt = time.time() - t0
    print(f"[time] {run}: {len(df)} boxes on {df.frame_id.nunique()}/612 frames, {dt:.1f}s ({dt / len(paths):.3f} s/frame)",
          flush=True)
print(f"[time] total {time.time() - t_all:.1f}s", flush=True)
