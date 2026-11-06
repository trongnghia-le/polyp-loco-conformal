"""Generate kaggle/run_seed{0,1,2}.ipynb: self-contained notebooks, one per seed (6 folds each, ~5-6 h per session;
nothing to edit). They are independent, so one failing never loses the others. Each embeds export_yolo.py and train_fold.py via
%%writefile, so the Kaggle dataset only has to hold data + manifest + splits.
Re-run this script after editing scripts/export_yolo.py or kaggle/train_fold.py."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_USER = "letronghia"  # Kaggle username that owns the private dataset polypgen-loco
export_src = (ROOT / "scripts" / "export_yolo.py").read_text(encoding="utf-8")
train_src = (ROOT / "kaggle" / "train_fold.py").read_text(encoding="utf-8")

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

SETUP = '''import glob, os, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.170"], check=True)
cands = [p for d in ("*", "*/*", "*/*/*") for p in glob.glob(f"/kaggle/input/{d}/manifest.csv")]
assert len(cands) == 1, f"expected one manifest.csv under /kaggle/input, found {cands}"
DS = str(Path(cands[0]).parent)
print("dataset:", DS, os.listdir(DS))
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
'''

RUN = '''# ===== CONFIG (generated; nothing to edit) =====
FOLDS = __FOLDS__
SEEDS = [__SEED__]
ALREADY_DONE = __DONE__          # runs to skip (empty: all 6 folds are trained)
EPOCHS, PATIENCE, SIZE = __EPOCHS__, __PATIENCE__, "__SIZE__"   # fixed for all folds/seeds
STOP_NEW_RUNS_AFTER_H = 8.0          # never start a new run after this many hours (session limit safety)
# ===============================================================
import json, shutil, time
import pandas as pd

W = Path("/kaggle/working"); RES = W / "results"; RES.mkdir(exist_ok=True)
runs = [(f, s) for s in SEEDS for f in FOLDS if (f, s) not in ALREADY_DONE]
t_session = time.time(); fails = 0
for fold, seed in runs:
    tag = f"{fold}_seed{seed}"; d = RES / tag
    if (d / f"run_meta_{tag}.json").exists():
        print("skip (done):", tag); continue
    if (time.time() - t_session) / 3600 > STOP_NEW_RUNS_AFTER_H:
        print("time guard: not starting", tag); break
    d.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(W / "train_fold.py"), "--fold", fold, "--seed", str(seed), "--size", SIZE,
           "--epochs", str(EPOCHS), "--patience", str(PATIENCE),
           "--images-root", f"{DS}/PolypGen2021_MultiCenterData_v3", "--meta-root", DS]
    t0 = time.time(); print(f"=== {tag} start", time.strftime("%H:%M:%S"), flush=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    with open(d / "log.txt", "w") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            lf.write(line)
            if "[time]" in line or "Traceback" in line or "Error" in line:
                print(line.rstrip()[:300], flush=True)
        rc = p.wait()
    ok = rc == 0 and (W / f"run_meta_{tag}.json").exists()
    if ok:
        for pat in (f"preds_{tag}.*", f"run_meta_{tag}.json", f"best_{tag}.pt"):
            for f in W.glob(pat):
                shutil.move(str(f), d / f.name)
        shutil.copy(W / f"index_{fold}.csv", d / f"index_{fold}.csv")
        shutil.copy(W / "runs" / tag / "results.csv", d / "results.csv")
        fails = 0
    else:
        fails += 1
        for f in W.glob(f"best_{tag}.pt"):                      # keep what training produced
            shutil.move(str(f), d / f.name)
        if (W / "runs" / tag / "results.csv").exists():
            shutil.copy(W / "runs" / tag / "results.csv", d / "results.csv")
        print(f"!!! {tag} FAILED (rc={rc}); log: {d/'log.txt'}", flush=True)
    shutil.rmtree(W / "runs" / tag, ignore_errors=True)      # keep /kaggle/working small
    shutil.rmtree(W / f"yolo_{fold}", ignore_errors=True)
    (W / "run_meta.json").unlink(missing_ok=True)
    print(f"=== {tag} {'OK' if ok else 'FAILED'} in {(time.time()-t0)/60:.1f} min "
          f"(session {(time.time()-t_session)/3600:.2f} h)", flush=True)
    if fails >= 2:
        print("two failures in a row: stopping to protect the GPU quota"); break

rows = []
for m in sorted(RES.glob("*/run_meta_*.json")):
    j = json.loads(m.read_text())
    rows.append(dict(fold=j["fold"], seed=j["args"]["seed"], epochs_run=j["epochs_run"], best_epoch=j["best_epoch"],
                     train_min=round(j["seconds"]["train_total"] / 60, 1), val_mAP50=round(j["val"]["mAP50"], 4),
                     val_mAP50_95=round(j["val"]["mAP50_95"], 4), gpu=j["gpu"], boxes=sum(j["n_pred_boxes"].values())))
summ = pd.DataFrame(rows); summ.to_csv(RES / "summary.csv", index=False); print(summ.to_string(index=False))
name = "results_seeds_" + "_".join(map(str, SEEDS))
shutil.make_archive(str(W / name), "zip", RES)
print("download:", W / f"{name}.zip")
'''

def build(seed, folds=("C1", "C2", "C3", "C4", "C5", "C6"), epochs=50, patience=10, name=None, label=None, size="s"):
    name = name or f"run_seed{seed}"
    run = (RUN.replace("__SEED__", str(seed)).replace("__DONE__", "set()").replace("__FOLDS__", repr(list(folds)))
           .replace("__EPOCHS__", str(epochs)).replace("__PATIENCE__", str(patience)).replace("__SIZE__", size))
    title = (f"# LOCO YOLO11 training: {label or f'seed {seed} (all folds)'}\n"
             "Settings: Accelerator = GPU, Internet = On. Then **Save Version > Save & Run All (Commit)**; runs in the "
             f"background. Download `results_seeds_{seed}.zip` from the Output tab into `outputs/kaggle/`.")
    nb = {
        "cells": [
            md(title),
            code(SETUP),
            code("%%writefile /kaggle/working/export_yolo.py\n" + export_src),
            code("%%writefile /kaggle/working/train_fold.py\n" + train_src),
            code(run),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for c in nb["cells"]:  # syntax check every code cell (magic line stripped)
        if c["cell_type"] == "code":
            t = "".join(c["source"])
            ast.parse(t.split("\n", 1)[1] if t.startswith("%%writefile") else t)
    out = ROOT / "kaggle" / f"{name}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(out, f"{out.stat().st_size / 1e3:.0f} KB")
    # ready-to-push folder for the Kaggle CLI (kernels push -p <dir>); outputs/ is git-ignored
    kd = ROOT / "outputs" / "kaggle_kernels" / name.replace("run_", "")
    kd.mkdir(parents=True, exist_ok=True)
    (kd / out.name).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    meta = {"id": f"{KAGGLE_USER}/polypgen-loco-{name.replace('run_', '')}", "title": f"polypgen loco {name.replace('run_', '')}",
            "code_file": out.name, "language": "python", "kernel_type": "notebook", "is_private": "true",
            "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
            "dataset_sources": [f"{KAGGLE_USER}/polypgen-loco"], "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


for seed in (0, 1, 2):
    build(seed)
# 1-epoch test on the fold with the biggest cal set (C2): exercises the full-size prediction path in ~6 min
build(99, folds=("C2",), epochs=1, patience=1, name="run_test", label="1-epoch pipeline test (C2)")
# YOLO11m pilot on the two hardest held-out centres: does a bigger detector lower the miss floor?
build(0, folds=("C1", "C4"), size="m", name="run_mpilot", label="YOLO11m pilot, seed 0 (C1, C4)")
