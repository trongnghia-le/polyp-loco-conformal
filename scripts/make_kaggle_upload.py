"""Build outputs/kaggle_upload/polypgen-loco/ (+ .zip) = exactly what the private Kaggle dataset needs:
the 8,037 images referenced by data/manifest.csv (no masks), manifest.csv, splits/fold_C*.csv, code/.

Kaggle rejects some characters in file names, so image paths are sanitized in the UPLOAD COPIES ONLY
(raw data and the repo manifest/splits are untouched). Known cases (2026-11-06):
  data_C3/images_C3/C3_EndoCV2021_00489].jpg   -> C3_EndoCV2021_00489.jpg      (stray ']')
  data_C4/images_C4/31_endocv2021_positive_329 2.jpg -> ..._positive_3292.jpg  (space)
rename_map.csv (in the upload) lists every changed path. Predictions come back with the sanitized
paths (`orig_path` in the parquet): when joining them with the repo manifest/splits locally, map them
back with rename_map.csv (columns path = original, path_clean = uploaded)."""
import re
import shutil
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "PolypGen2021_MultiCenterData_v3"
OUT = ROOT / "outputs" / "kaggle_upload" / "polypgen-loco"
if OUT.exists():
    shutil.rmtree(OUT)

def clean(p: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./-]", "", p)

m = pd.read_csv(ROOT / "data" / "manifest.csv")
m["path_clean"] = m.path.map(clean)
assert m.path_clean.is_unique, "sanitizing created a name collision"
rename = m.loc[m.path != m.path_clean, ["path", "path_clean"]]

for p, pc in zip(m.path, m.path_clean):
    dst = OUT / "PolypGen2021_MultiCenterData_v3" / pc
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC / p, dst)

up = m.drop(columns="path").rename(columns={"path_clean": "path"})
up = up[["path"] + [c for c in up.columns if c != "path"]]
up.to_csv(OUT / "manifest.csv", index=False)
rename.to_csv(OUT / "rename_map.csv", index=False)

(OUT / "splits").mkdir(exist_ok=True)
for f in sorted((ROOT / "splits").glob("fold_C*.csv")):
    s = pd.read_csv(f)
    s["path"] = s.path.map(clean)
    assert s.path.isin(set(m.path_clean)).all()
    s.to_csv(OUT / "splits" / f.name, index=False)

(OUT / "code").mkdir(exist_ok=True)
for f in [ROOT / "scripts" / "export_yolo.py", ROOT / "kaggle" / "train_fold.py"]:
    shutil.copy2(f, OUT / "code" / f.name)

# every uploaded name must be Kaggle-safe
names = [str(f.relative_to(OUT)).replace("\\", "/") for f in OUT.rglob("*") if f.is_file()]
bad = [n for n in names if re.search(r"[^A-Za-z0-9_./-]", n)]
assert not bad, bad[:5]
n_img = sum(1 for n in names if n.endswith(".jpg"))
assert n_img == len(m) == 8037, n_img
print(f"{OUT}: {n_img} images, {len(rename)} renamed: {rename.to_dict('records')}")

# Zip with forward slashes (PowerShell's zip writes backslashes, which Kaggle rejects)
zp = OUT.parent / "polypgen-loco.zip"
zp.unlink(missing_ok=True)
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(OUT).as_posix())
assert not any(re.search(r"[^A-Za-z0-9_./-]", n) for n in zipfile.ZipFile(zp).namelist())
print(f"{zp}: {zp.stat().st_size / 1e9:.2f} GB")
