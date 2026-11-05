"""Are multi-component masks one fragmented polyp or several polyps?
For every positive frame with >=2 components (after min_area): gap between components, and how a
merge distance d (components closer than d px become one polyp) changes agreement with the official
bbox count. Writes outputs/fragments_sheet.jpg with overlays for visual review."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "raw" / "PolypGen2021_MultiCenterData_v3"
cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
T, A = cfg["mask_threshold"], cfg["min_area"]
m = pd.read_csv(ROOT / "data" / "manifest.csv")
multi = m[m.n_polyps >= 2].copy()

def comps(mask_path):
    g = cv2.imread(str(R / mask_path), cv2.IMREAD_GRAYSCALE)
    n, lab, st, _ = cv2.connectedComponentsWithStats((g > T).astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= A]
    return np.isin(lab, keep).astype(np.uint8), g.shape

def count_after_merge(binm, d):
    if d == 0:
        k = binm
    else:
        k = cv2.dilate(binm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d + 1, d + 1)))
    n, lab = cv2.connectedComponents(k, connectivity=8)
    return len(set(np.unique(lab[binm > 0])) - {0})

ds = [0, 5, 10, 20, 40, 80]
rows = []
for r in multi.itertuples():
    binm, shape = comps(r.mask_path)
    off = json.loads(r.official_boxes) if isinstance(r.official_boxes, str) and r.official_boxes.startswith("[") else None
    rows.append({"path": r.path, "center": r.center, "seq": r.seq if isinstance(r.seq, str) else "single",
                 "n_off": None if off is None else len(off), "diag": float(np.hypot(*shape)),
                 **{f"n_d{d}": count_after_merge(binm, d) for d in ds}})
f = pd.DataFrame(rows)
f.to_csv(ROOT / "outputs" / "fragments.csv", index=False)
print(f"frames with >=2 components: {len(f)}  (by group:)")
print(f.groupby(["center", "seq"]).size().to_string())
known = f[f.n_off.notna()]
print(f"\nwith official box file: {len(known)}; official count distribution: {known.n_off.value_counts().to_dict()}")
for d in ds:
    eq = (known[f"n_d{d}"] == known.n_off).sum()
    under = (known[f"n_d{d}"] < known.n_off).sum()
    print(f"d={d:3d}px  count==official: {eq:4d}  fewer than official (over-merged real polyps): {under:3d}")

# visual review: worst fragmenters + cases where official says >=2 polyps
pick = pd.concat([f.sort_values("n_d0", ascending=False).head(6),
                  known[known.n_off >= 2].sample(min(6, (known.n_off >= 2).sum()), random_state=0)])
tiles = []
for r in pick.itertuples():
    img = cv2.imread(str(R / r.path))
    mp = m.loc[m.path == r.path, "mask_path"].iloc[0]
    binm, _ = comps(mp)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 4)
    off = m.loc[m.path == r.path, "official_boxes"].iloc[0]
    if isinstance(off, str) and off.startswith("["):
        for x1, y1, x2, y2 in json.loads(off):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 80, 0), 4)
    img = cv2.resize(img, (360, 270))
    cv2.putText(img, f"{r.seq} comp={r.n_d0} off={r.n_off}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(img)
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
cv2.imwrite(str(ROOT / "outputs" / "fragments_sheet.jpg"), grid)
print("wrote outputs/fragments.csv, outputs/fragments_sheet.jpg")
