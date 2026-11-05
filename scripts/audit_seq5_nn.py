"""For every single frame of C2 and C5: nearest frame among all 23 positive sequences
(downscaled grey, mean abs diff). Shows which sequences share video with which centre's singles."""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "raw" / "PolypGen2021_MultiCenterData_v3"

def v(p):
    return np.asarray(Image.open(p).convert("L").resize((64, 48)), dtype=np.float32).ravel()

seq_files, seq_lab = [], []
for k in range(1, 24):
    for p in sorted((R / f"sequenceData/positive/seq{k}/images_seq{k}").glob("*.jpg")):
        seq_files.append(v(p)); seq_lab.append((f"seq{k}", p.name))
S = np.stack(seq_files)

for c in ["C2", "C5", "C1", "C3", "C4", "C6"]:
    hits = {}
    for p in sorted((R / f"data_{c}/images_{c}").glob("*.jpg")):
        d = np.abs(S - v(p)).mean(axis=1)
        i = int(d.argmin())
        if d[i] < 6:  # near-identical frame (same video moment)
            hits.setdefault(seq_lab[i][0], []).append((p.name, seq_lab[i][1], round(float(d[i]), 2)))
    print(f"{c} singles with a near-identical sequence frame (MAD<6):")
    for s, lst in sorted(hits.items()):
        print(f"   {s}: {len(lst)}  e.g. {lst[:3]}")
    if not hits:
        print("   none")
