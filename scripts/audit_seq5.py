"""Check whether positive seq5 (mapped to C5 by the paper) is really C2 video:
pixel comparison with the C2 singles flagged as duplicates + a contact sheet vs C2/C5 singles."""
from pathlib import Path
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1] / "data" / "raw" / "PolypGen2021_MultiCenterData_v3"
s5 = R / "sequenceData/positive/seq5/images_seq5"
c2 = R / "data_C2/images_C2"
c5 = R / "data_C5/images_C5"

def g(p, size=(256, 192)):
    return np.asarray(Image.open(p).convert("L").resize(size), dtype=np.float32)

seq5 = sorted(s5.glob("*.jpg"))
seq5_px = [(p, g(p)) for p in seq5]
for name in ["EndoCV2021_001351.jpg", "EndoCV2021_001352.jpg", "EndoCV2021_001353.jpg"]:
    a = g(c2 / name)
    best = min(seq5_px, key=lambda t: np.abs(t[1] - a).mean())
    print(f"{name}: closest seq5 frame {best[0].name}, mean abs diff {np.abs(best[1] - a).mean():.2f} grey levels, "
          f"size {Image.open(c2 / name).size} vs {Image.open(best[0]).size}")

# contact sheet: seq5 / C2 singles at 1440x1064 / C5 singles
rows = [("seq5", seq5[::60][:4]),
        ("C2 1440x1064", [p for p in sorted(c2.glob("*.jpg")) if Image.open(p).size == (1440, 1064)][::40][:4]),
        ("C5 singles", sorted(c5.glob("*.jpg"))[::50][:4])]
W, H = 240, 180
sheet = Image.new("RGB", (W * 4, H * 3), "white")
for r, (_, ps) in enumerate(rows):
    for i, p in enumerate(ps):
        sheet.paste(Image.open(p).convert("RGB").resize((W, H)), (i * W, r * H))
out = Path(__file__).resolve().parents[1] / "outputs" / "seq5_sheet.jpg"
out.parent.mkdir(exist_ok=True)
sheet.save(out)
print("sheet:", out)
