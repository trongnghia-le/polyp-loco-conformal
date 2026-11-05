"""Deeper check of seq5's source centre.
1) JPEG quantization-table fingerprint per group (encoder pipeline signature).
2) Side-by-side full-res crops of the 3 C2-single / seq5 duplicate pairs.
3) Top-left overlay crops: seq5 vs C2 singles vs C2 sequences vs C5 singles vs C5 negative seqs."""
from collections import Counter
from pathlib import Path
import hashlib
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data" / "raw" / "PolypGen2021_MultiCenterData_v3"
groups = {
    "seq5 (paper: C5)": R / "sequenceData/positive/seq5/images_seq5",
    "C2 singles": R / "data_C2/images_C2",
    "C2 seq12": R / "sequenceData/positive/seq12/images_seq12",
    "C2 seq13": R / "sequenceData/positive/seq13/images_seq13",
    "C2 seq14": R / "sequenceData/positive/seq14/images_seq14",
    "C5 singles": R / "data_C5/images_C5",
    "C5 neg seq11": R / "sequenceData/negativeOnly/seq11_neg",
    "C1 seq11": R / "sequenceData/positive/seq11/images_seq11",
    "C4 seq1": R / "sequenceData/positive/seq1/images_seq1",
    "C6 seq16": R / "sequenceData/positive/seq16/images_seq16",
}

def qfp(p):
    q = Image.open(p).quantization or {}
    return hashlib.md5(repr(sorted(q.items())).encode()).hexdigest()[:8]

print("JPEG quantization fingerprints (fingerprint:count) and image sizes")
for g, d in groups.items():
    files = sorted(d.glob("*.jpg"))
    fp = Counter(qfp(p) for p in files)
    sz = Counter(Image.open(p).size for p in files)
    print(f"  {g:18s} n={len(files):4d}  q={dict(fp.most_common(4))}  size={dict(sz.most_common(3))}")

out = ROOT / "outputs"
out.mkdir(exist_ok=True)
pairs = [("EndoCV2021_001351.jpg", "EndoCV2021_seq5_12.jpg"),
         ("EndoCV2021_001352.jpg", "EndoCV2021_seq5_92.jpg"),
         ("EndoCV2021_001353.jpg", "EndoCV2021_seq5_200.jpg")]
W, H = 480, 355
sheet = Image.new("RGB", (W * 2, H * 3), "white")
for r, (a, b) in enumerate(pairs):
    sheet.paste(Image.open(groups["C2 singles"] / a).convert("RGB").resize((W, H)), (0, r * H))
    sheet.paste(Image.open(groups["seq5 (paper: C5)"] / b).convert("RGB").resize((W, H)), (W, r * H))
sheet.save(out / "seq5_pairs.jpg")

crops = []
for g in ["seq5 (paper: C5)", "C2 singles", "C2 seq13", "C5 singles", "C5 neg seq11"]:
    files = sorted(groups[g].glob("*.jpg"))
    if g == "C2 singles":
        files = [p for p in files if Image.open(p).size == (1440, 1064)]
    im = Image.open(files[len(files) // 2]).convert("RGB")
    w, h = im.size
    crops.append(im.crop((0, 0, int(w * 0.28), int(h * 0.45))).resize((300, 300)))
strip = Image.new("RGB", (300 * len(crops), 300), "white")
for i, c in enumerate(crops):
    strip.paste(c, (i * 300, 0))
strip.save(out / "seq5_overlay_crops.jpg")
print("wrote outputs/seq5_pairs.jpg, outputs/seq5_overlay_crops.jpg")
