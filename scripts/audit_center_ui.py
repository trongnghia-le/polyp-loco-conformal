"""Contact sheet: 4 frames from each group, to compare endoscope UI overlays between sources."""
import io, re, sys, zipfile
from PIL import Image, ImageDraw

zf = zipfile.ZipFile(sys.argv[1])
names = [n for n in zf.namelist() if not n.startswith("__MACOSX") and n.endswith(".jpg")]
groups = {
    "pos seq11 (?)": r"positive/seq11/images_seq11/",
    "C3 single 720x576": r"data_C3/images_C3/",
    "C1 single": r"data_C1/images_C1/",
    "neg seq4 (C1)": r"negativeOnly/seq4_neg/",
    "neg seq6 (C1)": r"negativeOnly/seq6_neg/",
    "neg seq7 (?)": r"negativeOnly/seq7_neg/",
    "neg seq1 (C4)": r"negativeOnly/seq1_neg/",
    "C4 single": r"data_C4/images_C4/",
}
TW, TH, K = 240, 192, 4
sheet = Image.new("RGB", (TW * K + 170, TH * len(groups)), "white")
d = ImageDraw.Draw(sheet)
for row, (label, rx) in enumerate(groups.items()):
    picks = []
    for n in sorted(n for n in names if re.search(rx, n)):
        im = Image.open(io.BytesIO(zf.read(n)))
        if label.startswith("C3") and im.size != (720, 576):
            continue
        picks.append(im)
        if len(picks) == K * 10:
            break
    picks = picks[:: max(1, len(picks) // K)][:K]
    d.text((5, row * TH + TH // 2), label, fill="black")
    for i, im in enumerate(picks):
        sheet.paste(im.convert("RGB").resize((TW, TH)), (170 + i * TW, row * TH))
sheet.save(sys.argv[2])
