"""Image resolutions per centre (single frames) and per sequence, read from JPEG headers inside the zip."""
import io, re, sys, zipfile
from collections import Counter, defaultdict
from PIL import Image

PATTERNS = [
    (re.compile(r"data_C(\d)/images_C\d/"), "C{}"),
    (re.compile(r"positive/seq(\d+)/images_seq\d+/"), "pos_seq{}"),
    (re.compile(r"negativeOnly/seq(\d+)_neg/"), "neg_seq{}"),
]
zf = zipfile.ZipFile(sys.argv[1])
groups = defaultdict(Counter)
for n in zf.namelist():
    if n.startswith("__MACOSX") or not n.endswith(".jpg"):
        continue
    for rx, fmt in PATTERNS:
        m = rx.search(n)
        if m:
            groups[fmt.format(m.group(1))][Image.open(io.BytesIO(zf.read(n))).size] += 1
            break
key = lambda k: (k[:3], int(re.sub(r"\D", "", k)))
for g in sorted(groups, key=key):
    print(f"{g:10s} n={sum(groups[g].values()):4d}  " + ", ".join(f"{w}x{h}:{c}" for (w, h), c in groups[g].most_common(4)))
