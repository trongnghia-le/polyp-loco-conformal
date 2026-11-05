"""Compare mask content, bbox files and CSV 'annotations' per center, read straight from the zip."""
import io, re, sys, zipfile, csv
import numpy as np
from PIL import Image

zf = zipfile.ZipFile(sys.argv[1])
root = "PolypGen2021_MultiCenterData_v3/"
names = [n for n in zf.namelist() if n.startswith(root)]
stem = lambda p: re.sub(r"_mask$", "", p.rsplit("/", 1)[-1].rsplit(".", 1)[0])

for c in range(1, 7):
    imgs = {stem(n) for n in names if f"data_C{c}/images_C{c}/" in n and n.endswith(".jpg")}
    masks = {stem(n): n for n in names if f"data_C{c}/masks_C{c}/" in n and n.endswith(".jpg")}
    bbox = {stem(n): n for n in names if f"data_C{c}/bbox_C{c}/" in n and n.endswith(".txt")}
    empty_mask = {k for k, n in masks.items() if np.asarray(Image.open(io.BytesIO(zf.read(n))).convert("L")).max() <= 127}
    nobox = imgs - set(bbox)
    zero_lines = {k for k, n in bbox.items() if not zf.read(n).strip()}
    csvn = [n for n in names if n.endswith(f"dataDetails_C{c}.csv")]
    csv0 = set()
    if csvn:
        for r in csv.DictReader(io.StringIO(zf.read(csvn[0]).decode())):
            if r["annotations"].strip() == "0":
                csv0.add(stem(r["fileList"]))
    print(f"C{c}: images={len(imgs)} masks={len(masks)} img-without-mask={len(imgs-set(masks))} "
          f"empty_masks={len(empty_mask)} no_bbox_file={len(nobox)} empty_bbox_file={len(zero_lines)} "
          f"csv_ann0={len(csv0)} | empty_mask&no_bbox={len(empty_mask & nobox)} empty_mask&csv0={len(empty_mask & csv0)} "
          f"csv0_but_mask_nonempty={len(csv0 - empty_mask)}")
