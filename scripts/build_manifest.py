"""Build data/manifest.csv (one row per PolypGen image) and docs/manifest_report.md.
Run: python -I scripts/build_manifest.py   (reads configs/data.yaml)"""
import json, os, collections
import numpy as np, pandas as pd, cv2, yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "PolypGen2021_MultiCenterData_v3")
CFG = yaml.safe_load(open(os.path.join(ROOT, "configs", "data.yaml")))
THR, MIN_AREA = CFG["mask_threshold"], CFG["min_area"]

# Sequence -> center, from docs/data_audit.md ("Sequence -> center" tables). Not stated in filenames.
# seq5 is C2, not C5 as the paper says: decided 2026-11-05, evidence in docs/notes/2026-11-05_seq5-and-fragments.md
POS_CENTER = {**dict.fromkeys([1, 2, 3, 4, 6, 7, 8, 9, 10], "C4"), 11: "C1",
              **dict.fromkeys([5, 12, 13, 14, 15], "C2"), **dict.fromkeys(range(16, 24), "C6")}
NEG_CENTER = {**dict.fromkeys([1, 2, 3, 8, 9, 10], "C4"), **dict.fromkeys([4, 5, 6, 7], "C1"),
              **dict.fromkeys([11, 12, 13], "C5"), **dict.fromkeys(range(14, 24), "C6")}
# data_audit.md "Final frames per center" table: (single, seq_pos, seq_neg)
EXPECTED = {"C1": (256, 228, 430), "C2": (301, 1115, 0), "C3": (457, 0, 0),
            "C4": (227, 450, 1359), "C5": (208, 0, 731), "C6": (88, 432, 1755)}
EXPECTED_TOTAL = 8037


def rel(p): return os.path.relpath(p, RAW).replace("\\", "/")
def jpgs(d): return sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))


def collect():
    """-> list of dicts: img, mdir, bdir, center, source, seq"""
    items = []
    for c in range(1, 7):
        b = os.path.join(RAW, f"data_C{c}")
        for f in jpgs(f"{b}/images_C{c}"):
            items.append(dict(img=f"{b}/images_C{c}/{f}", mdir=f"{b}/masks_C{c}", bdir=f"{b}/bbox_C{c}",
                              center=f"C{c}", source="single", seq=""))
    for k in range(1, 24):
        b = os.path.join(RAW, "sequenceData", "positive", f"seq{k}")
        for f in jpgs(f"{b}/images_seq{k}"):
            items.append(dict(img=f"{b}/images_seq{k}/{f}", mdir=f"{b}/masks_seq{k}", bdir=f"{b}/bbox_seq{k}",
                              center=POS_CENTER[k], source="seq_pos", seq=f"seq{k}"))
        b = os.path.join(RAW, "sequenceData", "negativeOnly", f"seq{k}_neg")
        for f in jpgs(b):
            items.append(dict(img=f"{b}/{f}", mdir=None, bdir=None, center=NEG_CENTER[k],
                              source="seq_neg", seq=f"seq{k}_neg"))
    return items


def find(d, stem, ext):
    if d is None: return None
    # C3 name mismatch: image is "C3_EndoCV2021_00489].jpg" but its mask is "C3_EndoCV2021_00489_mask.jpg"
    for s in (stem + "_mask", stem, stem.rstrip("]") + "_mask"):
        p = f"{d}/{s}{ext}"
        if os.path.exists(p): return p
    return None


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]); ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    i = iw * ih
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)


def match(boxes, off):
    """greedy IoU>=0.5 matching; returns (n_matched, mean matched IoU over max(len))"""
    used, n, tot = set(), 0, 0.0
    for b in boxes:
        best, bj = 0.0, None
        for j, o in enumerate(off):
            if j not in used and iou(b, o) > best: best, bj = iou(b, o), j
        if bj is not None and best >= 0.5: used.add(bj); n += 1; tot += best
    return n, tot / max(len(boxes), len(off), 1)


def gt_rule(boxes, cents, off):
    """Polyps for detection. Official boxes if a bbox file exists (1 official box = 1 polyp; mask fragments inside
    it belong to it); else mask component boxes. Components with centroid outside every official box are appended
    (conservative: separate polyps) and flagged. -> (gt_boxes, flag)"""
    if off is None: return [list(b) for b in boxes], ""
    gt, flag = [list(o) for o in off], ""
    out = [b for b, c in zip(boxes, cents)
           if not any(o[0] <= c[0] <= o[2] and o[1] <= c[1] <= o[3] for o in off)]
    if out: gt += out; flag = "outside_box_component"
    if not off and boxes: flag = "outside_box_component"  # official file lists 0 boxes but mask has components
    return gt, flag


def main():
    rows, areas, notes = [], [], collections.defaultdict(list)
    for it in collect():
        stem = os.path.splitext(os.path.basename(it["img"]))[0]
        with Image.open(it["img"]) as im: w, h = im.size
        mp = find(it["mdir"], stem, ".jpg")
        bp = find(it["bdir"], stem, ".txt")
        boxes, cents = [], []
        if it["source"] != "seq_neg":
            if mp is None: notes["no_mask"].append(rel(it["img"]))
            else:
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m.shape != (h, w): notes["mask_size_mismatch"].append(rel(mp))
                n, _, st, cen = cv2.connectedComponentsWithStats((m > THR).astype(np.uint8), connectivity=8)
                for i in range(1, n):
                    a = int(st[i, 4]); areas.append((a, rel(mp)))
                    if a >= MIN_AREA:
                        x, y, bw, bh = (int(v) for v in st[i, :4]); boxes.append([x, y, x+bw, y+bh]); cents.append(cen[i])
                if (m > THR).any() and not boxes: notes["all_components_dropped"].append(rel(mp))
        off = None
        if bp is not None:
            off = [[int(float(v)) for v in l.split()[1:5]] for l in open(bp).read().splitlines() if l.strip()]
        elif it["source"] != "seq_neg": notes["no_bbox_file"].append(rel(it["img"]))
        gt, flag = gt_rule(boxes, cents, off)
        if flag: notes[flag].append(rel(it["img"]))
        rows.append(dict(path=rel(it["img"]), mask_path=rel(mp) if mp else "", center=it["center"],
                         source=it["source"], seq=it["seq"], width=w, height=h, n_polyps=len(boxes),
                         boxes=json.dumps(boxes), official_boxes=json.dumps(off) if off is not None else "null",
                         has_polyp_mask=len(boxes) > 0, gt_boxes=json.dumps(gt), n_gt=len(gt), flag=flag))
    for r in rows: r["has_polyp"] = r["n_gt"] > 0
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "data", "manifest.csv"), index=False)
    report(df, areas, notes)
    check(df)
    return df


def check(df):
    """Runnable check: fails if row count or per-center totals drift."""
    assert len(df) == EXPECTED_TOTAL, f"rows {len(df)} != {EXPECTED_TOTAL}"
    assert df.path.is_unique
    t = df.groupby(["center", "source"]).size()
    for c, e in EXPECTED.items():
        got = tuple(int(t.get((c, k), 0)) for k in ("single", "seq_pos", "seq_neg"))
        assert got == e, f"{c}: {got} != {e}"
    print("manifest checks OK")


def report(df, areas, notes):
    L = ["# Manifest report", "",
         f"Generated by `scripts/build_manifest.py`. Mask threshold > {THR}, MIN_AREA = {MIN_AREA} (provisional, `configs/data.yaml`). Boxes are [x1,y1,x2,y2] with x2,y2 exclusive (x+w, y+h).", "",
         f"Rows: {len(df)} (expected {EXPECTED_TOTAL}).", "", "## Totals per center x source", "",
         "| center | single | seq_pos | seq_neg | total | data_audit.md (single + pos / neg) |", "|---|---|---|---|---|---|"]
    t = df.groupby(["center", "source"]).size().unstack(fill_value=0)
    for c in sorted(t.index):
        s, p, n = (int(t.loc[c].get(k, 0)) for k in ("single", "seq_pos", "seq_neg"))
        e = EXPECTED[c]; ok = "OK" if (s, p, n) == e else "MISMATCH"
        L.append(f"| {c} | {s} | {p} | {n} | {s+p+n} | {e[0]} + {e[1]} / {e[2]} {ok} |")
    L += [f"| all | {int(t.single.sum())} | {int(t.seq_pos.sum())} | {int(t.seq_neg.sum())} | {len(df)} | |", ""]
    L += [f"## Component areas (pixels; 8-connectivity; mask > {THR}; all components, before MIN_AREA)", ""]
    a = np.array([x[0] for x in areas])
    edges = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000, 10**4, 10**5, 10**9]
    L += ["| area bin | components |", "|---|---|"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        L.append(f"| {lo}-{hi-1} | {int(((a >= lo) & (a < hi)).sum())} |")
    small = sorted(areas)[:20]
    L += ["", f"Total components: {len(a)}. Smallest 20: " + ", ".join(f"{x} ({p})" for x, p in small), ""]
    L += ["MIN_AREA evidence (measured once at threshold 127 with MIN_AREA=1; static text): of 167+143+46+19+27+14+11 = 427 components below 200 px, "
          "all but one are in images that also hold a >=5000 px lesion (JPEG specks next to the lesion); the one exception (area 2-4, `EndoCV2021_001010`) is the C2 image whose CSV says no polyp. "
          "Sole components first appear at 500+ px. Sweep of images whose component count equals the official box count (3057 images with a bbox file): "
          "MIN_AREA 1: 2826, 5: 2909, 20: 2924, 50: 2930, 100: 2928, 200: 2930, 300: 2931, 500: 2928 (and 2 real polyps lost), 1000: 2910 (16 images lose all polyps). "
          "Agreement saturates at ~50 and real small polyps exist from ~220 px (e.g. `EndoCV2021_C5_50000140`), so 50 is chosen (provisional).", ""]
    L += ["## Polyp counts", "",
          f"Images with n_polyps > 1: {int((df.n_polyps > 1).sum())}; n_polyps histogram: {df.n_polyps.value_counts().sort_index().to_dict()}", "",
          f"Images with has_polyp: {int(df.has_polyp_mask.sum())} (mask) / {int(df.has_polyp.sum())} (gt_boxes); single frames with no polyp (mask), per center: {df[(df.source == 'single') & ~df.has_polyp_mask].center.value_counts().sort_index().to_dict()}", ""]
    ng = df.n_gt.sum(); nm = df.n_polyps.sum()
    byc = df.groupby("center").agg(mask=("n_polyps", "sum"), gt=("n_gt", "sum"), pos_mask=("has_polyp_mask", "sum"), pos_gt=("has_polyp", "sum"))
    L += ["## Polyp counting rule (`gt_boxes`, `n_gt`, `has_polyp`, `flag`)", "",
          "`gt_boxes` = official boxes when a bbox file exists (one official box = one polyp; mask fragments inside a box belong to it, snares split masks); mask component boxes when there is no bbox file (64 C3 images); [] for negative sequence frames. "
          "A mask component (>= MIN_AREA) whose centroid lies outside every official box is appended as a separate polyp and the frame gets `flag=outside_box_component` (also set if the official file has 0 boxes but the mask has components). "
          "`boxes`, `n_polyps`, `official_boxes`, `has_polyp_mask` stay mask-derived (segmentation).", "",
          f"Total polyps: mask components {int(nm)} -> gt_boxes {int(ng)}. Positive frames: mask {int(df.has_polyp_mask.sum())} -> gt {int(df.has_polyp.sum())}. n_gt histogram: {df.n_gt.value_counts().sort_index().to_dict()}", "",
          "| center | polyps mask | polyps gt | positive frames mask | positive frames gt |", "|---|---|---|---|---|"]
    for c, r in byc.iterrows(): L.append(f"| {c} | {r['mask']} | {r['gt']} | {r.pos_mask} | {r.pos_gt} |")
    fl = df[df.flag != ""]
    L += ["", f"Flagged frames ({len(fl)}): " + (", ".join(f"`{r.path}` ({r.flag}, n_polyps {r.n_polyps}, official {len(json.loads(r.official_boxes))}, n_gt {r.n_gt})" for r in fl.itertuples()) or "none"), ""]
    sub = df[df.official_boxes != "null"]
    res = []
    for r in sub.itertuples():
        b, o = json.loads(r.boxes), json.loads(r.official_boxes)
        n, q = match(b, o); res.append((r.path, len(b), len(o), n, q))
    res = pd.DataFrame(res, columns=["path", "n_mask", "n_off", "n_match", "quality"])
    full = res.n_match == res[["n_mask", "n_off"]].max(axis=1)
    L += ["## Box agreement (mask-derived `boxes` vs `official_boxes`, IoU >= 0.5)", "",
          f"Images with an official file: {len(res)}. Fully agreeing (equal counts, all matched): {int(full.sum())}. Disagreeing: {int((~full).sum())}.",
          f"Same box count: {int((res.n_mask == res.n_off).sum())}. Matched boxes: {int(res.n_match.sum())} (mask boxes {int(res.n_mask.sum())}, official boxes {int(res.n_off.sum())}).", "",
          "Worst 25 (path: mask boxes / official boxes / matched / mean matched IoU):", ""]
    for r in res[~full].sort_values(["quality", "n_match"]).head(25).itertuples():
        L.append(f"- `{r.path}`: {r.n_mask} / {r.n_off} / {r.n_match} / {r.quality:.2f}")
    L += ["", "## Known issues (data_audit.md)", ""]
    nb = notes["no_bbox_file"]
    c3 = [p for p in nb if p.startswith("data_C3/")]
    L.append(f"- Single/positive-sequence images with no bbox file: {len(nb)} total; in C3: {len(c3)} (of which non-empty mask: {int(df[df.path.isin(c3)].has_polyp_mask.sum())}).")
    L.append("  C3 files: " + ", ".join(f"`{os.path.basename(p)}`" for p in c3))
    L.append(f"  Missing bbox files outside C3: {[p for p in nb if p not in c3] or 'none'}")
    L.append(f"- Images without a mask: {notes['no_mask'] or 'none'}")
    L.append(f"- Mask/image size mismatches: {notes['mask_size_mismatch'] or 'none'}")
    L.append(f"- Masks with pixels > {THR} but every component dropped by MIN_AREA: {notes['all_components_dropped'] or 'none'}")
    c3d = os.path.join(RAW, "data_C3")
    strip = lambda f: os.path.splitext(f)[0].removesuffix("_mask")
    ims = {strip(f) for f in jpgs(f"{c3d}/images_C3")}
    ms = {strip(f) for f in jpgs(f"{c3d}/masks_C3")}
    bs = {strip(f) for f in os.listdir(f"{c3d}/bbox_C3") if f.endswith(".txt")}
    L.append(f"- C3 images with no mask of the same stem: {sorted(ims - ms)}; masks with no image: {sorted(ms - ims)}; bbox files with no image: {sorted(bs - ims)}")
    csv = pd.read_csv(os.path.join(RAW, "dataDetails_PolypGen_SingleFrames", "dataDetails_C2.csv"))
    csv["stem"] = csv.fileList.str.replace("\\", "/").str.rsplit("/", n=1).str[-1].str.rsplit(".", n=1).str[0]
    z = set(csv[csv.annotations == 0].stem)
    d2 = df[(df.center == "C2") & (df.source == "single") & (df.has_polyp_mask | df.mask_path.isin(notes["all_components_dropped"]))]
    d2 = d2[d2.path.map(lambda p: os.path.splitext(os.path.basename(p))[0]).isin(z)]
    L.append(f"- C2 images with CSV annotations = 0 but mask pixels > {THR}: {[(p, m) for p, m in zip(d2.path, d2.mask_path)] or 'none'} (at MIN_AREA={MIN_AREA} it is dropped as a speck, has_polyp_mask=False)")
    L.append("")
    open(os.path.join(ROOT, "docs", "manifest_report.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
