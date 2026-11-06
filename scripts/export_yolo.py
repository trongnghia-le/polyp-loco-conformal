"""Build a YOLO-format dataset for one LOCO fold (read-only on data/raw, manifest, splits).

  python scripts/export_yolo.py --fold C4 --images-root <dir holding data_C1/ ... data_C6/> --out <dir> [--link|--copy]

Layout: <out>/images/{train,val,cal,target}/<name>, <out>/labels/<role>/<stem>.txt, <out>/data.yaml, <out>/index.csv.
name = manifest path with '/' -> '__' (unique across folders). One class: polyp.
Boxes: gt_boxes ([x1,y1,x2,y2], x2/y2 exclusive) clipped to the image, written as normalized cx cy w h.
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import pandas as pd

ROLES = ["train", "val", "cal", "target"]
ROOT = Path(__file__).resolve().parents[1]


def yolo_lines(boxes, w, h):
    lines, dropped = [], 0
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        lines.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    return lines, dropped


def export_fold(fold, images_root, out, mode="copy", meta_root=ROOT, limits=None, seed=0):
    """limits: {role: n} keeps n images per role, half positive / half negative where possible (smoke tests)."""
    meta_root, images_root, out = Path(meta_root), Path(images_root), Path(out)
    mf = meta_root / "data" / "manifest.csv"
    man = pd.read_csv(mf if mf.is_file() else meta_root / "manifest.csv").set_index("path")  # repo or Kaggle layout
    sp = pd.read_csv(meta_root / "splits" / f"fold_{fold}.csv")
    assert set(sp.role) <= set(ROLES) and sp.path.is_unique
    rows, n_dropped = [], 0
    for role in ROLES:
        d = sp[sp.role == role]
        if limits and role in limits and limits[role] < len(d):
            pos = d[man.loc[d.path, "n_gt"].values > 0]
            neg = d.drop(pos.index)
            k = limits[role]
            npos = min(len(pos), k // 2 + max(0, k // 2 - len(neg)))
            d = pd.concat([pos.sample(npos, random_state=seed), neg.sample(k - npos, random_state=seed)])
        (out / "images" / role).mkdir(parents=True, exist_ok=True)
        (out / "labels" / role).mkdir(parents=True, exist_ok=True)
        for r in d.itertuples():
            m = man.loc[r.path]
            name = r.path.replace("/", "__")
            src = (images_root / r.path).resolve()
            if not src.is_file():
                sys.exit(f"missing image: {src}")
            dst = out / "images" / role / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "link":
                os.symlink(src, dst)
            else:
                shutil.copyfile(src, dst)
            lines, dr = yolo_lines(json.loads(m.gt_boxes), int(m.width), int(m.height))
            n_dropped += dr
            (out / "labels" / role / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            rows.append(dict(name=name, orig_path=r.path, center=r.center, cluster_id=r.cluster_id, role=role,
                             width=int(m.width), height=int(m.height), n_gt=int(m.n_gt)))
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    yaml_txt = f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: polyp\n"
    (out / "data.yaml").write_text(yaml_txt)
    print(f"exported {fold}: {len(rows)} images -> {out} ({mode}); boxes dropped after clipping: {n_dropped}")
    return n_dropped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--images-root", required=True, help="dir containing data_C1/.. (PolypGen2021_MultiCenterData_v3)")
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--link", action="store_true", help="symlink images (Linux/Kaggle)")
    g.add_argument("--copy", action="store_true", help="copy images (default; Windows)")
    ap.add_argument("--meta-root", default=str(ROOT), help="dir with splits/ and manifest.csv (or data/manifest.csv)")
    a = ap.parse_args()
    export_fold(a.fold, a.images_root, a.out, "link" if a.link else "copy", a.meta_root)
