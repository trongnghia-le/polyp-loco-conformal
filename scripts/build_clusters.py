"""Assign every PolypGen image a source-video cluster_id -> data/clusters.csv (see docs/clusters.md).

Rule: (1) name key per pattern (C1 single numeric prefix, C4 `N_endocv2021_{positive|negative}_` video id,
other sequences = their folder, a contiguous clip); (2) merge groups of the same center linked by
near-duplicate frames (32x32 corr >= 0.99 AND mean abs diff <= 3 grey levels at 128px); (3) one explicit
weak link C4 n40 <-> p2 (pair count asserted below); (4) everything else is a singleton.
Source: the zip if present, else the extracted folder (read-only). Runtime ~3 min (decodes all 8037 JPEGs).
Run: python -I scripts/build_clusters.py
"""
import os, re, zipfile, collections
import numpy as np, pandas as pd, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, "data/raw/PolypGen2021_MultiCenterData_v3.zip")
DIR = os.path.join(ROOT, "data/raw/PolypGen2021_MultiCenterData_v3")
OUT = os.path.join(ROOT, "data/clusters.csv")
R_CORR, R_MAD, WEAK_CORR = 0.99, 3.0, 0.97
WEAK_LINKS = [("C4_vn40", "C4_vp2")]  # seq10_neg <-> seq6/seq10/seq3_neg

# sequence folder number -> center (docs/data_audit.md)
POS = {**{k: "C4" for k in (1, 2, 3, 4, 6, 7, 8, 9, 10)}, 5: "C5", 11: "C1",
       **{k: "C2" for k in (12, 13, 14, 15)}, **{k: "C6" for k in range(16, 24)}}
NEG = {**{k: "C4" for k in (1, 2, 3, 8, 9, 10)}, **{k: "C1" for k in (4, 5, 6, 7)},
       **{k: "C5" for k in (11, 12, 13)}, **{k: "C6" for k in range(14, 24)}}
C4_NAME = re.compile(r"(\d+)_endocv2021_(positive|negative)_")


def list_images():
    """-> {path relative to PolypGen2021_MultiCenterData_v3/: loader()} (jpg images only, no masks/bboxes)."""
    if os.path.exists(ZIP):
        zf = zipfile.ZipFile(ZIP)
        src = {n.split("/", 1)[1]: n for n in zf.namelist() if not n.startswith("__MACOSX") and "/" in n}
        load = lambda rel: zf.read(src[rel])
    else:
        src = {os.path.relpath(os.path.join(d, f), DIR).replace("\\", "/"): os.path.join(d, f)
               for d, _, fs in os.walk(DIR) for f in fs}
        load = lambda rel: open(src[rel], "rb").read()
    ok = lambda n: n.lower().endswith(".jpg") and (
        re.match(r"data_C\d/images_C\d/", n) or
        (n.startswith("sequenceData/") and "mask" not in n.lower() and "bbox" not in n.lower()))
    return sorted(n for n in src if ok(n)), load


def center_key(p):
    """-> (center, base group key, evidence code)."""
    q = p.split("/"); f = q[-1]
    if q[0].startswith("data_C"):
        c = "C" + q[0][-1]
        m = re.match(r"(\d+)OLCV1_", f) if c == "C1" else C4_NAME.search(f) if c == "C4" else None
        if m and c == "C1": return c, f"C1_p{m.group(1)}", "name_prefix"
        if m: return c, f"C4_v{m.group(2)[0]}{m.group(1)}", "name_prefix"
        return c, f"{c}_s_{f[:-4]}", "singleton"
    folder = q[2]
    c = (NEG if folder.endswith("_neg") else POS)[int(re.match(r"seq(\d+)", folder).group(1))]
    m = C4_NAME.search(f) if c == "C4" else None
    if m: return c, f"C4_v{m.group(2)[0]}{m.group(1)}", "name_prefix"
    return c, f"{c}_{folder}", "folder"


def main():
    paths, load = list_images()
    assert len(paths) == 8037, len(paths)
    G = np.zeros((len(paths), 128, 128), np.uint8)
    for i, p in enumerate(paths):
        a = cv2.imdecode(np.frombuffer(load(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        G[i] = cv2.resize(a, (128, 128), interpolation=cv2.INTER_AREA)
    F = np.stack([cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).ravel() for g in G]).astype(np.float32)
    F -= F.mean(1, keepdims=True); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-9

    ck = [center_key(p) for p in paths]
    center = np.array([c for c, _, _ in ck]); key = np.array([k for _, k, _ in ck])
    parent = {k: k for k in set(key)}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    merged_by = collections.defaultdict(set)  # key -> {'near_dup','sim_link'}

    C = F @ F.T; np.fill_diagonal(C, -1)
    xc = []  # near-dups across centers (reported, never merged)
    for i, j in zip(*np.where(np.triu(C >= R_CORR, 1))):
        if key[i] == key[j]: continue
        if np.abs(G[i, :, 38:].astype(float) - G[j, :, 38:]).mean() > R_MAD: continue  # skip left UI panel
        if center[i] != center[j]: xc.append((i, j)); continue
        parent[find(key[i])] = find(key[j])
        merged_by[key[i]].add("near_dup"); merged_by[key[j]].add("near_dup")
    # weak evidence: pairs of different named groups (same center) with corr >= 0.97
    cnt = collections.Counter()
    for i, j in zip(*np.where(np.triu(C >= WEAK_CORR, 1))):
        if key[i] != key[j] and center[i] == center[j] and "_s_" not in key[i] + key[j]:
            cnt[tuple(sorted((key[i], key[j])))] += 1
    for a, b in WEAK_LINKS:
        n = cnt[tuple(sorted((a, b)))]
        assert n >= 100, f"weak-link evidence vanished: {a} {b} {n}"
        parent[find(a)] = find(b); merged_by[a].add("sim_link"); merged_by[b].add("sim_link")
    wl = {tuple(sorted(w)) for w in WEAK_LINKS}
    print(f"weak pairs NOT merged (>=30 pairs at corr>={WEAK_CORR}):")
    for k, v in cnt.most_common():
        if v >= 30 and k not in wl: print("  ", k, v)
    print("cross-center near-dup pairs:", len(xc), [(paths[i], paths[j]) for i, j in xc][:12])

    # cluster id = name of the largest base group in each merged set
    size = collections.Counter(key)
    best = {}
    for k in parent:
        r = find(k)
        if r not in best or (size[k], k) > (size[best[r]], best[r]): best[r] = k
    cid = np.array([best[find(k)] for k in key])
    csize = collections.Counter(cid)
    xc_idx = {i for pair in xc for i in pair}
    evidence = []
    for i in range(len(paths)):
        e = ["singleton"] if csize[cid[i]] == 1 else [ck[i][2] if ck[i][2] != "singleton" else "near_dup"]
        e += sorted(x for x in merged_by.get(key[i], []) if x not in e)
        if i in xc_idx: e.append("xcenter_dup")
        evidence.append("+".join(e))
    df = pd.DataFrame({"path": paths, "center": center, "cluster_id": cid, "evidence": evidence})
    df.to_csv(OUT, index=False)

    s = df.groupby("cluster_id").agg(center=("center", "first"), n=("path", "size"))
    for c, g in s.groupby("center"):
        print(c, "clusters", len(g), "images", g.n.sum(), "singletons", (g.n == 1).sum(),
              "size min/med/max", g.n.min(), int(g.n.median()), g.n.max())
    print("size bins:", dict(sorted(collections.Counter(pd.cut(s.n, [0, 1, 5, 20, 50, 100, 300, 10**6]).astype(str)).items())))
    print(df.evidence.value_counts().to_string())

    # ---- check: fails if a cluster spans two centers or the row count is wrong
    chk = pd.read_csv(OUT)
    assert len(chk) == 8037 and chk.path.is_unique, len(chk)
    assert (chk.groupby("cluster_id").center.nunique() == 1).all(), "cluster spans two centers"
    assert not chk.isna().any().any()
    print("checks passed")


if __name__ == "__main__":
    main()
