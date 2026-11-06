"""Fixed LOCO splits -> splits/fold_C{k}.csv, splits/target_draws_C{k}.json, docs/splits.md. See docs/splits.md.
Run: python -I scripts/build_splits.py   (reads data/manifest.csv + data/clusters.csv; filenames only, no images)

Source split: per source center, whole clusters are assigned to train/val/cal (70/10/20 of images) by a seeded
local search over clusters (images and positive-image shares both near 70/10/20; seed 2026 + center). The assignment depends
only on the center, so a center has the same train/val/cal clusters in every fold where it is a source.
Target draws: draw_target() below (seeded per fold and budget)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026
PROP = {"train": 0.70, "val": 0.10, "cal": 0.20}
BUDGETS = (20, 50, 100)
R = 1000
CENTERS = [f"C{i}" for i in range(1, 7)]


def assign_center(sizes, npos, seed, iters=20000):
    """sizes/npos: {cluster_id: n_images / n_positive_images} -> {cluster_id: role}. Whole clusters only.
    Seeded local search (random single-cluster moves, accept if not worse) minimising
    sum_r |image share_r - P_r| + sum_r |positive-image share_r - P_r|, so both images and positive frames follow
    70/10/20 as closely as cluster sizes allow. ponytail: stochastic, not optimal; fine for ~1e2-1e3 clusters."""
    rng = np.random.default_rng(seed)
    ids = sorted(sizes); roles = list(PROP)
    tot = sum(sizes.values()); totp = max(sum(npos.values()), 1)
    out = {c: roles[rng.choice(3, p=list(PROP.values()))] for c in ids}
    def score(o):
        im = dict.fromkeys(roles, 0); ps = dict.fromkeys(roles, 0)
        for c, r in o.items(): im[r] += sizes[c]; ps[r] += npos[c]
        return sum(abs(im[r] / tot - PROP[r]) + abs(ps[r] / totp - PROP[r]) for r in roles)
    cur = score(out)
    for _ in range(iters):
        c = ids[rng.integers(len(ids))]; old = out[c]; out[c] = roles[rng.integers(3)]
        new = score(out)
        if new <= cur: cur = new
        else: out[c] = old
    return out


def draw_target(clusters, pos, budget, n_draws, seed):
    """clusters: list of cluster ids of the target pool; pos: {cluster: positive images}. Each draw shuffles the
    clusters and adds whole clusters until positive images >= budget. -> list of lists of cluster ids."""
    rng = np.random.default_rng(seed); clusters = sorted(clusters); draws = []
    for _ in range(n_draws):
        got, d = 0, []
        for i in rng.permutation(len(clusters)):
            d.append(clusters[i]); got += pos[clusters[i]]
            if got >= budget: break
        assert got >= budget
        draws.append(d)
    return draws


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    m = pd.read_csv(ROOT / "data/manifest.csv")[["path", "center", "has_polyp", "n_gt"]]
    c = pd.read_csv(ROOT / "data/clusters.csv")[["path", "cluster_id"]]
    df = m.merge(c, on="path", validate="1:1"); assert len(df) == 8037
    role_of = {}
    for i, ctr in enumerate(CENTERS):
        g = df[df.center == ctr]
        gc = g.groupby("cluster_id")
        role_of.update(assign_center(gc.size().to_dict(), gc.has_polyp.sum().astype(int).to_dict(), SEED + i))
    df["src_role"] = df.cluster_id.map(role_of)
    (ROOT / "splits").mkdir(exist_ok=True)
    folds, stats = {}, []
    for k, tgt in enumerate(CENTERS):
        f = df.copy(); f["role"] = np.where(f.center == tgt, "target", f.src_role)
        # leakage checks
        assert f.groupby("cluster_id").role.nunique().max() == 1, "cluster in two roles"
        assert not (f[f.center == tgt].role != "target").any() and not (f[f.center != tgt].role == "target").any()
        assert len(f) == 8037 and f.role.notna().all() and f.path.is_unique
        f[["path", "center", "cluster_id", "role"]].to_csv(ROOT / f"splits/fold_{tgt}.csv", index=False)
        folds[tgt] = f
        pool = f[f.center == tgt]; pos = pool.groupby("cluster_id").has_polyp.sum().astype(int).to_dict()
        out = {}
        for b in BUDGETS:
            d = draw_target(list(pos), pos, b, R, SEED * 1000 + k * 10 + BUDGETS.index(b))
            out[str(b)] = d
            n = np.array([sum(pos[x] for x in dr) for dr in d]); kc = np.array([len(dr) for dr in d])
            one = np.mean([max(pos[x] for x in dr) >= b for dr in d])
            stats.append(dict(fold=tgt, budget=b, med_n=float(np.median(n)), med_k=float(np.median(kc)),
                              one_cluster=float(one), med_test_pos=float(sum(pos.values()) - np.median(n))))
        json.dump(out, open(ROOT / f"splits/target_draws_{tgt}.json", "w"), separators=(",", ":"))
    # reproducibility self-check: regenerate one draw set and compare with the file
    chk = json.load(open(ROOT / "splits/target_draws_C4.json"))
    p4 = folds["C4"][folds["C4"].center == "C4"].groupby("cluster_id").has_polyp.sum().astype(int).to_dict()
    assert draw_target(list(p4), p4, 20, 5, SEED * 1000 + 3 * 10) == chk["20"][:5]
    stats = pd.DataFrame(stats)
    write_doc(folds, stats)
    print(stats.to_string())


def write_doc(folds, stats):
    roles = ["train", "val", "cal", "target"]
    L = ["# LOCO splits (`splits/`)", "",
         "Generated by `scripts/build_splits.py` (seed 2026). Filenames only: `splits/fold_C{k}.csv` (`path,center,cluster_id,role`; `path` relative to `PolypGen2021_MultiCenterData_v3/`) and `splits/target_draws_C{k}.json`. No images or masks are redistributed (Synapse terms).", "",
         "## Rules",
         "- Six folds; fold Ck holds out center Ck: all its frames have role `target`.",
         "- The 5 source centers are split by `cluster_id` (`data/clusters.csv`), never within a cluster, per center: train ~70 %, val ~10 % (early stopping / model selection), cal ~20 % (conformal calibration; never used for training or model selection) of that center's images. Assignment: seeded local search (seed 2026 + center index) over whole clusters, minimising the deviation of both the image shares and the positive-image shares from 70/10/20. Because the assignment uses only the center, a center has identical train/val/cal clusters in every fold where it is a source. The `cal` rows of one center are the per-source-center calibration set (filter by `center`).",
         "- Few-shot target draws (`target_draws_C{k}.json` = {budget: [[cluster_ids of draw 1], ...]}, budgets 20/50/100 positive images, R = 1000): `draw_target()` shuffles the target clusters and adds whole clusters until positive images (`has_polyp`, from `gt_boxes`) >= budget; negative-only clusters met on the way are added too (zero positives). The remaining target clusters are the test set of that draw. Seed `2026*1000 + 10*fold_index + budget_index`.",
         "- Asserts in the script: no cluster in two roles within a fold; no target-center frame in a source role and vice versa; 8,037 rows per fold.", ""]
    for tgt, f in folds.items():
        L += [f"## Fold {tgt} (target {tgt})", "", "| role | images | % of source | clusters | positive frames | polyps (`n_gt`) |", "|---|---|---|---|---|---|"]
        src = (f.role != "target").sum()
        for r in roles:
            g = f[f.role == r]
            pct = "" if r == "target" else f"{100 * len(g) / src:.1f}"
            L.append(f"| {r} | {len(g)} | {pct} | {g.cluster_id.nunique()} | {int(g.has_polyp.sum())} | {int(g.n_gt.sum())} |")
        cal = f[f.role == "cal"].groupby("center").agg(images=("path", "size"), clusters=("cluster_id", "nunique"), pos=("has_polyp", "sum"))
        L += ["", "cal per source center (images / clusters / positive frames): " + "; ".join(f"{c} {r.images} / {r.clusters} / {int(r.pos)}" for c, r in cal.iterrows()), ""]
    L += ["## Target recalibration draws (R = 1000 per fold and budget)", "",
          "n = positive images in the drawn calibration clusters (can exceed the budget because whole clusters are added); k = clusters in the draw; `one cluster >= budget` = share of draws in which a single cluster alone holds at least `budget` positive images (effective sample size 1); test pos = positive frames left in the test set (median).", "",
          "| fold | budget | median n | median k | one cluster >= budget | median test pos |", "|---|---|---|---|---|---|"]
    for r in stats.itertuples():
        L.append(f"| {r.fold} | {r.budget} | {r.med_n:.0f} | {r.med_k:.0f} | {100 * r.one_cluster:.1f} % | {r.med_test_pos:.0f} |")
    L += ["", "## Caveats",
          "- C1 and C5 positives come from few clusters (C1: one positive sequence, seq11 = 1 cluster of 228 frames, plus singles grouped by case prefix; C5 positives are singles only now that seq5 is C2). Cal sets and draws for these centers carry few independent units.",
          "- C2 and C3 have no negative sequences: false alarms on negatives cannot be measured there (only a few empty-mask singles). C2 positives are dominated by the large seq5/12-15 clusters.",
          "- C3 is all singleton clusters, and singles of C2/C5/C6 are singletons too: the singleton count is an upper bound on independent videos/patients (`docs/clusters.md`, remaining uncertainty 1). Frames of one video that are not near-duplicates cannot be linked, so val/cal/test splits there may still share videos; optimistic bias is possible.",
          "- C4 has a 448-image cluster (`C4_vn40`); C4 positives sit in few videos, so draws at C4 have small k.",
          "- Cluster sizes make the 70/10/20 proportions approximate; achieved numbers are in the fold tables."]
    (ROOT / "docs/splits.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
