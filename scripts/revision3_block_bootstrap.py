"""Round-3 M4: are the target-video intervals and the recalibration results robust to a coarser grouping of the
unlinked single frames? Stored YOLO11s predictions only (no training), IoU 0.1, alpha 0.10.
Note: docs/notes/2026-11-09_block-bootstrap.md.

Groupings of the held-out centre's positive frames (frames of sequence folders keep their cluster):
  paper  : the video clusters of data/clusters.csv (as in the paper).
  idx10  : unlinked single frames grouped into blocks of 10 consecutive file-name indices (id // 10).
  runs   : single frames grouped into runs of consecutive indices, split where the index jumps by more than 5.
A block that shares a frame with a cluster is merged with it (connected components), so no cluster is split.

Part 1, target bootstrap (folds C2, C3, C5, C6): thresholds of revision2_R6a_weighted.csv held fixed (frame: equal,
  balanced_conservative; cluster: equal, balanced_plugin), groups resampled with replacement (B = 1000), seed drawn per
  resample; frame-averaged (risk_fr) and paper-video-averaged (risk_cl) miss rate. The estimand is unchanged; only the
  resampling unit is coarser. 'paper' uses the exact resamples of revision2_balanced.py and is asserted to reproduce
  its intervals.
Part 2, recalibration with one frame x k units (folds C3, C5; k = 10, 20, capped at n_units - MIN_TEST): units are the
  groups; k units drawn uniformly, one random positive frame each, image-level CRC, test on the frames of the undrawn
  units, so no block sits on both sides. Outcomes on the unit-averaged miss rate (the quantity the design certifies).
  'paper' uses the draws of conformal_cluster.py and is asserted to reproduce fs_b and revision_R4 outcomes.
Usage: .venv/Scripts/python.exe -I scripts/revision3_block_bootstrap.py -> outputs/eval/revision3_block_bootstrap.csv,
       outputs/eval/revision3_block_recal.csv"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conformal import clean, G, load_run, crc_idx  # noqa: E402
from conformal_cluster import MIN_TEST  # noqa: E402
from revision_analyses import B, DRAWS, interval  # noqa: E402

EV = ROOT / "outputs" / "eval"
T, ALPHA = 0.1, 0.10
BOOT_FOLDS, RECAL_FOLDS, KS = ("C2", "C3", "C5", "C6"), ("C3", "C5"), (10, 20)
VARIANTS = (("frame", "equal"), ("frame", "balanced_conservative"), ("cluster", "equal"), ("cluster", "balanced_plugin"))
GROUPINGS = ("paper", "idx10", "runs")


def groups_of(fold, idx, ci, tgtP, pos, src):
    """-> {grouping: group label (0..n-1) for each positive target cluster in tgtP order}."""
    tp = np.flatnonzero((idx.role == "target").to_numpy() & pos)
    single = np.array([src[p] == "single" for p in idx.orig_path.iloc[tp]])
    fid = idx.orig_path.iloc[tp].str.extract(r"(\d+)\D*\.jpg$")[0].astype(int).to_numpy()
    pos_in = {c: i for i, c in enumerate(tgtP)}
    cl = np.array([pos_in[c] for c in ci[tp]])
    out = {"paper": np.arange(len(tgtP))}
    o = np.unique(fid[single])
    run = dict(zip(o, np.r_[0, np.cumsum(np.diff(o) > 5)]))
    for name, blk in (("idx10", fid[single] // 10), ("runs", np.array([run[i] for i in fid[single]]))):
        _, b = np.unique(blk, return_inverse=True)
        n = len(tgtP) + b.max() + 1
        A = coo_matrix((np.ones(single.sum()), (cl[single], len(tgtP) + b)), shape=(n, n))
        _, lab = connected_components(A, directed=False)
        out[name] = np.unique(lab[:len(tgtP)], return_inverse=True)[1]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    GT = {clean(p): np.asarray(json.loads(gb), float).reshape(-1, 4) for p, gb in zip(man.path, man.gt_boxes)}
    src = {clean(p): s for p, s in zip(man.path, man.source)}
    R6 = pd.read_csv(EV / "revision2_R6a_weighted.csv", dtype={"seed": str})
    R6 = R6[(R6.iou == T) & (R6.alpha == ALPHA)]
    cc = pd.read_csv(EV / "conformal_cluster_summary.csv", dtype={"seed": str})
    R4 = pd.read_csv(EV / "revision_R4_degenerate_fa.csv")
    boot_rows, rec_rows, n_assert = [], [], 0

    for fold in sorted(set(BOOT_FOLDS) | set(RECAL_FOLDS)):
        fi = int(fold[1:])
        per_seed, rec = {}, []
        for seed in (0, 1, 2):
            d = ROOT / "outputs" / "kaggle" / f"{fold}_seed{seed}"
            idx, K, N, L, _ = load_run(d, fold, GT, np.random.default_rng([2026, fi, seed]))
            tgt, pos = (idx.role == "target").to_numpy(), K > 0
            ucl, ci = np.unique(idx.cluster_id.to_numpy(), return_inverse=True); C = len(ucl)
            m = np.bincount(ci, weights=pos, minlength=C)
            A = np.zeros((C, len(idx))); A[ci[pos], np.flatnonzero(pos)] = 1.0 / m[ci[pos]]
            Lc = A @ L[T]
            c_tgt = np.bincount(ci, weights=tgt, minlength=C) > 0
            tgtP = np.flatnonzero(c_tgt & (m > 0)); nT, mT = len(tgtP), m[tgtP]
            grp = groups_of(fold, idx, ci, tgtP, pos, src)
            if seed == 0:
                print(f"{fold}: positive target frames {int(mT.sum())}, groups "
                      f"{ {g: int(v.max() + 1) for g, v in grp.items()} }", flush=True)
            # ---- part 1: per-group loss sums at each fixed threshold
            for lev, var in VARIANTS:
                r = R6[(R6.fold == fold) & (R6.seed == str(seed)) & (R6.level == lev) & (R6.variant == var)]
                assert len(r) == 1
                j = int(np.rint(r.lam.iloc[0] * 1000)) - 1
                assert G[j] == r.lam.iloc[0]
                assert abs((mT * Lc[tgtP, j]).sum() / mT.sum() - r.risk_fr.iloc[0]) < 1e-12
                assert abs(Lc[tgtP, j].mean() - r.risk_cl.iloc[0]) < 1e-12
                per_seed[(lev, var, seed)] = Lc[tgtP, j]
            per_seed[("m", seed)] = mT
            # ---- part 2: recalibration with one frame per unit
            if fold in RECAL_FOLDS:
                pf_c = {c: np.flatnonzero(pos & (ci == c)) for c in tgtP}
                for gname in GROUPINGS:
                    g = grp[gname]; nU = g.max() + 1
                    unit_frames = [np.concatenate([pf_c[tgtP[i]] for i in np.flatnonzero(g == u)]) for u in range(nU)]
                    fr_unit = np.full(len(idx), -1); fr_clus = np.full(len(idx), -1)
                    for u, f in enumerate(unit_frames):
                        fr_unit[f] = u
                    for i, c in enumerate(tgtP):
                        fr_clus[pf_c[c]] = i
                    tp = np.flatnonzero(fr_unit >= 0)
                    for k in KS:
                        ke = min(k, nU - MIN_TEST)
                        if ke < 10:
                            rec_rows.append(dict(fold=fold, grouping=gname, n_units=nU, k=k, k_eff=ke, seed=str(seed),
                                                 feasible=False))
                            continue
                        if gname == "paper":  # exactly conformal_cluster.py
                            r1, r2 = np.random.default_rng([2026, fi, k, 11]), np.random.default_rng([2026, fi, k, 12])
                            sel = np.array([r1.choice(tgtP, ke, replace=False) for _ in range(DRAWS)])
                            sel = np.searchsorted(tgtP, sel)
                        else:
                            gi = GROUPINGS.index(gname)
                            r1, r2 = (np.random.default_rng([2026, fi, k, 11, gi]),
                                      np.random.default_rng([2026, fi, k, 12, gi]))
                            sel = np.array([r1.choice(nU, ke, replace=False) for _ in range(DRAWS)])
                        S = np.zeros((DRAWS, G.size))
                        for r in range(DRAWS):
                            S[r] = L[T][[r2.choice(unit_frames[u]) for u in sel[r]]].sum(0)
                        jj, dg = crc_idx(S, np.full(DRAWS, float(ke)), ALPHA)
                        drawn = np.zeros((DRAWS, nU), bool); np.put_along_axis(drawn, sel, True, 1)
                        Lfr = L[T][tp][:, jj].T  # (DRAWS, positive target frames)
                        test = ~drawn[:, fr_unit[tp]]
                        risk_fr = (test * Lfr).sum(1) / test.sum(1)
                        # unit-averaged: mean over test units of the mean frame loss in the unit
                        U1 = np.zeros((nU, len(tp))); U1[fr_unit[tp], np.arange(len(tp))] = 1
                        U1 /= U1.sum(1, keepdims=True)
                        risk_u = ((~drawn) * (Lfr @ U1.T)).sum(1) / (~drawn).sum(1)
                        Cm = np.zeros((nT, len(tp))); Cm[fr_clus[tp], np.arange(len(tp))] = 1
                        Cm /= Cm.sum(1, keepdims=True)
                        ctest = (Cm @ test.T).T > 0  # paper clusters in the test set
                        risk_cl = (ctest * (Lfr @ Cm.T)).sum(1) / ctest.sum(1)
                        if gname == "paper":
                            ref = cc[(cc.fold == fold) & (cc.seed == str(seed)) & (cc.iou == T) & (cc.alpha == ALPHA)
                                     & (cc.strategy == "fs_b") & (cc.budget == k)].iloc[0]
                            assert abs(risk_cl.mean() - ref.risk_cl_mean) < 1e-12, (fold, seed, k)
                            assert abs(risk_fr.mean() - ref.risk_fr_mean) < 1e-12
                            assert abs(dg.mean() - ref.degenerate_frac) < 1e-12
                            assert np.allclose(risk_u, risk_cl); n_assert += 1
                        rec.append(pd.DataFrame(dict(fold=fold, grouping=gname, n_units=nU, k=k, k_eff=ke, seed=seed,
                                                     degenerate=dg, risk_u=risk_u, risk_fr=risk_fr, risk_cl=risk_cl,
                                                     n_test_units=(~drawn).sum(1))))
        # ---- part 1 bootstrap (fold level, after all seeds are loaded)
        if fold in BOOT_FOLDS:
            # grp: groups from the last seed's index; the index (and so the grouping) is the same for every seed
            pick_s = np.random.default_rng([2026, fi, 5152]).integers(0, 3, B)
            for gname in GROUPINGS:
                g = grp[gname]; nG = g.max() + 1
                if gname == "paper":
                    U = np.random.default_rng([2026, fi, 5150]).integers(0, nT, (B, nT))
                else:
                    U = np.random.default_rng([2026, fi, 5160, GROUPINGS.index(gname)]).integers(0, nG, (B, nG))
                Cnt = np.zeros((B, nG)); np.add.at(Cnt, (np.repeat(np.arange(B), U.shape[1]), U.ravel()), 1)
                for lev, var in VARIANTS:
                    for met in ("risk_fr", "risk_cl"):
                        pts, bs = [], []
                        for seed in (0, 1, 2):
                            Lv, mm = per_seed[(lev, var, seed)], per_seed[("m", seed)]
                            num = np.bincount(g, weights=Lv * mm if met == "risk_fr" else Lv, minlength=nG)
                            den = np.bincount(g, weights=mm if met == "risk_fr" else np.ones(nT), minlength=nG)
                            pts.append(num.sum() / den.sum()); bs.append(Cnt @ num / (Cnt @ den))
                            boot_rows.append(dict(fold=fold, grouping=gname, n_groups=nG, level=lev, variant=var,
                                                  metric=met, seed=str(seed), **interval(pts[-1], bs[-1], ALPHA)))
                        mix = np.array([bs[i][b] for b, i in enumerate(pick_s)])
                        boot_rows.append(dict(fold=fold, grouping=gname, n_groups=nG, level=lev, variant=var, metric=met,
                                              seed="all", **interval(np.mean(pts), mix, ALPHA)))
        if rec:
            rl = pd.concat(rec, ignore_index=True)
            for (gname, k), h in rl.groupby(["grouping", "k"]):
                for s, hh in [("all", h)] + [(str(x), y) for x, y in h.groupby("seed")]:
                    nd = ~hh.degenerate.to_numpy()
                    rec_rows.append(dict(fold=fold, grouping=gname, n_units=hh.n_units.iloc[0], k=k,
                                         k_eff=hh.k_eff.iloc[0], seed=s, feasible=True, n_draws=len(hh),
                                         cannot_certify=1 - nd.mean(),
                                         held=(nd & (hh.risk_u <= ALPHA).to_numpy()).mean(),
                                         violated=(nd & (hh.risk_u > ALPHA).to_numpy()).mean(),
                                         risk_u=hh.risk_u.mean(), risk_fr=hh.risk_fr.mean(), risk_cl=hh.risk_cl.mean(),
                                         viol_fr=(hh.risk_fr > ALPHA).mean(),
                                         risk_u_certified=hh.risk_u[nd].mean(),
                                         n_test_units_median=hh.n_test_units.median()))
                if gname == "paper":  # three-outcome shares of revision_R4 (seed-pooled)
                    ref = R4[(R4.fold == fold) & (R4.iou == T) & (R4.alpha == ALPHA) & (R4.strategy == "fs_b")
                             & (R4.budget == k)].iloc[0]
                    r = rec_rows[-4]
                    assert r["seed"] == "all"
                    for a_, b_ in (("cannot_certify", "outcome_cannot_certify"), ("held", "outcome_held"),
                                   ("violated", "outcome_violated")):
                        assert abs(r[a_] - ref[b_]) < 1e-12, (fold, k, a_)
                    n_assert += 1

    BT = pd.DataFrame(boot_rows)
    ref = BT[BT.grouping == "paper"].merge(R6, on=["fold", "level", "variant", "seed"])
    for met in ("fr", "cl"):
        x = ref[ref.metric == f"risk_{met}"]
        assert len(x) == len(BOOT_FOLDS) * len(VARIANTS) * 4
        assert np.allclose(x.point, x[f"risk_{met}"], atol=1e-12)
        assert np.allclose(x.lo, x[f"lo_{met}"], atol=1e-12) and np.allclose(x.hi, x[f"hi_{met}"], atol=1e-12)
        assert np.allclose(x.frac_above, x[f"boot_frac_above_{met}"], atol=1e-12)
    print(f"SANITY: paper-cluster grouping reproduces revision2_R6a_weighted.csv intervals (point, lo, hi, share above "
          f"alpha) in {len(ref)} cells; recalibration with paper units reproduces conformal_cluster fs_b and R4 outcomes "
          f"({n_assert} checks) - OK")
    BT.to_csv(EV / "revision3_block_bootstrap.csv", index=False)
    RC = pd.DataFrame(rec_rows); RC.to_csv(EV / "revision3_block_recal.csv", index=False)
    print("wrote outputs/eval/revision3_block_bootstrap.csv, revision3_block_recal.csv")

    s = BT[BT.seed == "all"]
    s = s.assign(c=[f"{r.point:.3f} ({r.lo:.3f}-{r.hi:.3f}) {r.frac_above:.2f}" for r in s.itertuples()],
                 g=s.grouping + "(" + s.n_groups.astype(str) + ")")
    for met in ("risk_fr", "risk_cl"):
        print(f"\nTarget bootstrap, {met}, seed all: point (2.5-97.5%) share>alpha")
        print(s[s.metric == met].pivot_table(index=["fold", "g"], columns=["level", "variant"], values="c",
                                             aggfunc="first").to_string())
    q = RC[(RC.seed == "all")]
    print("\nRecalibration one frame x k units (seed all):")
    print(q.round(3).to_string(index=False))
    verdicts(BT)


def verdicts(BT, min_groups=8):
    """Part 3 (main-text Table 4): per held-out centre, how many calibration-set weightings gave a mean miss rate above
    alpha (among those that certified a threshold), and the interval verdict of the four Table 2 variants, with a flag
    if a coarser grouping (>= min_groups groups) changes it. Reads the existing result files only."""
    f = lambda df: df[(df.iou == T) & (df.alpha == ALPHA) & (df.seed.astype(str) == "all")]
    a = f(pd.read_csv(EV / "revision2_R6a_weighted.csv"))
    b = f(pd.read_csv(EV / "revision2_R6b_stratified.csv"))
    c = f(pd.read_csv(EV / "revision2_R6c_loso_cluster.csv"))
    r3 = f(pd.read_csv(EV / "revision_R3_calibration_composition.csv"))
    w = [(x.fold, x.level, f"R6a {x.variant}", x.risk_fr if x.level == "frame" else x.risk_cl)
         for x in a.itertuples() if x.degenerate == 0]
    w += [(x.fold, x.level, "R6b stratified", x.risk_fr if x.level == "frame" else x.risk_cl)
          for x in b.itertuples() if x.degenerate_frac < 0.5]
    w += [(x.fold, "cluster", f"R6c excl {x.excluded}", x.risk_cl) for x in c.itertuples() if x.degenerate == 0]
    w += [(x.fold, "frame", f"R3 {x.variant}", x.risk_fr_mean) for x in r3.itertuples()
          if x.degenerate_frac == 0 and x.variant != "all"]  # R3 'all' == R6a equal
    W = pd.DataFrame(w, columns=["fold", "level", "weighting", "risk"])
    v = lambda lo, hi: "above" if lo > ALPHA else ("below" if hi < ALPHA else "includes")
    rows = []
    for fold in sorted(set(W.fold)):
        r = dict(fold=fold)
        for lev, met in (("frame", "fr"), ("cluster", "cl")):
            g = W[(W.fold == fold) & (W.level == lev)]
            r.update({f"{met}_n_weightings": len(g), f"{met}_n_above": int((g.risk > ALPHA).sum()),
                      f"{met}_min": g.risk.min(), f"{met}_max": g.risk.max()})
        for lev, var in VARIANTS:
            met = "fr" if lev == "frame" else "cl"
            x = a[(a.fold == fold) & (a.level == lev) & (a.variant == var)].iloc[0]
            key = f"{met}_{var}"
            r[key] = v(x[f"lo_{met}"], x[f"hi_{met}"])
            y = BT[(BT.fold == fold) & (BT.level == lev) & (BT.variant == var) & (BT.metric == f"risk_{met}")
                   & (BT.seed == "all") & (BT.grouping != "paper") & (BT.n_groups >= min_groups)]
            r[key + "_changes_with_grouping"] = ";".join(f"{z.grouping}:{v(z.lo, z.hi)}" for z in y.itertuples()
                                                         if v(z.lo, z.hi) != r[key])
        rows.append(r)
    V = pd.DataFrame(rows)
    V.to_csv(EV / "revision3_verdicts.csv", index=False)
    print("\nwrote outputs/eval/revision3_verdicts.csv"); print(V.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
