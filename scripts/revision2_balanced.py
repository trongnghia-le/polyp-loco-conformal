"""Referee round-2 M1: does pooled-source CRC still fail at the held-out centre when every SOURCE CENTRE counts equally?
Existing YOLO11s predictions only (18 runs), CPU, no training. Note: docs/notes/2026-11-09_balanced-calibration.md.

Unit u = positive source-cal frame (level 'frame', loss L_i of conformal.py) or positive source-cal cluster (level
'cluster', loss L_c = mean of L_i over the cluster's positive frames, as conformal_cluster.py). K = 5 source centres,
n_k units of centre k.
R6a weighted CRC (Angelopoulos et al., Conformal Risk Control, sec. 4.1; Tibshirani et al. 2019): target = equal
    mixture of the K source centres, likelihood ratio w_u = (1/K)/(n_k/n) ~ 1/n_k. Rule: largest lam with
    sum_u p_u L_u(lam) + p_test * B <= alpha, p_u = w_u / (sum w + w_test), p_test = w_test / (sum w + w_test), B = 1.
    The test unit's centre is unknown -> conservative w_test = max_u w_u = 1/n_min. Rescaling w to max 1 (w' = n_min/n_k)
    the rule is (sum w'L + 1)/(sum w' + 1) <= alpha, i.e. conformal.crc_idx with summed loss w'@L and n = sum w' =
    K*n_min: ordinary CRC on the centre-balanced empirical risk with effective size K*n_min. With equal weights it is
    exactly the published pooled / pooled_cl rule (asserted). 'balanced_plugin' drops the p_test*B term:
    sum_u w_u L_u / sum_u w_u <= alpha (no finite-sample correction; sensitivity only).
R6b equal-count stratified subsampling: m = min_k n_k units per centre without replacement, R_STRAT draws (seed
    independent), ordinary CRC on the K*m units.
R6c leave-one-source-out, cluster level: pooled_cl without each source centre's calibration clusters.
All variants: target risk frame-averaged (risk_fr) and video-averaged (risk_cl), false alarms per polyp-free target
frame (fa_neg, C1/C4/C5/C6). R6a: target-cluster bootstrap identical to R1 (same resamples, hierarchical over seeds).
Usage: .venv/Scripts/python.exe -I scripts/revision2_balanced.py -> outputs/eval/revision2_R6{a,b,c}_*.csv"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conformal import clean, G, ALPHAS, IOUS, NEG_CENTRES, load_run, crc_idx  # noqa: E402
from metrics import crc_lambda  # noqa: E402
from revision_analyses import B, interval, pick  # noqa: E402

EV = ROOT / "outputs" / "eval"
R_STRAT = 200
REFEREE_NOC3 = {"C1": .054, "C2": .054, "C4": .095, "C5": .181, "C6": .107}  # review_round2.md M1, IoU .1, alpha .1


def balanced_w(cen):
    """w_u ~ (1/K)/n_k, rescaled so max w = 1 (the conservative test weight)."""
    _, inv, cnt = np.unique(cen, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    return w / w.max()


def weighted_crc(w, Lu, alpha):
    """Weighted CRC with p_test = max normalised weight; w must have max 1 (w_test = 1). -> (lam index, degenerate)."""
    assert abs(w.max() - 1) < 1e-12
    j, d = crc_idx((w @ Lu)[None], np.array([w.sum()]), alpha)
    return int(j[0]), bool(d[0])


def plugin_crc(w, Lu, alpha):
    ok = (w @ Lu) / w.sum() <= alpha
    return (int(G.size - 1 - np.argmax(ok[::-1])), False) if ok.any() else (0, True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    GT = {clean(p): np.asarray(json.loads(gb), float).reshape(-1, 4) for p, gb in zip(man.path, man.gt_boxes)}
    cs = pd.read_csv(EV / "conformal_summary.csv", dtype={"seed": str})
    cc = pd.read_csv(EV / "conformal_cluster_summary.csv", dtype={"seed": str})
    R1 = pd.read_csv(EV / "revision_R1_target_bootstrap.csv", dtype={"seed": str})
    runs = sorted(d for d in (ROOT / "outputs" / "kaggle").glob("C*_seed*") if next(d.glob("preds_*.parquet"), None))
    assert len(runs) == 18, len(runs)
    ra, rb, rc, boots, n_assert, counts = [], [], [], {}, 0, {}

    for d in runs:
        fold, seed = d.name.split("_seed"); seed = int(seed); fi = int(fold[1:])
        idx, K, N, L, _ = load_run(d, fold, GT, np.random.default_rng([2026, fi, seed]))
        cal, tgt, pos = (idx.role == "cal").to_numpy(), (idx.role == "target").to_numpy(), K > 0
        centre = idx.center.to_numpy()
        ucl, ci = np.unique(idx.cluster_id.to_numpy(), return_inverse=True); C = len(ucl)
        m = np.bincount(ci, weights=pos, minlength=C)
        A = np.zeros((C, len(idx))); A[ci[pos], np.flatnonzero(pos)] = 1.0 / m[ci[pos]]
        Lc = {t: A @ L[t] for t in IOUS}  # cluster loss, exactly as conformal_cluster.py
        c_cal, c_tgt = np.bincount(ci, weights=cal, minlength=C) > 0, np.bincount(ci, weights=tgt, minlength=C) > 0
        c_centre = pd.Series(centre).groupby(ci).first().to_numpy()
        calP, tgtP = np.flatnonzero(c_cal & (m > 0)), np.flatnonzero(c_tgt & (m > 0))
        nT, mT = len(tgtP), m[tgtP]
        tneg = tgt & ~pos
        cp = np.flatnonzero(cal & pos)
        units = {"frame": (cp, centre[cp], lambda t: L[t][cp]), "cluster": (calP, c_centre[calP], lambda t: Lc[t][calP])}
        cfg = dict(fold=fold, seed=str(seed))
        # target-cluster bootstrap resamples: identical to revision_analyses.py R1
        U = np.random.default_rng([2026, fi, 5150]).integers(0, nT, (B, nT))
        Cnt = np.zeros((B, nT)); np.add.at(Cnt, (np.repeat(np.arange(B), nT), U.ravel()), 1)
        # R6b draws (seed independent): m units per centre, positions within each centre's unit list
        strat = {}
        for lev, (u, cen, _) in units.items():
            cens, cnt = np.unique(cen, return_counts=True)
            counts[(fold, seed, lev)] = dict(zip(cens, cnt))
            mm = int(cnt.min()); g = np.random.default_rng([2026, fi, 66, lev == "cluster"])
            W = np.zeros((R_STRAT, len(u)))
            for r in range(R_STRAT):
                for c in cens:
                    W[r, g.choice(np.flatnonzero(cen == c), mm, replace=False)] = 1
            assert (W.sum(1) == mm * len(cens)).all()
            strat[lev] = (mm, W)

        for t in IOUS:
            LT = Lc[t][tgtP]
            pf0, pc0 = (mT[:, None] * LT).sum(0) / mT.sum(), LT.mean(0)  # frame-avg / video-avg target risk per lam
            PC, PF = Cnt @ LT / nT, (Cnt * mT) @ LT / (Cnt * mT).sum(1)[:, None]
            FA = N[tneg].mean(0) if fold in NEG_CENTRES else np.full(G.size, np.nan)
            ev = lambda j: dict(lam=G[j], risk_fr=pf0[j], risk_cl=pc0[j], fa_neg=FA[j])
            for alpha in ALPHAS:
                for lev, (u, cen, Lf) in units.items():
                    Lu = Lf(t)
                    # sanity: equal weights == published pooled (frame) / pooled_cl (cluster)
                    j, dg = weighted_crc(np.ones(len(u)), Lu, alpha)
                    assert (G[j], dg) == crc_lambda(Lu, alpha, G)
                    if lev == "frame":
                        ref = pick(cs, **cfg, iou=t, alpha=alpha, strategy="pooled", budget=0)
                        assert G[j] == ref.lam_median and abs(pf0[j] - ref.risk_mean) < 1e-12
                    else:
                        ref = pick(cc, **cfg, iou=t, alpha=alpha, strategy="pooled_cl", budget=0)
                        assert G[j] == ref.lam_median and abs(pc0[j] - ref.risk_cl_mean) < 1e-12
                        assert abs(pf0[j] - ref.risk_fr_mean) < 1e-12
                    n_assert += 1
                    wb = balanced_w(cen)
                    nmin = int(np.unique(cen, return_counts=True)[1].min())
                    assert abs(wb.sum() - len(set(cen)) * nmin) < 1e-9  # sum w' = K * n_min
                    for var, (j, dg), neff in (("equal", (j, dg), len(u)), ("balanced_conservative", weighted_crc(wb, Lu, alpha), wb.sum()),
                                               ("balanced_plugin", plugin_crc(wb, Lu, alpha), wb.sum())):
                        ra.append(dict(**cfg, iou=t, alpha=alpha, level=lev, variant=var, degenerate=dg, n_units=len(u),
                                       n_min=nmin, n_eff=neff, p_test=1 / (neff + 1) if var != "balanced_plugin" else 0.0,
                                       **ev(j)))
                        for met, PB, p0 in (("risk_fr", PF, pf0), ("risk_cl", PC, pc0)):
                            boots[(fold, t, alpha, lev, var, met, seed)] = (p0[j], PB[:, j])
                    # R6b
                    mm, W = strat[lev]
                    js, ds = crc_idx(W @ Lu, W.sum(1), alpha)
                    rb.append(pd.DataFrame(dict(**cfg, iou=t, alpha=alpha, level=lev, rep=np.arange(R_STRAT), m=mm,
                                                n_cal=W.sum(1), degenerate=ds, **ev(js))))
                # R6c: cluster-level leave-one-source-out
                u, cen, Lf = units["cluster"]
                for ex in sorted(set(cen)):
                    keep = (cen != ex).astype(float)
                    j, dg = crc_idx((keep @ Lf(t))[None], np.array([keep.sum()]), alpha)
                    rc.append(dict(**cfg, iou=t, alpha=alpha, excluded=ex, n_clusters=int(keep.sum()), degenerate=bool(dg[0]),
                                   **ev(int(j[0]))))
        print(f"  {d.name}: positive cal frames per centre {counts[(fold, seed, 'frame')]}, "
              f"clusters {counts[(fold, seed, 'cluster')]}", flush=True)
    print(f"SANITY: equal-weight weighted CRC == metrics.crc_lambda == published pooled / pooled_cl (lam and risk) "
          f"in {n_assert} cells - OK")

    # ---------- R6a: per seed + hierarchical 'all' (seed drawn per resample, as R1)
    A6 = pd.DataFrame(ra)
    keys = sorted({k[:6] for k in boots})
    iv = []
    for k in keys:
        fold, t, alpha = k[:3]
        seeds = sorted(s for (*kk, s) in boots if tuple(kk) == k)
        for s in seeds:
            iv.append(dict(fold=fold, iou=t, alpha=alpha, level=k[3], variant=k[4], metric=k[5], seed=str(s),
                           **interval(*boots[(*k, s)], alpha)))
        pick_s = np.random.default_rng([2026, int(fold[1:]), 5152]).integers(0, len(seeds), B)
        mix = np.array([boots[(*k, seeds[i])][1][b] for b, i in enumerate(pick_s)])
        iv.append(dict(fold=fold, iou=t, alpha=alpha, level=k[3], variant=k[4], metric=k[5], seed="all",
                       **interval(np.mean([boots[(*k, s)][0] for s in seeds]), mix, alpha)))
    IV = pd.DataFrame(iv)
    # sanity: equal-weight intervals == R1 pooled (frame-avg) and pooled_cl (video-avg)
    for lev, st, met in (("frame", "pooled", "risk_fr"), ("frame", "pooled", "risk_cl"), ("cluster", "pooled_cl", "risk_cl"),
                         ("cluster", "pooled_cl", "risk_fr")):
        x = IV[(IV.level == lev) & (IV.variant == "equal") & (IV.metric == met)].merge(
            R1[(R1.strategy == st) & (R1.metric == met)], on=["fold", "iou", "alpha", "seed"], suffixes=("", "_r1"))
        assert len(x) == 6 * 2 * len(ALPHAS) * 4 and np.allclose(x[["point", "lo", "hi"]].to_numpy(),
                                                                 x[["point_r1", "lo_r1", "hi_r1"]].to_numpy(), atol=1e-12)
    print("SANITY: equal-weight bootstrap intervals == revision_R1 pooled / pooled_cl (both estimands, all seeds) - OK")
    IVw = IV.pivot_table(index=["fold", "iou", "alpha", "level", "variant", "seed"], columns="metric",
                         values=["lo", "hi", "frac_above"]).reset_index()
    IVw.columns = [f"{a}_{b}".replace("risk_", "") if b else a for a, b in IVw.columns]
    ka = ["fold", "iou", "alpha", "level", "variant"]
    allr = A6.groupby(ka).agg(lam=("lam", "median"), degenerate=("degenerate", "mean"), n_units=("n_units", "first"),
                              n_min=("n_min", "first"), n_eff=("n_eff", "first"), p_test=("p_test", "first"),
                              risk_fr=("risk_fr", "mean"), risk_cl=("risk_cl", "mean"), fa_neg=("fa_neg", "mean"),
                              seeds_above_fr=("risk_fr", lambda x: x.gt(A6.loc[x.index, "alpha"]).mean()),
                              seeds_above_cl=("risk_cl", lambda x: x.gt(A6.loc[x.index, "alpha"]).mean())).reset_index()
    out_a = pd.concat([allr.assign(seed="all"), A6], ignore_index=True).merge(IVw, on=ka + ["seed"])
    out_a = out_a.rename(columns={"frac_above_fr": "boot_frac_above_fr", "frac_above_cl": "boot_frac_above_cl"})
    out_a.to_csv(EV / "revision2_R6a_weighted.csv", index=False)

    # ---------- R6b
    Bl = pd.concat(rb, ignore_index=True)
    Bl["viol_fr"], Bl["viol_cl"] = Bl.risk_fr > Bl.alpha, Bl.risk_cl > Bl.alpha
    q = lambda p: (lambda x: x.quantile(p))
    agg = dict(m=("m", "first"), n_cal=("n_cal", "first"), degenerate_frac=("degenerate", "mean"),
               risk_fr=("risk_fr", "mean"), risk_fr_q05=("risk_fr", q(.05)), risk_fr_q95=("risk_fr", q(.95)),
               viol_fr=("viol_fr", "mean"), risk_cl=("risk_cl", "mean"), risk_cl_q05=("risk_cl", q(.05)),
               risk_cl_q95=("risk_cl", q(.95)), viol_cl=("viol_cl", "mean"), fa_neg=("fa_neg", "mean"),
               lam_median=("lam", "median"), n_rows=("risk_fr", "size"))
    kb = ["fold", "iou", "alpha", "level"]
    out_b = pd.concat([Bl.groupby(kb).agg(**agg).reset_index().assign(seed="all"),
                       Bl.groupby(kb + ["seed"]).agg(**agg).reset_index()], ignore_index=True)
    out_b["feasible"] = out_b.n_cal >= 1 / out_b.alpha - 1 - 1e-9
    out_b.to_csv(EV / "revision2_R6b_stratified.csv", index=False)

    # ---------- R6c
    Cl = pd.DataFrame(rc)
    kc = ["fold", "iou", "alpha", "excluded"]
    out_c = pd.concat([Cl.groupby(kc).agg(n_clusters=("n_clusters", "first"), degenerate=("degenerate", "mean"),
                                          lam=("lam", "median"), risk_cl=("risk_cl", "mean"), risk_fr=("risk_fr", "mean"),
                                          fa_neg=("fa_neg", "mean"),
                                          seeds_above_cl=("risk_cl", lambda x: x.gt(Cl.loc[x.index, "alpha"]).mean()),
                                          seeds_above_fr=("risk_fr", lambda x: x.gt(Cl.loc[x.index, "alpha"]).mean()))
                       .reset_index().assign(seed="all"), Cl], ignore_index=True)
    out_c.to_csv(EV / "revision2_R6c_loso_cluster.csv", index=False)
    x = out_c[(out_c.seed == "all") & (out_c.iou == 0.1) & (out_c.alpha == 0.10) & (out_c.excluded == "C3")]
    for r in x.itertuples():
        assert round(r.risk_cl, 3) == REFEREE_NOC3[r.fold], (r.fold, r.risk_cl)
    print(f"SANITY: referee's no-C3 cluster-level numbers reproduced {sorted(REFEREE_NOC3.items())} - OK")
    print("wrote outputs/eval/revision2_R6a_weighted.csv, revision2_R6b_stratified.csv, revision2_R6c_loso_cluster.csv")

    # ---------- compact printouts
    for t in IOUS:
        for alpha in (0.10, 0.20):
            s = out_a[(out_a.seed == "all") & (out_a.iou == t) & (out_a.alpha == alpha)]
            print(f"\nR6a IoU {t}, alpha {alpha:.2f}: point [2.5,97.5%] Pboot>a | seeds>a | degenerate frac")
            for met in ("fr", "cl"):
                tab = {(r.level, r.variant): None for r in s.itertuples()}
                tab = {k: {r.fold: f"{getattr(r, 'risk_' + met):.3f} [{getattr(r, 'lo_' + met):.3f},{getattr(r, 'hi_' + met):.3f}] "
                              f"{getattr(r, 'boot_frac_above_' + met):.2f}|{getattr(r, 'seeds_above_' + met):.2f}|{r.degenerate:.2f}"
                           for r in s[(s.level == k[0]) & (s.variant == k[1])].itertuples()} for k in tab}
                print(f" risk_{met}:"); print(pd.DataFrame(tab).T.to_string())
            s = out_b[(out_b.seed == "all") & (out_b.iou == t) & (out_b.alpha == alpha)]
            print(f"R6b IoU {t}, alpha {alpha:.2f}: m, n, feasible, degenerate, risk_fr (viol), risk_cl (viol), fa")
            print(s[["fold", "level", "m", "n_cal", "feasible", "degenerate_frac", "risk_fr", "viol_fr", "risk_cl", "viol_cl",
                     "fa_neg"]].round(3).to_string(index=False))
            s = out_c[(out_c.seed == "all") & (out_c.iou == t) & (out_c.alpha == alpha)]
            print(f"R6c IoU {t}, alpha {alpha:.2f}: risk_cl / risk_fr (seeds>a cl) by excluded centre")
            print(s.assign(c=[f"{a:.3f}/{b:.3f} ({c:.2f}){' D' if dg else ''}" for a, b, c, dg in
                              zip(s.risk_cl, s.risk_fr, s.seeds_above_cl, s.degenerate)])
                  .pivot(index="excluded", columns="fold", values="c").fillna("-").to_string())


if __name__ == "__main__":
    main()
