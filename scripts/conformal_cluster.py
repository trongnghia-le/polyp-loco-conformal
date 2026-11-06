"""Cluster-level conformal risk control: the source VIDEO/cluster (data/clusters.csv `cluster_id`) is the exchangeable
unit instead of the frame. Builds on scripts/conformal.py (loading, matched-score losses, vectorized CRC; not modified).

Unit = cluster with >= 1 positive frame (n_gt > 0). Cluster loss L_c(lam) = mean over its m_c positive frames of the
per-image miss fraction L_i(lam) (docs/metric.md sec. 3). lam_hat = largest lam with (sum_c L_c(lam) + 1)/(n_c + 1) <= alpha.
Guarantee (clusters exchangeable): E[L_{new cluster}(lam_hat)] <= alpha, i.e. the CLUSTER-AVERAGED risk (each video counts
once, whatever its length), not the frame-averaged risk reported by conformal.py. Both are reported (risk_cl, risk_fr).

Strategies (per run <root>/<fold>_seed<k>, IoU in {0.1, 0.5}, alpha in conformal.ALPHAS):
  in_domain_cl      : --splits random 50/50 splits of the source-cal POSITIVE clusters; cluster CRC on one half, test other.
  in_domain_img     : same splits, image-level CRC on all positive frames of the calibration half (comparison).
  pooled_cl (+_boot): cluster CRC on all source-cal positive clusters -> whole target (boot: stratified cluster bootstrap).
  worst_cl (+_boot) : min over source centres of the per-centre cluster lam_hat (needs n_c >= 1/alpha - 1 clusters per
                      centre, else that centre is degenerate). worst_cl_feas: min over the centres where it is feasible.
  fs_a / fs_b / fs_c: k in K_BUDGETS positive target clusters drawn uniformly (k capped at n_pos_clusters - MIN_TEST);
                      (a) cluster CRC on them, (b) image CRC on ONE random positive frame per drawn cluster, (c) image CRC
                      on all positive frames of the drawn clusters. Test = all target clusters not drawn.
  fs_frames         : the stored frame-budget draws (splits/target_draws_<fold>.json, budgets 20/50/100, first --draws)
                      with image CRC = conformal.py few_shot, re-evaluated here with both risks and the cluster count.
Draws are identical across seeds/IoU/alpha: clusters ~ default_rng([2026, fold_idx, k, 11]).choice(sorted positive target
clusters, k, replace=False) per draw in order; frame for (b) ~ default_rng([2026, fold_idx, k, 12]).
Labelling cost (cost_frames): (a)/(c) all frames of the drawn clusters (every frame must be annotated to know which are
positive and their losses); (b) k frames. cost_pos = positive frames used for calibration.

Usage: python scripts/conformal_cluster.py [--root outputs/kaggle] [--draws 500] [--boot 200] [--splits 200]
Writes outputs/eval/conformal_cluster_long{sfx}.parquet, conformal_cluster_summary{sfx}.csv (sfx as in conformal.py)."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conformal import clean, G, ALPHAS, IOUS, BUDGETS, NEG_CENTRES, load_run, crc_idx, evaluate  # noqa: E402
from metrics import crc_lambda  # noqa: E402

K_BUDGETS, MIN_TEST = (5, 10, 20, 40), 10


def brute_lambda(Lc, alpha):
    """Plain loop over the grid (sanity reference for crc_idx on cluster losses)."""
    best, deg = G[0], True
    for g in range(G.size):
        if (sum(Lc[c, g] for c in range(Lc.shape[0])) + 1) / (Lc.shape[0] + 1) <= alpha:
            best, deg = G[g], False
    return best, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/kaggle")
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--splits", type=int, default=200)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    root = (ROOT / a.root).resolve()
    sfx = "" if root.name == "kaggle" else f"_{root.name}"
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    GT = {clean(p): np.asarray(json.loads(gb), float).reshape(-1, 4) for p, gb in zip(man.path, man.gt_boxes)}
    runs = sorted(d for d in root.glob("C*_seed*") if next(d.glob("preds_*.parquet"), None))
    print(f"{len(runs)} runs under {root}: {[d.name for d in runs]}")
    out, caps, n_brute, n_tie = [], {}, 0, 0
    for d in runs:
        fold, seed = d.name.split("_seed"); seed = int(seed); fi = int(fold[1:])
        rng = np.random.default_rng([2026, fi, seed, 99])
        idx, K, N, L, _ = load_run(d, fold, GT, rng)
        cal, tgt, pos = (idx.role == "cal").to_numpy(), (idx.role == "target").to_numpy(), K > 0
        ucl, ci = np.unique(idx.cluster_id.to_numpy(), return_inverse=True)
        C = len(ucl)
        m = np.bincount(ci, weights=pos, minlength=C)  # positive frames per cluster
        nfr = np.bincount(ci, minlength=C)  # all frames per cluster
        cpos = m > 0
        A = np.zeros((C, len(idx))); A[ci[pos], np.flatnonzero(pos)] = 1.0 / m[ci[pos]]
        Lc = {t: A @ L[t] for t in IOUS}  # (C, |G|), 0 for clusters without positives
        for t in IOUS:
            assert (Lc[t] >= -1e-12).all() and (Lc[t] <= 1 + 1e-12).all() and (np.diff(Lc[t], axis=1) >= -1e-12).all()
        c_cal = np.bincount(ci, weights=cal, minlength=C) > 0
        c_tgt = np.bincount(ci, weights=tgt, minlength=C) > 0
        assert not (c_cal & c_tgt).any()
        c_centre = pd.Series(idx.center.to_numpy()).groupby(ci).first().to_numpy()
        calP, tgtP = np.flatnonzero(c_cal & cpos), np.flatnonzero(c_tgt & cpos)
        img = lambda V: V[:, ci]  # cluster weights (R, C) -> image weights (R, n)
        centres = sorted(set(c_centre[calP]))
        fa_ok = fold in NEG_CENTRES

        def onehot(sel, R):
            V = np.zeros((R, C)); np.put_along_axis(V, np.asarray(sel), 1.0, 1); return V

        # --- in-domain splits of the positive source-cal clusters
        half = len(calP) // 2
        perm = np.array([rng.permutation(calP) for _ in range(a.splits)])
        Vin_cal, Vin_test = onehot(perm[:, :half], a.splits), onehot(perm[:, half:], a.splits)
        # --- cluster bootstrap of source-cal positive clusters, stratified by centre
        Vboot = np.zeros((a.boot, C))
        for c in centres:
            mc = calP[c_centre[calP] == c]
            for r in range(a.boot):
                np.add.at(Vboot[r], rng.choice(mc, len(mc)), 1)
        V0 = np.zeros((1, C)); V0[0, calP] = 1
        Vtgt = np.zeros((1, C)); Vtgt[0, c_tgt] = 1
        # --- few-shot cluster draws (seed independent)
        fs, pf = {}, {c: np.flatnonzero(pos & (ci == c)) for c in tgtP}
        for k in K_BUDGETS:
            ke = min(k, len(tgtP) - MIN_TEST); caps[fold] = len(tgtP) - MIN_TEST
            r1, r2 = np.random.default_rng([2026, fi, k, 11]), np.random.default_rng([2026, fi, k, 12])
            sel = np.array([r1.choice(tgtP, ke, replace=False) for _ in range(a.draws)])
            Vd = onehot(sel, a.draws)
            Vtest = np.where(c_tgt[None], 1 - Vd, 0)
            Wone = np.zeros((a.draws, len(idx)))
            for r in range(a.draws):
                Wone[r, [r2.choice(pf[c]) for c in sel[r]]] = 1
            fs[k] = (ke, Vd, Vtest, Wone, (Vd * nfr).sum(1), (Vd * m).sum(1))
        # --- stored frame-budget draws (as conformal.py few_shot)
        draws = json.loads((ROOT / "splits" / f"target_draws_{fold}.json").read_text())
        fsf = {}
        for b in BUDGETS:
            Vd = np.array([np.isin(ucl, x) for x in draws[str(b)][:a.draws]], float)
            assert (Vd <= c_tgt).all()
            fsf[b] = (Vd, np.where(c_tgt[None], 1 - Vd, 0))

        def rows(strategy, alpha, t, j, deg, Vtest, Wtest, n_cal, k_cal, budget=0, k_eff=0, cost=np.nan,
                 cost_pos=np.nan, binding=""):
            R = len(j)
            Vt = Vtest * cpos
            ev = evaluate(Wtest, j, K, L[t], N)
            r = dict(model=root.name, fold=fold, seed=seed, iou=t, alpha=alpha, strategy=strategy, budget=budget,
                     k_eff=k_eff, rep=np.arange(R), lam=G[j], degenerate=deg, lam_max=j == G.size - 1,
                     risk_cl=(Vt * Lc[t][:, j].T).sum(1) / Vt.sum(1), risk_fr=ev["risk"], micro_miss=ev["micro_miss"],
                     fa_neg=ev["fa_neg"] if (fa_ok and not strategy.startswith("in_domain")) else np.nan,
                     fp_pos=ev["fp_pos"], n_test_fr=ev["n_test"], k_test=Vt.sum(1), n_cal=n_cal, k_cal=k_cal,
                     cost_frames=cost, cost_pos=cost_pos, binding=binding)
            r["viol_cl"], r["viol_fr"] = r["risk_cl"] > alpha, r["risk_fr"] > alpha
            if fold == "C3" and strategy.startswith("fs_"):  # all singletons: both risks coincide
                assert np.allclose(r["risk_cl"], r["risk_fr"])
            out.append(pd.DataFrame(r))

        clcrc = lambda V, t, alpha: crc_idx(V @ Lc[t], V.sum(1), alpha)
        imcrc = lambda W, t, alpha: crc_idx((W * pos) @ L[t], (W * pos).sum(1), alpha)
        for t in IOUS:
            for alpha in ALPHAS:
                # sanity: cluster lam_hat == metrics.crc_lambda == brute-force loop (pooled; plus one bootstrap)
                j, deg = clcrc(V0, t, alpha)
                assert (G[j[0]], deg[0]) == crc_lambda(Lc[t][calP], alpha, G)
                if alpha in (0.10, 0.20):
                    jb, db = clcrc(Vboot[:1], t, alpha)
                    Lrep = np.repeat(Lc[t], Vboot[0].astype(int), axis=0)
                    for (jj, dd), LL in (((j[0], deg[0]), Lc[t][calP]), ((jb[0], db[0]), Lrep)):
                        assert (G[jj], dd) == brute_lambda(LL, alpha); n_brute += 1
                rows("pooled_cl", alpha, t, j, deg, Vtgt, img(Vtgt) * 1.0, V0.sum(1), V0.sum(1))
                j, deg = clcrc(Vboot, t, alpha)
                rows("pooled_cl_boot", alpha, t, j, deg, np.repeat(Vtgt, a.boot, 0), np.repeat(img(Vtgt), a.boot, 0),
                     Vboot.sum(1), (Vboot > 0).sum(1))
                for name, V in (("worst_cl", V0), ("worst_cl_boot", Vboot)):
                    js, ds = map(np.array, zip(*[clcrc(V * (c_centre == c), t, alpha) for c in centres]))
                    ncl = np.array([(c_centre[calP] == c).sum() for c in centres])
                    feas = 1.0 / (ncl + 1) <= alpha
                    R = len(V)
                    for nm, keep in ((name, np.ones_like(feas)), (name.replace("worst_cl", "worst_cl_feas"), feas)):
                        if not keep.any():
                            continue
                        jj = np.where(keep[:, None], js, G.size - 1).min(0)
                        dd = (ds & keep[:, None]).any(0)
                        bind = np.where(keep[:, None], js, G.size).argmin(0)
                        rows(nm, alpha, t, jj, dd, np.repeat(Vtgt, R, 0), np.repeat(img(Vtgt), R, 0), V.sum(1),
                             (V > 0).sum(1), binding=np.array(centres)[bind])
                # in-domain
                j, deg = clcrc(Vin_cal, t, alpha)
                Wte = img(Vin_test)
                rows("in_domain_cl", alpha, t, j, deg, Vin_test, Wte, Vin_cal.sum(1), Vin_cal.sum(1))
                Wc = img(Vin_cal)
                j, deg = imcrc(Wc, t, alpha)
                rows("in_domain_img", alpha, t, j, deg, Vin_test, Wte, (Wc * pos).sum(1), Vin_cal.sum(1))
                # few-shot by clusters
                for k, (ke, Vd, Vtest, Wone, cost_all, cost_p) in fs.items():
                    Wt = img(Vtest)
                    ja, da = clcrc(Vd, t, alpha)
                    rows("fs_a", alpha, t, ja, da, Vtest, Wt, Vd.sum(1), Vd.sum(1), k, ke, cost_all, cost_p)
                    jb, db = imcrc(Wone, t, alpha)
                    rows("fs_b", alpha, t, jb, db, Vtest, Wt, Wone.sum(1), Vd.sum(1), k, ke, np.full(a.draws, ke),
                         np.full(a.draws, ke))
                    jc, dc = imcrc(img(Vd), t, alpha)
                    rows("fs_c", alpha, t, jc, dc, Vtest, Wt, cost_p, Vd.sum(1), k, ke, cost_all, cost_p)
                    if fold == "C3":  # all singletons: cluster CRC == image CRC (both designs)
                        Sa, Sc = Vd @ Lc[t], (img(Vd) * pos) @ L[t]
                        assert np.allclose(Sa, Sc) and np.allclose(Sa, Wone @ L[t])
                        for jx, dx in ((jb, db), (jc, dc)):  # same lam_hat up to float ties at the alpha boundary
                            bad = np.flatnonzero((ja != jx) | (da != dx))
                            crit = (Sa[bad, np.maximum(ja, jx)[bad]] + 1) / (Vd.sum(1)[bad] + 1)
                            assert np.allclose(crit, alpha, atol=1e-9), (alpha, t, k, bad[:5])
                            n_tie += len(bad)
                for b, (Vd, Vtest) in fsf.items():
                    Wd = img(Vd)
                    j, deg = imcrc(Wd, t, alpha)
                    kp = (Vd * cpos).sum(1)
                    rows("fs_frames", alpha, t, j, deg, Vtest, img(Vtest), (Wd * pos).sum(1), kp, b, 0,
                         (Vd * nfr).sum(1), (Vd * m).sum(1))
        print(f"  {d.name}: source-cal positive clusters {len(calP)} ({dict(zip(centres, [(c_centre[calP] == c).sum() for c in centres]))}), "
              f"target positive clusters {len(tgtP)} (k cap {len(tgtP) - MIN_TEST}), target pos frames {int((tgt & pos).sum())}")
    print(f"SANITY: cluster loss in [0,1] and non-decreasing in lam (all runs, both IoU) - OK; cluster lam_hat == "
          f"metrics.crc_lambda (all cells) and == brute-force loop ({n_brute} cases) - OK; C3 (all singletons): cluster "
          f"CRC == image CRC for designs a/b/c (summed losses allclose; lam_hat equal except {n_tie} float ties at "
          f"the alpha boundary) and risk_cl == risk_fr - OK")

    long = pd.concat(out, ignore_index=True)
    ev = ROOT / "outputs" / "eval"
    long.to_parquet(ev / f"conformal_cluster_long{sfx}.parquet", index=False)
    keys = ["model", "fold", "iou", "alpha", "strategy", "budget"]
    q = lambda p: (lambda x: x.quantile(p))
    agg = dict(n_rows=("risk_cl", "size"), risk_cl_mean=("risk_cl", "mean"), risk_cl_q05=("risk_cl", q(.05)),
               risk_cl_q95=("risk_cl", q(.95)), risk_fr_mean=("risk_fr", "mean"), risk_fr_q05=("risk_fr", q(.05)),
               risk_fr_q95=("risk_fr", q(.95)), viol_cl=("viol_cl", "mean"), viol_fr=("viol_fr", "mean"),
               micro_miss_mean=("micro_miss", "mean"), fa_neg_mean=("fa_neg", "mean"), fp_pos_mean=("fp_pos", "mean"),
               degenerate_frac=("degenerate", "mean"), lam_max_frac=("lam_max", "mean"), lam_median=("lam", "median"),
               k_eff=("k_eff", "max"), n_cal_median=("n_cal", "median"), k_cal_median=("k_cal", "median"),
               k_test_median=("k_test", "median"), cost_frames_median=("cost_frames", "median"),
               cost_frames_q95=("cost_frames", q(.95)), cost_pos_median=("cost_pos", "median"),
               binding_mode=("binding", lambda x: x.mode().iloc[0] if (x != "").any() else ""))
    per_seed = long.groupby(keys + ["seed"]).agg(**agg).reset_index()
    alls = long.groupby(keys).agg(**agg).reset_index().assign(seed="all")
    alls = alls.merge(long.groupby(keys).seed.nunique().rename("n_seeds").reset_index(), on=keys)
    summ = pd.concat([alls, per_seed.assign(seed=per_seed.seed.astype(str))], ignore_index=True)
    summ.to_csv(ev / f"conformal_cluster_summary{sfx}.csv", index=False)
    print(f"wrote {ev / f'conformal_cluster_long{sfx}.parquet'} ({len(long)} rows), {ev / f'conformal_cluster_summary{sfx}.csv'}")
    print(f"k cap per target centre (positive clusters - {MIN_TEST}): {caps}")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for strat in ("in_domain_cl", "in_domain_img"):
        ind = per_seed[per_seed.strategy == strat]
        ok = ind.degenerate_frac == 0
        for col in ("risk_cl_mean", "risk_fr_mean"):
            ex = (ind[col] - ind.alpha)[ok]
            print(f"IN-DOMAIN {strat}, {col}: {int(ok.sum())}/{len(ind)} non-degenerate (fold,seed,IoU,alpha) cells; "
                  f"max(mean - alpha) = {ex.max():+.4f}; cells > alpha + 0.01: {(ex > 0.01).sum()}, > alpha + 0.02: {(ex > 0.02).sum()}")
    print("\nDEGENERATE fraction (all seeds, mean over folds), IoU x strategy x alpha:")
    print(alls.pivot_table(index=["iou", "strategy", "budget"], columns="alpha", values="degenerate_frac").round(2).to_string())

    s = alls[alls.iou == 0.1].set_index(["fold", "alpha", "strategy", "budget"])
    for alpha in (0.10, 0.20):
        print(f"\nIoU 0.1, alpha = {alpha:.2f}: mean target risk cluster-avg/frame-avg (violation freq cluster/frame). "
              f"in-domain: 200 splits x seeds; pooled: point estimate (violation over {a.boot} cluster bootstraps x seeds); "
              f"fs: {a.draws} draws x seeds; cost = median annotated frames for a/c (b costs k frames)")
        tab = {}
        for f in sorted(set(alls.fold)):
            c = lambda st, b=0: s.loc[(f, alpha, st, b)]
            cell = lambda x, v=None: f"{x.risk_cl_mean:.3f}/{x.risk_fr_mean:.3f} ({(v if v is not None else x).viol_cl:.2f}/{(v if v is not None else x).viol_fr:.2f})"
            row = {"in-dom cl": cell(c("in_domain_cl")), "pooled cl": cell(c("pooled_cl"), c("pooled_cl_boot"))}
            for k in (10, 20):
                for dz in "abc":
                    row[f"{dz}{k}"] = cell(c(f"fs_{dz}", k))
                row[f"cost{k}"] = f"{c('fs_a', k).cost_frames_median:.0f} (k={c('fs_a', k).k_eff:.0f})"
            tab[f] = row
        print(pd.DataFrame(tab).T.to_string())


if __name__ == "__main__":
    main()
