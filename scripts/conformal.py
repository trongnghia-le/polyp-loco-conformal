"""Conformal risk control (CRC) of the per-image polyp miss fraction under a hospital change (LOCO on PolypGen).
Spec: docs/metric.md (loss, CRC threshold, cluster units; decision 2026-11-06: IoU 0.1 primary, 0.5 secondary).

Per run <root>/<fold>_seed<k>/ (preds_*.parquet + index_*.csv) and IoU in {0.1, 0.5}, alpha in ALPHAS:
  in_domain        : 50/50 cluster splits of the source cal set (calibrate one half, test the other) = exchangeable ref.
  pooled           : lambda-hat on all source cal positives, tested on all target images (rep = -1).
  worst_case       : min over source centres of the per-centre lambda-hat (lower threshold = conservative).
  pooled_boot / worst_case_boot : same, on cluster-bootstrap resamples of the cal set (stratified by centre).
  few_shot         : stored target draws (budget 20/50/100, first R), calibrate on drawn positives, test the rest.
  shift-weighted CRC: skipped (no image features stored).
L_i(lam) uses the prefix property of greedy matching: each GT keeps the score of the box that matched it on the full
sorted list, and is missed at lam iff that score < lam (checked against metrics.miss_fraction on random images).

Usage: python scripts/conformal.py [--root outputs/kaggle] [--draws 500] [--boot 200] [--splits 200]
Writes outputs/eval/conformal_long{sfx}.parquet and conformal_summary{sfx}.csv; sfx = "" for outputs/kaggle,
else "_<root dir name>" (e.g. _kaggle_m)."""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metrics import iou, miss_fraction, crc_lambda  # noqa: E402

clean = lambda p: re.sub(r"[^A-Za-z0-9_./-]", "", p)  # same sanitizing as eval_detection.py / make_kaggle_upload.py
G = np.round(np.arange(1, 1000) * 1e-3, 3)  # lambda grid 0.001..0.999
ALPHAS, IOUS, BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.30), (0.1, 0.5), (20, 50, 100)
NEG_CENTRES = {"C1", "C4", "C5", "C6"}  # centres with negative sequences (FA reported only there)


def matched_scores(gt, boxes, scores, thr):
    """Score of the box matched to each GT polyp (0 if never matched); greedy one-to-one over ALL boxes by
    descending score (stable), each box takes the free GT with highest IoU >= thr (docs/metric.md sec. 2)."""
    out = np.zeros(len(gt))
    if len(gt) == 0 or len(scores) == 0:
        return out
    order = np.argsort(-scores, kind="stable")
    free = np.ones(len(gt), bool)
    for s, row in zip(scores[order], iou(boxes[order], gt)):
        row = np.where(free, row, -1.0); j = row.argmax()
        if row[j] >= thr:
            free[j], out[j] = False, s
            if not free.any():
                break
    return out


def load_run(d, fold, GT, rng):
    preds = pd.read_parquet(next(d.glob("preds_*.parquet")))
    idx = pd.read_csv(next(d.glob("index_*.csv")))
    idx = idx[idx.role.isin(["cal", "target"])].reset_index(drop=True)
    assert (idx.loc[idx.role == "target", "center"] == fold).all() and (idx.loc[idx.role == "cal", "center"] != fold).all()
    g = {k: (v[["x1", "y1", "x2", "y2"]].to_numpy(float), v.score.to_numpy(float)) for k, v in preds.groupby("orig_path")}
    empty = (np.zeros((0, 4)), np.zeros(0))
    K = np.array([len(GT[p]) for p in idx.orig_path])
    assert (K == idx.n_gt.to_numpy()).all(), "index n_gt disagrees with manifest gt_boxes"
    N = np.zeros((len(idx), G.size))  # boxes with score >= lam
    L = {t: np.zeros((len(idx), G.size)) for t in IOUS}
    for i, p in enumerate(idx.orig_path):
        b, s = g.get(p, empty)
        N[i] = len(s) - np.searchsorted(np.sort(s), G, side="left")
        if K[i]:
            for t in IOUS:
                L[t][i] = np.searchsorted(np.sort(matched_scores(GT[p], b, s, t)), G, side="left") / K[i]
    # prefix-property check against the reference implementation; fall back to direct evaluation if it fails
    pos = np.flatnonzero(K > 0)
    chk = rng.choice(pos, min(200, len(pos)), replace=False)
    lam_idx = np.unique(np.r_[np.searchsorted(G, [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 0.999]), rng.integers(0, G.size, 4)])
    for t in IOUS:
        ref = np.array([[miss_fraction(GT[idx.orig_path[i]], *g.get(idx.orig_path[i], empty), G[j], thr=t) for j in lam_idx]
                        for i in chk])
        if not np.allclose(ref, L[t][np.ix_(chk, lam_idx)], atol=1e-12):
            print(f"WARNING {d.name} IoU {t}: prefix property failed, falling back to direct evaluation (slow)")
            for i in pos:
                L[t][i] = [miss_fraction(GT[idx.orig_path[i]], *g.get(idx.orig_path[i], empty), lam, thr=t) for lam in G]
    assert all((np.diff(L[t], axis=1) >= -1e-12).all() for t in IOUS)  # non-decreasing in lam
    return idx, K, N, L, len(chk) * len(lam_idx)


def crc_idx(S, n, alpha):
    """Vectorized metrics.crc_lambda: S (R, |G|) summed losses, n (R,) calibration sizes -> (lambda index, degenerate)."""
    ok = (S + 1.0) / (n[:, None] + 1.0) <= alpha
    anyok = ok.any(1)
    return np.where(anyok, G.size - 1 - np.argmax(ok[:, ::-1], 1), 0), ~anyok


def evaluate(W, j, K, L, N):
    """W (R, n) test weights over images, j (R,) lambda index -> realized risk, micro miss, FA/neg frame, FP/pos frame."""
    Lj, Nj = L[:, j].T, N[:, j].T
    Wp, Wn = W * (K > 0), W * (K == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(risk=(Wp * Lj).sum(1) / Wp.sum(1), micro_miss=(Wp * Lj * K).sum(1) / (Wp * K).sum(1),
                    fa_neg=(Wn * Nj).sum(1) / Wn.sum(1), fp_pos=(Wp * (Nj - K * (1 - Lj))).sum(1) / Wp.sum(1),
                    n_test=Wp.sum(1))


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
    out, n_checked = [], 0
    runs = sorted(d for d in root.glob("C*_seed*") if next(d.glob("preds_*.parquet"), None))
    print(f"{len(runs)} runs under {root}: {[d.name for d in runs]}")
    for d in runs:
        fold, seed = d.name.split("_seed"); seed = int(seed); fi = int(fold[1:])
        rng = np.random.default_rng([2026, fi, seed])
        idx, K, N, L, nc = load_run(d, fold, GT, rng); n_checked += nc
        cal, tgt = (idx.role == "cal").to_numpy(), (idx.role == "target").to_numpy()
        pos = K > 0
        clus = idx.cluster_id.to_numpy()
        # --- random structures shared by both IoUs and all alphas (common random numbers)
        cal_cl = np.unique(clus[cal]); cl_ix = {c: i for i, c in enumerate(cal_cl)}
        img_cl = np.array([cl_ix.get(c, -1) for c in clus])
        cl_centre = pd.Series(idx.center.to_numpy()[cal], index=clus[cal]).groupby(level=0).first().reindex(cal_cl).to_numpy()
        boot = np.zeros((a.boot, len(cal_cl)))
        cl_pos = np.isin(cal_cl, clus[cal & pos])
        for c in np.unique(cl_centre):  # bootstrap of POSITIVE clusters, stratified by source centre (every centre keeps
            m = np.flatnonzero((cl_centre == c) & cl_pos)  # n_c > 0; negative-only clusters carry no calibration loss)
            for r in range(a.boot):
                np.add.at(boot[r], rng.choice(m, len(m)), 1)
        Wboot = np.where(cal, boot[:, img_cl], 0)  # (B, n) image weights
        half = np.array([rng.permutation(len(cal_cl)) < len(cal_cl) // 2 for _ in range(a.splits)])  # (S, clusters)
        Wa, Wb = np.where(cal, half[:, img_cl], 0).astype(float), np.where(cal, ~half[:, img_cl], 0).astype(float)
        # diagnostic: image-level 50/50 splits (ignores clusters) -> exactly exchangeable, mean risk must be <= alpha
        ci = np.flatnonzero(cal); WaI = np.zeros((a.splits, len(idx)))
        for r in range(a.splits):
            WaI[r, ci[rng.permutation(len(ci)) < len(ci) // 2]] = 1
        WbI = cal - WaI
        draws = json.loads((ROOT / "splits" / f"target_draws_{fold}.json").read_text())
        tcl = set(clus[tgt])
        Wdraw = {}
        for bgt in BUDGETS:
            dl = draws[str(bgt)][:a.draws]
            assert all(set(x) <= tcl for x in dl)
            M = np.array([np.isin(clus, x) & tgt for x in dl], float)
            assert ((M * pos).sum(1) >= bgt).all()
            kpos = np.array([len(set(clus[(m > 0) & pos])) for m in M])
            Wdraw[bgt] = (M, kpos)
        centres = sorted(set(idx.center[cal]))
        T = tgt[None].astype(float)
        fa_ok = fold in NEG_CENTRES

        def rows(strategy, alpha, t, j, deg, ev, n_cal, k_cal, budget=0, rep=None, binding=None):
            R = len(j)
            r = dict(model=root.name, fold=fold, seed=seed, iou=t, alpha=alpha, strategy=strategy, budget=budget,
                     rep=np.arange(R) if rep is None else rep, lam=G[j], degenerate=deg, lam_max=j == G.size - 1,
                     n_cal=n_cal, k_cal=k_cal, binding=binding if binding is not None else "", **ev)
            r["violation"] = r["risk"] > alpha
            if not fa_ok and not strategy.startswith("in_domain"):
                r["fa_neg"] = np.full(R, np.nan)
            out.append(pd.DataFrame(r))

        for t in IOUS:
            Lt = L[t]
            Lp = Lt  # negatives have L = 0; weights below are restricted to positives anyway
            for alpha in ALPHAS:
                # in-domain reference
                j, deg = crc_idx((Wa * pos) @ Lp, (Wa * pos).sum(1), alpha)
                rows("in_domain", alpha, t, j, deg, evaluate(Wb, j, K, Lt, N), (Wa * pos).sum(1),
                     np.array([len(set(clus[(w > 0) & pos])) for w in Wa]))
                j, deg = crc_idx((WaI * pos) @ Lp, (WaI * pos).sum(1), alpha)
                rows("in_domain_image", alpha, t, j, deg, evaluate(WbI, j, K, Lt, N), (WaI * pos).sum(1),
                     np.array([len(set(clus[(w > 0) & pos])) for w in WaI]))
                # pooled + bootstrap
                w0 = (cal & pos).astype(float)
                lam_ref, deg_ref = crc_lambda(Lt[cal & pos], alpha, G)
                for name, W in (("pooled", w0[None]), ("pooled_boot", Wboot * pos)):
                    j, deg = crc_idx(W @ Lp, W.sum(1), alpha)
                    if name == "pooled":
                        assert G[j[0]] == lam_ref and deg[0] == deg_ref, "crc_idx disagrees with metrics.crc_lambda"
                    rows(name, alpha, t, j, deg, evaluate(np.repeat(T, len(j), 0), j, K, Lt, N), W.sum(1),
                         np.full(len(j), len(set(clus[cal & pos]))), rep=np.array([-1]) if name == "pooled" else None)
                # per-source worst case + bootstrap
                for name, W in (("worst_case", w0[None]), ("worst_case_boot", Wboot * pos)):
                    js, ds = zip(*[crc_idx((W * (idx.center == c).to_numpy()) @ Lp,
                                           (W * (idx.center == c).to_numpy()).sum(1), alpha) for c in centres])
                    js, ds = np.array(js), np.array(ds)  # (centres, R)
                    j = js.min(0); deg = ds.any(0); bind = np.array(centres)[js.argmin(0)]
                    rows(name, alpha, t, j, deg, evaluate(np.repeat(T, len(j), 0), j, K, Lt, N), W.sum(1),
                         np.full(len(j), len(set(clus[cal & pos]))), rep=np.array([-1]) if name == "worst_case" else None,
                         binding=bind)
                # target few-shot
                for bgt, (M, kpos) in Wdraw.items():
                    Mp = M * pos
                    j, deg = crc_idx(Mp @ Lp, Mp.sum(1), alpha)
                    rows("few_shot", alpha, t, j, deg, evaluate(T * (1 - M), j, K, Lt, N), Mp.sum(1), kpos, budget=bgt)
        print(f"  {d.name}: cal pos {int((cal & pos).sum())}, target pos {int((tgt & pos).sum())}, "
              f"target neg {int((tgt & ~pos).sum())}")
    print(f"prefix property: L_i(lam) from matched scores == metrics.miss_fraction on {n_checked} (image, lam) pairs "
          f"per IoU ({len(runs)} runs x 200 images) - OK")

    long = pd.concat(out, ignore_index=True)
    ev = ROOT / "outputs" / "eval"; ev.mkdir(parents=True, exist_ok=True)
    long.to_parquet(ev / f"conformal_long{sfx}.parquet", index=False)

    # summary: per seed and over seeds ('all')
    keys = ["model", "fold", "iou", "alpha", "strategy", "budget"]
    agg = dict(n_rows=("risk", "size"), risk_mean=("risk", "mean"), risk_q05=("risk", lambda x: x.quantile(0.05)),
               risk_q95=("risk", lambda x: x.quantile(0.95)), micro_miss_mean=("micro_miss", "mean"),
               viol_freq=("violation", "mean"), degenerate_frac=("degenerate", "mean"), lam_max_frac=("lam_max", "mean"),
               lam_median=("lam", "median"), fa_neg_mean=("fa_neg", "mean"), fp_pos_mean=("fp_pos", "mean"),
               n_cal_median=("n_cal", "median"), k_cal_median=("k_cal", "median"), n_test_median=("n_test", "median"),
               binding_mode=("binding", lambda x: x.mode().iloc[0] if (x != "").any() else ""))
    per_seed = long.groupby(keys + ["seed"]).agg(**agg).reset_index()
    allseeds = long.groupby(keys).agg(**agg).reset_index().assign(seed="all")
    allseeds = allseeds.merge(long.groupby(keys).seed.nunique().rename("n_seeds").reset_index(), on=keys)
    summ = pd.concat([allseeds, per_seed.assign(seed=per_seed.seed.astype(str))], ignore_index=True)
    summ.to_csv(ev / f"conformal_summary{sfx}.csv", index=False)
    print(f"wrote {ev / f'conformal_long{sfx}.parquet'} ({len(long)} rows), {ev / f'conformal_summary{sfx}.csv'}")

    # in-domain check: CRC bounds E[risk] under exchangeability. Image-level splits are exactly exchangeable (bug
    # check); cluster-level splits are the honest in-domain reference (frames of one video are not exchangeable).
    pd.set_option("display.width", 250)
    for strat, what in (("in_domain_image", "image-level splits (exchangeable -> bug check)"),
                        ("in_domain", "cluster-level splits (reference)")):
        ind = per_seed[per_seed.strategy == strat]
        ok = ind.degenerate_frac == 0
        bad = ind[ok & (ind.risk_mean > ind.alpha + 0.02)]
        print(f"\nIN-DOMAIN CHECK, {what}: {len(ind)} (fold, seed, IoU, alpha) cells, {int(ok.sum())} without degenerate "
              f"splits; max(mean risk - alpha) there = {(ind.risk_mean - ind.alpha)[ok].max():+.4f}; cells with mean risk "
              f"> alpha + 0.02: {len(bad)}" + ("  -> PASSED" if bad.empty else "  -> FLAGGED"))
        if not bad.empty:
            print(bad[["fold", "seed", "iou", "alpha", "risk_mean", "viol_freq"]].to_string(index=False))
        print(f"  cells with degenerate splits (risk includes the miss floor; not a CRC failure): "
              f"{sorted(set(zip(ind[~ok].iou, ind[~ok].alpha)))}")

    # degenerate counts
    print("\nDEGENERATE fraction (all seeds), by strategy x IoU x alpha (mean over folds):")
    print(allseeds.pivot_table(index=["strategy", "budget", "iou"], columns="alpha", values="degenerate_frac").round(3).to_string())

    # compact table, IoU 0.1
    s = allseeds[allseeds.iou == 0.1].copy()
    s["col"] = s.strategy.map({"in_domain": "in-domain", "pooled": "pooled", "worst_case": "worst-case",
                               "few_shot": "few-shot"}) + s.budget.map(lambda b: f" {b}" if b else "")
    boot = s[s.strategy.str.endswith("_boot")].assign(col=lambda x: x.strategy.str.replace("_boot", "").map(
        {"pooled": "pooled", "worst_case": "worst-case"})).set_index(["fold", "alpha", "col"]).viol_freq
    s = s.dropna(subset=["col"]).set_index(["fold", "alpha", "col"])
    cols = ["in-domain", "pooled", "worst-case", "few-shot 20", "few-shot 50", "few-shot 100"]
    for alpha in (0.10, 0.20):
        print(f"\nIoU 0.1, alpha = {alpha:.2f}: mean realized target risk (violation freq). in-domain/few-shot: over "
              f"splits/draws x seeds; pooled/worst-case: point estimate mean over seeds (violation freq over "
              f"{a.boot} cluster-bootstrap resamples x seeds)")
        tab = pd.DataFrame({c: [f"{s.loc[(f, alpha, c)].risk_mean:.3f} ({(boot if c in ('pooled', 'worst-case') else s.viol_freq).loc[(f, alpha, c)]:.2f})"
                                for f in sorted(s.index.levels[0])] for c in cols}, index=sorted(s.index.levels[0]))
        print(tab.to_string())


if __name__ == "__main__":
    main()
