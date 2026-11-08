"""Experiment B1 on CVC-ClinicDB (612 frames, 29 videos; Bernal et al. 2015): does a random FRAME split overstate the
conformal guarantee compared with a VIDEO split? Plus external validation of PolypGen-calibrated CRC (ClinicDB = an
unseen 7th centre) and detection quality. Loss, matching and CRC threshold: docs/metric.md, reused from
scripts/conformal.py (matched_scores, crc_idx, G, ALPHAS, IOUS) and scripts/metrics.py (not modified).

GT: ClinicDB has no official boxes. Polyp = 8-connected component of (grey mask > mask_threshold), components smaller
than min_area dropped (configs/data.yaml); GT box = tight half-open box of the component (docs/metric.md sec. 1).

Per model (18 = C1..C6 x seed 0..2, outputs/clinicdb/preds_<fold>_seed<k>.parquet), IoU in {0.1, 0.5}, alpha in ALPHAS,
R draws (identical across models/IoU/alpha = common random numbers):
 (i)  video_split    : shuffle the 29 videos, add whole videos until >= n positive frames (n in N_CAL); image CRC on
                       them, test = remaining videos.
      frame_split    : same draw r, n_r = that video split's calibration size; n_r random positive frames (any video),
                       test = the other frames. Exchangeable by construction -> mean test risk must be <= alpha.
      video_split_cl : same video draws, cluster-level CRC (unit = video, video loss = mean over its positive frames;
                       as scripts/conformal_cluster.py).
 (ii) pooled_source  : lambda-hat of conformal.py strategy "pooled" (all positive source-cal frames of that PolypGen fold,
                       read from outputs/eval/conformal_long.parquet; recomputed for one run as a check) -> all ClinicDB.
      fs_a / fs_b / fs_c : k in K_VIDEOS ClinicDB videos drawn uniformly; (a) cluster CRC, (b) image CRC on one random
                       positive frame per drawn video, (c) image CRC on all their positive frames; test = other videos.
 (iii) detection     : AP50 and max recall (every box, conf >= 0.001) at IoU 0.1 / 0.5.
risk_fr = mean per-frame miss fraction over the test positive frames; risk_cl = mean over test videos of the per-video
mean (each video counts once). viol_* = risk_* > alpha; exc05_* = risk_* > alpha + 0.05 (tail). degenerate = no lambda
qualifies (lambda-hat = min grid). Summary run="all": mean over the 18 per-model means, *_sd_models = s.d. across models,
*_q05/_q95/_sd_draws pooled over models x draws.

Usage: .venv/Scripts/python.exe -I scripts/clinicdb_crc.py [--draws 500]
Writes outputs/eval/clinicdb_long.parquet, outputs/eval/clinicdb_summary.csv."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conformal import matched_scores, crc_idx, load_run, clean, G, ALPHAS, IOUS  # noqa: E402
from metrics import miss_fraction, crc_lambda  # noqa: E402

DATA = ROOT / "data" / "raw" / "cvc-clinicdb"
N_CAL, K_VIDEOS = (300, 100), (5, 10, 20)


def clinicdb_gt():
    """Frame table (frame_id, sequence_id sorted by frame) and GT boxes per frame from the masks."""
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    meta = pd.read_csv(DATA / "metadata.csv").sort_values("frame_id").reset_index(drop=True)
    gt, n_small = [], 0
    for p in meta.png_mask_path:
        m = cv2.imread(str(DATA / p), cv2.IMREAD_GRAYSCALE)
        assert m.shape == (288, 384), p
        n, _, st, _ = cv2.connectedComponentsWithStats((m > cfg["mask_threshold"]).astype(np.uint8), connectivity=8)
        keep = [i for i in range(1, n) if st[i, 4] >= cfg["min_area"]]
        n_small += n - 1 - len(keep)
        gt.append(np.array([[st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3]] for i in keep], float).reshape(-1, 4))
    return meta, gt, n_small


def losses(gt, preds):
    """L[t] (frames, |G|) per-frame miss fraction via matched scores (prefix property, as conformal.load_run) and
    the matched score of every GT polyp per IoU (for AP / max recall)."""
    g = {k: (v[["x1", "y1", "x2", "y2"]].to_numpy(float), v.score.to_numpy(float)) for k, v in preds.groupby("frame_id")}
    empty = (np.zeros((0, 4)), np.zeros(0))
    L, ms = {t: np.zeros((len(gt), G.size)) for t in IOUS}, {t: [] for t in IOUS}
    for i, b in enumerate(gt):
        bx, s = g.get(i + 1, empty)
        for t in IOUS:
            m = matched_scores(b, bx, s, t); ms[t].append(m)
            if len(b):
                L[t][i] = np.searchsorted(np.sort(m), G, side="left") / len(b)
    return L, {t: np.concatenate(v) for t, v in ms.items()}, g


def ap50_maxrec(ms, scores, n_gt):
    """AP (all-point interpolated) from matched scores: at threshold s, TP = #GT with matched score >= s,
    #detections = #boxes with score >= s (one PR point per distinct score; ties merged)."""
    s = np.unique(scores)[::-1]
    tp = len(ms) - np.searchsorted(np.sort(ms), s, side="left")  # matched scores >= s (unmatched = 0 < any score)
    nd = len(scores) - np.searchsorted(np.sort(scores), s, side="left")
    rec, prec = tp / n_gt, tp / nd
    mprec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * mprec)), float((ms > 0).sum() / n_gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=500)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    R = a.draws
    meta, GT, n_small = clinicdb_gt()
    K = np.array([len(b) for b in GT]); pos = K > 0
    vid = meta.sequence_id.to_numpy()
    uv, vi = np.unique(vid, return_inverse=True); C = len(uv)
    m = np.bincount(vi, weights=pos, minlength=C)
    print(f"GT: {len(meta)} frames, {C} videos; polyps per frame {dict(zip(*np.unique(K, return_counts=True)))}; "
          f"components < min_area dropped: {n_small}; positive frames per video min/median/max "
          f"{m.min():.0f}/{np.median(m):.0f}/{m.max():.0f}")
    print(f"  frames with 0 polyps: {meta.frame_id[K == 0].tolist()}; with > 1: {meta.frame_id[K > 1].tolist()}")
    cpos = m > 0
    A = np.zeros((C, len(meta))); A[vi[pos], np.flatnonzero(pos)] = 1.0 / m[vi[pos]]  # frame -> video mean (positives)
    Vind = (vi[None] == np.arange(C)[:, None]).astype(float)  # (C, frames)

    # --- draws (shared by all models, IoU, alpha)
    rng = np.random.default_rng([2026, 1101])
    draws = {}
    P = np.flatnonzero(pos)
    for n in N_CAL:
        Vc = np.zeros((R, C))
        for r in range(R):
            acc = 0
            for c in rng.permutation(np.flatnonzero(cpos)):
                Vc[r, c] = 1; acc += m[c]
                if acc >= n:
                    break
        Wv = Vc[:, vi] * pos  # video-split calibration frames (positives)
        nr = Wv.sum(1).astype(int)
        Wf = np.zeros((R, len(meta)))
        for r in range(R):
            Wf[r, rng.permutation(P)[:nr[r]]] = 1
        draws[n] = dict(Vc=Vc, Wv=Wv, Wf=Wf, Vt=(1 - Vc) * cpos, nr=nr)
    fs = {}
    for k in K_VIDEOS:
        sel = np.array([rng.choice(np.flatnonzero(cpos), k, replace=False) for _ in range(R)])
        Vd = np.zeros((R, C)); np.put_along_axis(Vd, sel, 1.0, 1)
        Wone = np.zeros((R, len(meta)))
        for r in range(R):
            Wone[r, [rng.choice(np.flatnonzero(pos & (vi == c))) for c in sel[r]]] = 1
        fs[k] = dict(Vd=Vd, Wone=Wone, Wall=Vd[:, vi] * pos, Vt=(1 - Vd) * cpos)

    def ev(Vt_or_W, j, Lt, by_video):
        """Test risks for lambda index j (R,). Test set given as video weights (R, C) or frame weights (R, frames)."""
        W = (Vt_or_W[:, vi] if by_video else Vt_or_W) * pos
        Lj = Lt[:, j].T  # (R, frames)
        num, den = (W * Lj) @ Vind.T, W @ Vind.T  # per video
        with np.errstate(invalid="ignore"):
            rcl = np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), 1)
        return (W * Lj).sum(1) / W.sum(1), rcl, W.sum(1), (den > 0).sum(1)

    pooled = pd.read_parquet(ROOT / "outputs/eval/conformal_long.parquet",
                             columns=["model", "fold", "seed", "iou", "alpha", "strategy", "lam", "degenerate"])
    pooled = pooled[(pooled.strategy == "pooled") & (pooled.model == "kaggle")].set_index(["fold", "seed", "iou", "alpha"])
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    GTpg = {clean(p): np.asarray(json.loads(gb), float).reshape(-1, 4) for p, gb in zip(man.path, man.gt_boxes)}

    out, det, n_crc_chk, n_mf_chk = [], [], 0, 0
    runs = sorted((ROOT / "outputs" / "clinicdb").glob("preds_C*_seed*.parquet"))
    assert len(runs) == 18, len(runs)
    for pq in runs:
        run = pq.stem.removeprefix("preds_"); fold, seed = run.split("_seed"); seed = int(seed)
        preds = pd.read_parquet(pq)
        assert (preds.groupby("frame_id").sequence_id.first() == meta.set_index("frame_id").sequence_id.reindex(
            preds.frame_id.unique())).all()
        L, ms, g = losses(GT, preds)
        # checks: monotone; matched-score losses == metrics.miss_fraction on random frames x lambdas
        crng = np.random.default_rng([2026, int(fold[1:]), seed, 7])
        chk, lj = crng.choice(P, 60, replace=False), crng.integers(0, G.size, 12)
        for t in IOUS:
            assert (np.diff(L[t], axis=1) >= -1e-12).all()
            ref = [[miss_fraction(GT[i], *g.get(i + 1, (np.zeros((0, 4)), np.zeros(0))), G[j], thr=t) for j in lj] for i in chk]
            assert np.allclose(ref, L[t][np.ix_(chk, lj)]); n_mf_chk += ref.__len__() * len(lj)
            ap50, mr = ap50_maxrec(ms[t], preds.score.to_numpy(float), K.sum())
            det.append(dict(run=run, fold=fold, seed=seed, iou=t, strategy="detection", AP=ap50, max_recall=mr,
                            n_polyps=int(K.sum())))
        if run == "C1_seed0":  # pooled lambda-hat from the parquet == recomputed from the PolypGen source cal set
            idx, Kp, _, Lp, _ = load_run(ROOT / "outputs/kaggle" / run, fold, GTpg, np.random.default_rng(0))
            cm = ((idx.role == "cal").to_numpy()) & (Kp > 0)
            for t in IOUS:
                for alpha in ALPHAS:
                    assert (crc_lambda(Lp[t][cm], alpha, G)[0] == pooled.loc[(fold, seed, t, alpha)].lam), (t, alpha)
            print("pooled-source lambda-hat recomputed for C1_seed0 (both IoU, all alpha) == conformal_long.parquet")

        def rows(strategy, budget, alpha, t, j, deg, test, by_video, n_cal, k_cal):
            rf, rc, nt, kt = ev(test, j, L[t], by_video)
            out.append(pd.DataFrame(dict(run=run, fold=fold, seed=seed, iou=t, alpha=alpha, strategy=strategy,
                                         budget=budget, rep=np.arange(len(j)), lam=G[j], degenerate=deg, risk_fr=rf,
                                         risk_cl=rc, viol_fr=rf > alpha, viol_cl=rc > alpha, n_cal=n_cal, k_cal=k_cal,
                                         n_test=nt, k_test=kt)))

        for t in IOUS:
            Lt, Lc = L[t], A @ L[t]
            for alpha in ALPHAS:
                for n, d in draws.items():
                    kc = d["Vc"].sum(1)
                    j, deg = crc_idx(d["Wv"] @ Lt, d["Wv"].sum(1), alpha)
                    rows("video_split", n, alpha, t, j, deg, d["Vt"], True, d["nr"], kc)
                    jf, degf = crc_idx(d["Wf"] @ Lt, d["Wf"].sum(1), alpha)
                    rows("frame_split", n, alpha, t, jf, degf, (1 - d["Wf"]) * pos, False, d["nr"],
                         ((d["Wf"] @ Vind.T) > 0).sum(1))
                    jc, degc = crc_idx(d["Vc"] @ Lc, kc, alpha)
                    rows("video_split_cl", n, alpha, t, jc, degc, d["Vt"], True, d["nr"], kc)
                    for r in range(2):  # lambda-hat == metrics.crc_lambda on explicit subsets
                        assert (G[j[r]], deg[r]) == crc_lambda(Lt[d["Wv"][r] > 0], alpha, G)
                        assert (G[jf[r]], degf[r]) == crc_lambda(Lt[d["Wf"][r] > 0], alpha, G)
                        assert (G[jc[r]], degc[r]) == crc_lambda(Lc[d["Vc"][r] > 0], alpha, G)
                        n_crc_chk += 3
                lam = pooled.loc[(fold, seed, t, alpha)].lam
                j0 = np.array([int(np.flatnonzero(np.isclose(G, lam))[0])])
                rows("pooled_source", 0, alpha, t, j0, np.array([pooled.loc[(fold, seed, t, alpha)].degenerate]),
                     cpos[None].astype(float), True, np.array([np.nan]), np.array([np.nan]))
                for k, d in fs.items():
                    ja, da = crc_idx(d["Vd"] @ Lc, d["Vd"].sum(1), alpha)
                    rows("fs_a", k, alpha, t, ja, da, d["Vt"], True, d["Wall"].sum(1), d["Vd"].sum(1))
                    jb, db = crc_idx(d["Wone"] @ Lt, d["Wone"].sum(1), alpha)
                    rows("fs_b", k, alpha, t, jb, db, d["Vt"], True, d["Wone"].sum(1), d["Vd"].sum(1))
                    jc, dc = crc_idx(d["Wall"] @ Lt, d["Wall"].sum(1), alpha)
                    rows("fs_c", k, alpha, t, jc, dc, d["Vt"], True, d["Wall"].sum(1), d["Vd"].sum(1))
        print(f"  {run}: {len(preds)} boxes; max recall IoU0.1/0.5 = {det[-2]['max_recall']:.3f}/{det[-1]['max_recall']:.3f}",
              flush=True)
    print(f"SANITY: losses non-decreasing in lambda (18 runs x 2 IoU) - OK; matched-score loss == metrics.miss_fraction "
          f"on {n_mf_chk} (frame, lambda, IoU) cases - OK; crc_idx lambda-hat == metrics.crc_lambda on {n_crc_chk} draws - OK")

    long = pd.concat(out, ignore_index=True)
    long["exc05_fr"], long["exc05_cl"] = long.risk_fr > long.alpha + 0.05, long.risk_cl > long.alpha + 0.05  # tail
    ev_dir = ROOT / "outputs" / "eval"
    long.to_parquet(ev_dir / "clinicdb_long.parquet", index=False)
    keys = ["iou", "alpha", "strategy", "budget"]
    q = lambda p: (lambda x: x.quantile(p))
    agg = dict(n_rows=("risk_fr", "size"), risk_fr_mean=("risk_fr", "mean"), risk_fr_q05=("risk_fr", q(.05)),
               risk_fr_q95=("risk_fr", q(.95)), risk_cl_mean=("risk_cl", "mean"), risk_cl_q05=("risk_cl", q(.05)),
               risk_cl_q95=("risk_cl", q(.95)), viol_fr=("viol_fr", "mean"), viol_cl=("viol_cl", "mean"),
               risk_fr_sd_draws=("risk_fr", "std"), risk_cl_sd_draws=("risk_cl", "std"),
               exc05_fr=("exc05_fr", "mean"), exc05_cl=("exc05_cl", "mean"),
               degenerate_frac=("degenerate", "mean"), lam_median=("lam", "median"), n_cal_median=("n_cal", "median"),
               k_cal_median=("k_cal", "median"), n_test_median=("n_test", "median"), k_test_median=("k_test", "median"))
    per = long.groupby(["run", "fold", "seed"] + keys).agg(**agg).reset_index()
    # over models: mean and s.d. ACROSS the 18 per-model means (what the figure's error bars show)
    allm = per.groupby(keys).agg(n_models=("run", "nunique"), **{c: (c, "mean") for c in per.columns if c.endswith(("_mean", "_frac")) or c.startswith(("viol", "exc05"))},
                                 risk_fr_sd_models=("risk_fr_mean", "std"), risk_cl_sd_models=("risk_cl_mean", "std"),
                                 n_cal_median=("n_cal_median", "median"), k_cal_median=("k_cal_median", "median"),
                                 k_test_median=("k_test_median", "median")).reset_index().assign(run="all")
    # draw-level quantiles and s.d. pooled over all models x draws (per-model values are in the per-model rows)
    pool = long.groupby(keys).agg(**{k: v for k, v in agg.items() if k.endswith(("_q05", "_q95", "_sd_draws"))})
    allm = allm.drop(columns=[c for c in pool.columns if c in allm.columns]).merge(pool.reset_index(), on=keys)
    detd = pd.DataFrame(det)
    detm = detd.groupby("iou").agg(AP=("AP", "mean"), max_recall=("max_recall", "mean"), AP_sd=("AP", "std"),
                                   max_recall_sd=("max_recall", "std")).reset_index().assign(run="all", strategy="detection")
    summ = pd.concat([allm, per, detm, detd], ignore_index=True)
    summ.to_csv(ev_dir / "clinicdb_summary.csv", index=False)
    print(f"wrote {ev_dir / 'clinicdb_long.parquet'} ({len(long)} rows), {ev_dir / 'clinicdb_summary.csv'}")

    # sanity: frame split is exchangeable by construction -> per-model mean risk <= alpha + tol (non-degenerate cells)
    fsp = per[(per.strategy == "frame_split") & (per.degenerate_frac == 0)]
    ex = fsp.risk_fr_mean - fsp.alpha
    print(f"FRAME-SPLIT CHECK: {len(fsp)}/{(per.strategy == 'frame_split').sum()} non-degenerate (model, IoU, alpha, n) "
          f"cells; max(mean risk - alpha) = {ex.max():+.4f}")
    assert ex.max() <= 0.01, fsp.loc[ex.idxmax()]
    vsp = per[(per.strategy == "video_split") & (per.degenerate_frac == 0)]
    print(f"VIDEO-SPLIT: max(mean risk - alpha) over {len(vsp)} non-degenerate cells = {(vsp.risk_fr_mean - vsp.alpha).max():+.4f}")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nDEGENERATE fraction (mean over models):")
    print(allm.pivot_table(index=["iou", "strategy", "budget"], columns="alpha", values="degenerate_frac").round(2).to_string())
    s = allm.set_index(["iou", "alpha", "strategy", "budget"])
    for t in IOUS:
        for alpha in (0.10, 0.20):
            print(f"\nIoU {t}, alpha {alpha:.2f}: mean over 18 models of [mean test risk - alpha] (s.d. across models) "
                  f"[violation freq] {{degenerate frac}}; frame-avg risk / video-avg risk")
            tab = {}
            for st, b in [(x, n) for n in N_CAL for x in ("frame_split", "video_split", "video_split_cl")] + \
                         [("pooled_source", 0)] + [(f"fs_{z}", k) for k in K_VIDEOS for z in "abc"]:
                x = s.loc[(t, alpha, st, b)]
                tab[f"{st} {b}"] = dict(
                    frame=f"{x.risk_fr_mean - alpha:+.3f} ({x.risk_fr_sd_models:.3f}) [{x.viol_fr:.2f}]",
                    video=f"{x.risk_cl_mean - alpha:+.3f} ({x.risk_cl_sd_models:.3f}) [{x.viol_cl:.2f}]",
                    spread_fr=f"sd {x.risk_fr_sd_draws:.3f} q95 {x.risk_fr_q95:.3f} P(>a+.05) {x.exc05_fr:.2f}", degenerate=f"{x.degenerate_frac:.2f}", n_cal=x.n_cal_median, k_cal=x.k_cal_median,
                    k_test=x.k_test_median)
            print(pd.DataFrame(tab).T.to_string())
    print("\nDETECTION on ClinicDB (mean, s.d. over 18 models; AP at the given IoU, max recall keeping every box):")
    print(detm.round(3).to_string(index=False))
    print(detd.pivot_table(index="fold", columns="iou", values=["AP", "max_recall"]).round(3).to_string())


if __name__ == "__main__":
    main()
