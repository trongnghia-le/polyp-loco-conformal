"""Second-round review checks on the STORED YOLO11s predictions (18 runs) - no training, no new data.
Kept under paper/review/ (local, manuscript-only). Imports the project functions unchanged (conformal.load_run,
crc_idx, matched_scores; clinicdb_crc.clinicdb_gt / losses; metrics.iou). IoU >= 0.1 unless stated.
Writes outputs/eval/revision2_checks.csv (long format: check, fold, key, value) and prints a summary.

 A  bound  : E[(Rhat_{k+1}(lam_min) - alpha)_+] for target recalibration with k videos (designs a: cluster loss;
             b: one random positive frame per cluster), i.e. the excess over alpha that the CRC proof allows when
             keeping every box does not reach alpha (miss floor). Monte Carlo, 4000 draws of k+1 positive target
             clusters without replacement, k capped as conformal_cluster.py (n_pos_clusters - 10). Mean over seeds.
 B  dom    : dominance of single target videos: Kish effective number of videos for the frame average,
             (sum m)^2 / sum m^2, largest positive video share; pooled frame-/video-averaged risk at the fixed pooled
             threshold with the largest positive target video removed (hierarchical target x seed bootstrap, B=1000).
 C  nested : interval for the EXPECTED target risk of the pooled procedure: outer = R1 target-video bootstrap
             (same resamples, seed drawn per resample), inner = mean over the 200 stored source-calibration
             bootstrap thresholds (conformal_long 'pooled_boot', conformal_cluster_long 'pooled_cl_boot').
 D  ap     : AP over positive target frames and over all target frames at IoU 0.1 and 0.5 (all-point, from matched
             scores as clinicdb_crc.ap50_maxrec); the IoU 0.5 value is compared with revision_R5 AP50_pos / AP50_all.
 E  hits   : at the pooled threshold (alpha 0.10), IoU >= 0.1 hits: share with IoU < 0.5, with box area > 2x / > 4x the
             ground-truth box, and with box centre inside the ground-truth box. Matched count asserted == (1-L)K.
 F  clinic : CVC-ClinicDB pooled-source thresholds: hierarchical video (29) x detector (18) bootstrap, B=1000;
             per-detector share of resamples above alpha.
 G  neg    : share of polyp-free target frames that lie in positive videos (could be inside a recalibration draw).
Usage: .venv/Scripts/python.exe -I scripts/revision2_checks.py"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conformal import clean, G, IOUS, load_run, matched_scores  # noqa: E402
from metrics import iou as box_iou  # noqa: E402

EV, OUT = ROOT / "outputs" / "eval", ROOT / "outputs" / "eval" / "revision2_checks.csv"
B, R_MC, A0 = 1000, 4000, 0.10
rows = []
put = lambda check, fold, key, value: rows.append(dict(check=check, fold=fold, key=key, value=float(value)))


def ap_from_matched(ms, scores, n_gt):
    s = np.unique(scores)[::-1]
    tp = len(ms) - np.searchsorted(np.sort(ms), s, side="left")
    nd = len(scores) - np.searchsorted(np.sort(scores), s, side="left")
    rec, prec = tp / n_gt, tp / nd
    mprec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * mprec))


def greedy_pairs(gt, boxes, scores, lam, thr):
    """Same greedy one-to-one rule as metrics.miss_fraction; returns (box, gt) index pairs and IoUs."""
    order = np.argsort(-scores, kind="stable")
    order = order[scores[order] >= lam]
    m = box_iou(boxes[order], gt)
    free, out = np.ones(len(gt), bool), []
    for i, row in enumerate(m):
        row = np.where(free, row, -1.0)
        j = row.argmax()
        if row[j] >= thr:
            free[j] = False; out.append((order[i], j, row[j]))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    GT = {clean(p): np.asarray(json.loads(gb), float).reshape(-1, 4) for p, gb in zip(man.path, man.gt_boxes)}
    cs = pd.read_csv(EV / "conformal_summary.csv", dtype={"seed": str})
    cc = pd.read_csv(EV / "conformal_cluster_summary.csv", dtype={"seed": str})
    r5 = pd.read_csv(EV / "revision_R5_detection_floor.csv")
    lo = pd.read_parquet(EV / "conformal_long.parquet", columns=["fold", "seed", "iou", "alpha", "strategy", "rep", "lam"])
    lo = lo[(lo.strategy == "pooled_boot") & (lo.iou == 0.1) & (lo.alpha == A0)]
    lc = pd.read_parquet(EV / "conformal_cluster_long.parquet", columns=["fold", "seed", "iou", "alpha", "strategy", "rep", "lam"])
    lc = lc[(lc.strategy == "pooled_cl_boot") & (lc.iou == 0.1) & (lc.alpha == A0)]
    runs = sorted(d for d in (ROOT / "outputs" / "kaggle").glob("C*_seed*") if next(d.glob("preds_*.parquet"), None))
    assert len(runs) == 18
    acc = {}  # (fold) -> per-seed bootstrap vectors
    for d in runs:
        fold, seed = d.name.split("_seed"); seed = int(seed); fi = int(fold[1:])
        idx, K, N, L, _ = load_run(d, fold, GT, np.random.default_rng([2026, fi, seed]))
        tgt, pos = (idx.role == "target").to_numpy(), K > 0
        ucl, ci = np.unique(idx.cluster_id.to_numpy(), return_inverse=True); C = len(ucl)
        m = np.bincount(ci, weights=pos, minlength=C)
        A = np.zeros((C, len(idx))); A[ci[pos], np.flatnonzero(pos)] = 1.0 / m[ci[pos]]
        c_tgt = np.bincount(ci, weights=tgt, minlength=C) > 0
        tgtP = np.flatnonzero(c_tgt & (m > 0)); nT, mT = len(tgtP), m[tgtP]
        t = 0.1
        Lc = A @ L[t]; LT = Lc[tgtP]
        pick = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
        j_pool = int(np.searchsorted(G, pick(cs, fold=fold, seed=str(seed), iou=t, alpha=A0, strategy="pooled", budget=0).lam_median))
        j_pcl = int(np.searchsorted(G, pick(cc, fold=fold, seed=str(seed), iou=t, alpha=A0, strategy="pooled_cl", budget=0).lam_median))
        pf = lambda w, j: (w * mT) @ LT[:, j] / (w * mT).sum()
        pcv = lambda w, j: w @ LT[:, j] / w.sum()
        one = np.ones(nT)
        assert abs(pf(one, j_pool) - pick(cs, fold=fold, seed=str(seed), iou=t, alpha=A0, strategy="pooled", budget=0).risk_mean) < 1e-12

        # ---- A: theorem excess bound for recalibration on k target videos
        g = np.random.default_rng([2026, fi, seed, 77])
        flat = np.concatenate([L[t][(ci == c) & pos, 0] for c in tgtP])  # frame floors, grouped by target cluster
        off = np.r_[0, np.cumsum(mT)[:-1]].astype(int)
        for k in (5, 10, 20, 40):
            ke = min(k, nT - 10)
            sel = np.argsort(g.random((R_MC, nT)), 1)[:, :ke + 1]  # k+1 clusters without replacement
            ra = LT[sel, 0].mean(1)
            rb = flat[off[sel] + np.floor(g.random(sel.shape) * mT[sel]).astype(int)].mean(1)
            for alpha in (0.05, 0.10, 0.20):
                put("A_bound_a", fold, f"seed{seed}_k{k}_a{alpha}", np.maximum(ra - alpha, 0).mean())
                put("A_bound_b", fold, f"seed{seed}_k{k}_a{alpha}", np.maximum(rb - alpha, 0).mean())
        put("A_floor_video", fold, f"seed{seed}", LT[:, 0].mean())

        # ---- B: dominance of single target videos
        if seed == 0:
            put("B_kish_eff_videos", fold, "frame_avg", mT.sum() ** 2 / (mT ** 2).sum())
            put("B_n_pos_videos", fold, "", nT)
            put("B_largest_share", fold, ucl[tgtP[np.argmax(mT)]], mT.max() / mT.sum())
        drop = one.copy(); drop[np.argmax(mT)] = 0
        U = np.random.default_rng([2026, fi, 5150]).integers(0, nT, (B, nT))  # R1 resamples
        Cnt = np.zeros((B, nT)); np.add.at(Cnt, (np.repeat(np.arange(B), nT), U.ravel()), 1)
        U2 = np.random.default_rng([2026, fi, 7171]).integers(0, nT - 1, (B, nT - 1))
        keep = np.flatnonzero(drop)
        Cnt2 = np.zeros((B, nT)); np.add.at(Cnt2, (np.repeat(np.arange(B), nT - 1), keep[U2.ravel()]), 1)
        PFj = lambda W, j: ((W * mT) @ LT[:, j]) / (W * mT).sum(1)
        PCj = lambda W, j: (W @ LT[:, j]) / W.sum(1)
        # ---- C: nested source-calibration bootstrap thresholds
        jb = np.searchsorted(G, lo[(lo.fold == fold) & (lo.seed == seed)].sort_values("rep").lam.to_numpy())
        jbc = np.searchsorted(G, lc[(lc.fold == fold) & (lc.seed == seed)].sort_values("rep").lam.to_numpy())
        assert len(jb) == 200 and len(jbc) == 200
        PF_all = ((Cnt * mT) @ LT) / (Cnt * mT).sum(1)[:, None]
        PC_all = (Cnt @ LT) / Cnt.sum(1)[:, None]
        acc.setdefault(fold, {})[seed] = dict(
            drop_fr=(pf(drop, j_pool), PFj(Cnt2, j_pool)), drop_cl=(pcv(drop, j_pcl), PCj(Cnt2, j_pcl)),
            nest_fr=(np.mean([pf(one, j) for j in jb]), PF_all[:, jb].mean(1)),
            nest_cl=(np.mean([pcv(one, j) for j in jbc]), PC_all[:, jbc].mean(1)),
            fixed_fr=(pf(one, j_pool), PF_all[:, j_pool]), fixed_cl=(pcv(one, j_pcl), PC_all[:, j_pcl]))

        # ---- D: AP at IoU 0.1 / 0.5;  E: hit geometry at the pooled threshold
        preds = pd.read_parquet(next(d.glob("preds_*.parquet")))
        gp = {k: (v[["x1", "y1", "x2", "y2"]].to_numpy(float), v.score.to_numpy(float)) for k, v in preds.groupby("orig_path")}
        empty = (np.zeros((0, 4)), np.zeros(0))
        tg = np.flatnonzero(tgt)
        for tt in (0.1, 0.5):
            ms_pos, sc_pos, sc_all, ngt = [], [], [], 0
            for i in tg:
                b, s = gp.get(idx.orig_path[i], empty)
                sc_all.append(s)
                if K[i]:
                    ms_pos.append(matched_scores(GT[idx.orig_path[i]], b, s, tt)); sc_pos.append(s); ngt += K[i]
            msp, scp, sca = np.concatenate(ms_pos), np.concatenate(sc_pos), np.concatenate(sc_all)
            ap_pos, ap_all = ap_from_matched(msp, scp, ngt), ap_from_matched(msp, sca, ngt)
            put("D_AP_pos", fold, f"seed{seed}_iou{tt}", ap_pos); put("D_AP_all", fold, f"seed{seed}_iou{tt}", ap_all)
            if tt == 0.5:
                ref = r5[(r5.fold == fold) & (r5.seed == seed) & (r5.role == "target")].iloc[0]
                put("D_AP_check_pos", fold, f"seed{seed}", ap_pos - ref.AP50_pos)
                put("D_AP_check_all", fold, f"seed{seed}", ap_all - ref.AP50_all)
        lam = G[j_pool]; ious, ratio, cin, nmatch = [], [], [], 0
        for i in tg[K[tg] > 0]:
            gt = GT[idx.orig_path[i]]; b, s = gp.get(idx.orig_path[i], empty)
            prs = greedy_pairs(gt, b, s, lam, 0.1)
            nmatch += len(prs)
            for bi, gi, v in prs:
                bb, gg = b[bi], gt[gi]
                ious.append(v)
                ratio.append(np.prod(bb[2:] - bb[:2]) / np.prod(gg[2:] - gg[:2]))
                cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                cin.append(gg[0] <= cx < gg[2] and gg[1] <= cy < gg[3])
        expect = ((1 - L[0.1][tg[K[tg] > 0], j_pool]) * K[tg[K[tg] > 0]]).sum()
        assert abs(nmatch - expect) < 1e-6, (d.name, nmatch, expect)
        ious, ratio = np.array(ious), np.array(ratio)
        put("E_hits_iou_lt_0.5", fold, f"seed{seed}", (ious < 0.5).mean())
        put("E_hits_area_gt2", fold, f"seed{seed}", (ratio > 2).mean())
        put("E_hits_area_gt4", fold, f"seed{seed}", (ratio > 4).mean())
        put("E_hits_centre_in_gt", fold, f"seed{seed}", np.mean(cin))
        put("E_n_hits", fold, f"seed{seed}", len(ious))

        # ---- G: polyp-free target frames inside positive target videos
        if seed == 0:
            neg = tgt & ~pos
            in_pos = np.isin(ci, tgtP)
            put("G_neg_in_pos_videos", fold, f"of_{int(neg.sum())}", (neg & in_pos).sum() / max(neg.sum(), 1))
        print(f"  {d.name}: done", flush=True)

    # hierarchical seed mixing exactly as R1 (seed picked per resample)
    for fold, per in acc.items():
        fi = int(fold[1:]); seeds = sorted(per)
        pick_s = np.random.default_rng([2026, fi, 5152]).integers(0, len(seeds), B)
        for key in per[seeds[0]]:
            mix = np.array([per[seeds[i]][key][1][b] for b, i in enumerate(pick_s)])
            pt = np.mean([per[s][key][0] for s in seeds])
            put("BC_" + key, fold, "point", pt); put("BC_" + key, fold, "lo", np.nanquantile(mix, .025))
            put("BC_" + key, fold, "hi", np.nanquantile(mix, .975)); put("BC_" + key, fold, "frac_above", np.nanmean(mix > A0))
    # R1 reproduction for the fixed-threshold keys
    r1 = pd.read_csv(EV / "revision_R1_target_bootstrap.csv", dtype={"seed": str})
    df = pd.DataFrame(rows)
    for key, strat, met in (("fixed_fr", "pooled", "risk_fr"), ("fixed_cl", "pooled_cl", "risk_cl")):
        for fold in acc:
            ref = r1[(r1.fold == fold) & (r1.seed == "all") & (r1.iou == 0.1) & (r1.alpha == A0) & (r1.strategy == strat) & (r1.metric == met)].iloc[0]
            got = df[(df.check == "BC_" + key) & (df.fold == fold)].set_index("key").value
            assert abs(got["lo"] - ref.lo) < 1e-12 and abs(got["hi"] - ref.hi) < 1e-12, (key, fold)
    print("R1 fixed-threshold intervals reproduced exactly (pooled risk_fr, pooled_cl risk_cl)")

    # ---- F: ClinicDB pooled-source bootstrap
    from clinicdb_crc import clinicdb_gt, losses  # noqa: E402
    meta, CGT, _ = clinicdb_gt()
    Kc = np.array([len(b) for b in CGT]); posc = Kc > 0
    uv, vi = np.unique(meta.sequence_id.to_numpy(), return_inverse=True); V = len(uv)
    mv = np.bincount(vi, weights=posc, minlength=V)
    Ac = np.zeros((V, len(meta))); Ac[vi[posc], np.flatnonzero(posc)] = 1.0 / mv[vi[posc]]
    pooled = pd.read_parquet(EV / "conformal_long.parquet", columns=["model", "fold", "seed", "iou", "alpha", "strategy", "lam"])
    pooled = pooled[(pooled.strategy == "pooled") & (pooled.model == "kaggle") & (pooled.iou == 0.1) & (pooled.alpha == A0)]
    cdb = pd.read_csv(EV / "clinicdb_summary.csv", dtype={"run": str})
    Uv = np.random.default_rng([2026, 1101, 5150]).integers(0, V, (B, V))
    Cv = np.zeros((B, V)); np.add.at(Cv, (np.repeat(np.arange(B), V), Uv.ravel()), 1)
    per = {}
    for pq in sorted((ROOT / "outputs" / "clinicdb").glob("preds_C*_seed*.parquet")):
        run = pq.stem.replace("preds_", ""); fold, seed = run.split("_seed")
        Lcl, _, _ = losses(CGT, pd.read_parquet(pq))
        j = int(np.searchsorted(G, pooled[(pooled.fold == fold) & (pooled.seed == int(seed))].lam.iloc[0]))
        lv = Ac @ Lcl[0.1][:, j]  # video loss
        fr, cl = (Cv * mv) @ lv / (Cv * mv).sum(1), Cv @ lv / V
        p_fr, p_cl = (mv @ lv) / mv.sum(), lv.mean()
        ref = cdb[(cdb.strategy == "pooled_source") & (cdb.run == run) & (cdb.iou == 0.1) & (cdb.alpha == A0)].iloc[0]
        assert abs(p_fr - ref.risk_fr_mean) < 1e-9 and abs(p_cl - ref.risk_cl_mean) < 1e-9, run
        per[run] = (p_fr, p_cl, fr, cl)
        put("F_detector_frac_above_fr", run, "", np.mean(fr > A0)); put("F_detector_lo_fr", run, "", np.quantile(fr, .025))
        put("F_detector_frac_above_cl", run, "", np.mean(cl > A0)); put("F_detector_lo_cl", run, "", np.quantile(cl, .025))
    names = sorted(per); ps = np.random.default_rng([2026, 1101, 5152]).integers(0, len(names), B)
    for k_, lab in ((2, "fr"), (3, "cl")):
        mix = np.array([per[names[i]][k_][b] for b, i in enumerate(ps)])
        put(f"F_pooled_{lab}", "ClinicDB", "point", np.mean([per[n][k_ - 2] for n in names]))
        put(f"F_pooled_{lab}", "ClinicDB", "lo", np.quantile(mix, .025)); put(f"F_pooled_{lab}", "ClinicDB", "hi", np.quantile(mix, .975))
        put(f"F_pooled_{lab}", "ClinicDB", "frac_above", np.mean(mix > A0))
    print("ClinicDB pooled-source point values reproduced (clinicdb_summary.csv, 18 detectors)")

    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    # summaries
    def seedmean(check, filt):
        x = df[(df.check == check) & df.key.str.contains(filt, regex=False)]
        return x.groupby("fold").value.agg(["mean", "max"]).round(4)
    for k in (5, 10, 20, 40):
        print(f"A bound, k={k}, alpha 0.10: design a\n", seedmean("A_bound_a", f"_k{k}_a0.1").T.to_string(),
              "\n design b\n", seedmean("A_bound_b", f"_k{k}_a0.1").T.to_string())
    print("A bound k=20 alpha 0.05 b\n", seedmean("A_bound_b", "_k20_a0.05").T.to_string())
    print("A floor video\n", df[df.check == "A_floor_video"].groupby("fold").value.mean().round(4).to_dict())
    print(df[df.check.str.startswith(("B_", "G_"))].to_string(index=False))
    bc = df[df.check.str.startswith("BC_")].pivot_table(index=["check", "fold"], columns="key", values="value").round(3)
    print(bc.to_string())
    for c in ("D_AP_pos", "D_AP_all", "D_AP_check_pos", "D_AP_check_all", "E_hits_iou_lt_0.5", "E_hits_area_gt2",
              "E_hits_area_gt4", "E_hits_centre_in_gt"):
        x = df[df.check == c]; x = x.assign(iou=x.key.str.extract(r"iou([\d.]+)")[0])
        print(c, x.groupby(["fold", "iou"], dropna=False).value.agg(["mean", "min", "max"]).round(4).to_string())
    print(df[df.check.str.startswith("F_pooled")].to_string(index=False))
    fdet = df[df.check.str.startswith("F_detector")].pivot_table(index="fold", columns="check", values="value").round(3)
    print(fdet.to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
