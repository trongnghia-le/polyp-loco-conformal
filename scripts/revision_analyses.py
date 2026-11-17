"""Referee round-1 re-analyses of the EXISTING YOLO11s LOCO results (no training, CPU only).
Note: docs/notes/2026-11-09_revision-analyses.md. Spec of loss / CRC: docs/metric.md.

R1 (M4/M5) target-cluster bootstrap (B resamples of the held-out centre's positive clusters, with replacement) of the
   realised held-out risk for: pooled (image CRC), worst_case (per-source min), pooled_cl (cluster CRC), and target
   recalibration with k = 20 videos, fs_a (cluster CRC) / fs_b (image CRC on one random positive frame per video).
   fs_a/fs_b re-run the whole recalibration inside each resample (D_BOOT draws of k units; test = units whose video is
   not in the calibration draw). Per fold x seed x IoU x alpha; seed 'all' = hierarchical (seed drawn per resample).
R2 (M5) in-domain cluster-split risk distribution (mean, q95, max over splits) vs the single pooled deployment.
R3 (M1/M2) pooled image CRC with at most m positive frames per cal cluster (m in CAPS, R_CAP random subsamples) and
   leaving out source centres; realised target risk.
R4 (M6/M9) target recalibration: degenerate fraction, risk over non-degenerate vs all draws, false alarms per
   polyp-free frame (median, IQR) and fraction of polyp-free frames with any alarm.
R5 (M7) AP50 on polyp-containing frames vs all frames; image-level vs polyp-level (and video-level) miss floor.

Reuses conformal.py / conformal_cluster.py / metrics.py by import, and eval_detection.py's match/evaluate (its module
code has no main guard, so only its function definitions are executed). Sanity asserts reproduce pooled, worst_case,
pooled_cl, fs_a/fs_b (k = 20), few_shot, AP50 and max_recall from the existing result files.
Usage: .venv/Scripts/python.exe -I scripts/revision_analyses.py   -> outputs/eval/revision_R{1..5}_*.csv (~20 min)"""
import ast
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conformal import clean, G, ALPHAS, IOUS, NEG_CENTRES, load_run, crc_idx, evaluate  # noqa: E402
from conformal_cluster import MIN_TEST  # noqa: E402
from metrics import iou  # noqa: E402

EV = ROOT / "outputs" / "eval"
B, D_BOOT, CHUNK, K_REC, DRAWS = 1000, 100, 100, 20, 500  # D_BOOT recalibration draws per bootstrap resample
CAPS, R_CAP = (1, 5, 20), 200

_src = ast.parse((ROOT / "scripts" / "eval_detection.py").read_text(encoding="utf-8"))
DET = dict(np=np, iou=iou)
exec(compile(ast.Module([n for n in _src.body if isinstance(n, ast.FunctionDef)], []), "eval_detection.py", "exec"), DET)


def pick(df, **kw):
    s = df
    for k, v in kw.items():
        s = s[s[k] == v]
    assert len(s) == 1, (kw, len(s))
    return s.iloc[0]


def interval(point, boot, alpha):
    return dict(point=point, lo=np.nanquantile(boot, .025), hi=np.nanquantile(boot, .975),
                frac_above=float(np.nanmean(boot > alpha)), boot_mean=float(np.nanmean(boot)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    GT = {clean(p): np.asarray(json.loads(gb), float).reshape(-1, 4) for p, gb in zip(man.path, man.gt_boxes)}
    DET["GT"] = GT
    cs = pd.read_csv(EV / "conformal_summary.csv", dtype={"seed": str})
    cc = pd.read_csv(EV / "conformal_cluster_summary.csv", dtype={"seed": str})
    dsum, mrec = pd.read_csv(EV / "detection_summary.csv"), pd.read_csv(EV / "max_recall.csv")
    runs = sorted(d for d in (ROOT / "outputs" / "kaggle").glob("C*_seed*") if next(d.glob("preds_*.parquet"), None))
    assert len(runs) == 18, len(runs)
    r1, boots, r3, r5, alarm, n_assert, empty_test = [], {}, [], [], {}, 0, []

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
        cpos = m > 0
        calP, tgtP = np.flatnonzero(c_cal & cpos), np.flatnonzero(c_tgt & cpos)
        nT, mT = len(tgtP), m[tgtP]
        tneg = tgt & ~pos
        centres = sorted(set(centre[cal]))
        T = tgt[None].astype(float)
        w0 = (cal & pos).astype(float)
        if fold in NEG_CENTRES:
            alarm[(fold, seed)] = ((N[tneg] > 0).mean(0), N[tneg].mean(0))
        cfg = dict(fold=fold, seed=str(seed))

        # ---------- R3 weights (seed independent: same subsamples for every seed)
        cp = np.flatnonzero(cal & pos); cpc = ci[cp]
        variants = [("all", 0, "", w0[None])]
        for cap in CAPS:
            g = np.random.default_rng([2026, fi, 33, cap]); W = np.zeros((R_CAP, len(idx)))
            for r in range(R_CAP):
                o = np.lexsort((g.random(len(cp)), cpc)); sc = cpc[o]
                W[r, cp[o[np.arange(len(o)) - np.searchsorted(sc, sc) < cap]]] = 1
            assert (W.sum(1) == np.minimum(np.bincount(cpc), cap).sum()).all()
            variants.append((f"cap{cap}", cap, "", W))
        excl = [(c,) for c in centres] + ([tuple(c for c in ("C2", "C4") if c in centres)] if fold not in ("C2", "C4") else [])
        for ex in excl:
            variants.append(("excl_" + "+".join(ex), 0, "+".join(ex), (w0 * ~np.isin(centre, ex))[None]))

        # ---------- recalibration draws on the original sample, exactly as conformal_cluster.py (k = 20)
        ke = min(K_REC, nT - MIN_TEST)
        g1, g2 = np.random.default_rng([2026, fi, K_REC, 11]), np.random.default_rng([2026, fi, K_REC, 12])
        sel = np.array([g1.choice(tgtP, ke, replace=False) for _ in range(DRAWS)])
        pf = {c: np.flatnonzero(pos & (ci == c)) for c in tgtP}
        Vd = np.zeros((DRAWS, C)); np.put_along_axis(Vd, sel, 1.0, 1)
        Wone = np.zeros((DRAWS, len(idx)))
        for r in range(DRAWS):
            Wone[r, [g2.choice(pf[c]) for c in sel[r]]] = 1
        Vtest = np.where(c_tgt[None], 1 - Vd, 0); Vt = Vtest * cpos; Wt = Vtest[:, ci]

        # ---------- bootstrap of target positive clusters (seed independent -> common resamples across seeds)
        U = np.random.default_rng([2026, fi, 5150]).integers(0, nT, (B, nT))
        Cnt = np.zeros((B, nT)); np.add.at(Cnt, (np.repeat(np.arange(B), nT), U.ravel()), 1)
        fr_order = np.concatenate([pf[c] for c in tgtP]); offs = np.r_[0, np.cumsum(mT)[:-1]].astype(int)

        for t in IOUS:
            LT = Lc[t][tgtP]
            PC, PF = Cnt @ LT / nT, (Cnt * mT) @ LT / (Cnt * mT).sum(1)[:, None]  # (B, |G|) resampled risks
            pc0, pf0 = LT.mean(0), (mT[:, None] * LT).sum(0) / mT.sum()
            FA = N[tneg].mean(0) if fold in NEG_CENTRES else np.full(G.size, np.nan)
            for alpha in ALPHAS:
                sv = lambda st, b=0: pick(cs, **cfg, iou=t, alpha=alpha, strategy=st, budget=b)
                cv = lambda st, b=0: pick(cc, **cfg, iou=t, alpha=alpha, strategy=st, budget=b)
                fixed = {}
                # pooled (exact conformal.py path) -> sanity before anything is modified
                j, dg = crc_idx(w0[None] @ L[t], w0[None].sum(1), alpha)
                ev = evaluate(T, j, K, L[t], N)
                assert abs(ev["risk"][0] - sv("pooled").risk_mean) < 1e-12 and G[j[0]] == sv("pooled").lam_median
                assert abs(pf0[j[0]] - ev["risk"][0]) < 1e-12; n_assert += 2
                fixed["pooled"] = j[0]
                js = [crc_idx((w0[None] * (centre == c)) @ L[t], (w0[None] * (centre == c)).sum(1), alpha)[0][0]
                      for c in centres]
                fixed["worst_case"] = min(js)
                assert abs(pf0[min(js)] - sv("worst_case").risk_mean) < 1e-12; n_assert += 1
                V0 = np.zeros((1, C)); V0[0, calP] = 1
                j, dg = crc_idx(V0 @ Lc[t], V0.sum(1), alpha)
                assert abs(pc0[j[0]] - cv("pooled_cl").risk_cl_mean) < 1e-12; n_assert += 1
                fixed["pooled_cl"] = j[0]
                for st, j in fixed.items():
                    for met, p0, PB in (("risk_fr", pf0, PF), ("risk_cl", pc0, PC)):
                        boots[(fold, t, alpha, st, met, seed)] = (p0[j], PB[:, j])
                # R3
                for name, cap, ex, W in variants:
                    j, dg = crc_idx(W @ L[t], W.sum(1), alpha)
                    if name == "all":
                        assert abs(pf0[j[0]] - sv("pooled").risk_mean) < 1e-12; n_assert += 1
                    kc = [len(set(ci[w > 0])) for w in W]
                    r3.append(pd.DataFrame(dict(**cfg, iou=t, alpha=alpha, variant=name, cap=cap, excluded=ex,
                                                rep=np.arange(len(j)), lam=G[j], degenerate=dg, n_cal=W.sum(1),
                                                k_cal=kc, risk_fr=pf0[j], risk_cl=pc0[j], fa_neg=FA[j])))
                # recalibration, original sample: exact reproduction of conformal_cluster.py fs_a / fs_b (k = 20)
                for st, (j, dg) in (("fs_a", crc_idx(Vd @ Lc[t], Vd.sum(1), alpha)),
                                    ("fs_b", crc_idx((Wone * pos) @ L[t], (Wone * pos).sum(1), alpha))):
                    rcl = (Vt * Lc[t][:, j].T).sum(1) / Vt.sum(1)
                    rfr = evaluate(Wt, j, K, L[t], N)["risk"]
                    ref = cv(st, K_REC)
                    assert abs(rcl.mean() - ref.risk_cl_mean) < 1e-12 and abs(rfr.mean() - ref.risk_fr_mean) < 1e-12
                    n_assert += 1
                    boots[(fold, t, alpha, st, "risk_cl", seed)] = [rcl.mean(), np.zeros(B)]
                    boots[(fold, t, alpha, st, "risk_fr", seed)] = [rfr.mean(), np.zeros(B)]
            # recalibration inside each bootstrap resample (chunks of CHUNK resamples x D_BOOT draws)
            LposT = L[t][fr_order]
            for ch in range(B // CHUNK):
                rc = np.random.default_rng([2026, fi, 5151, ch])  # seed independent
                bs = np.arange(ch * CHUNK, (ch + 1) * CHUNK).repeat(D_BOOT); R = len(bs)
                slots = rc.random((R, nT)).argsort(1)[:, :ke]  # ke distinct units of the resample
                calc = U[bs[:, None], slots]  # (R, ke) positions in tgtP
                u = rc.random(calc.shape)
                incl = np.zeros((R, nT)); np.add.at(incl, (np.arange(R)[:, None].repeat(ke, 1), calc), 1)
                testw = Cnt[bs] * (incl == 0)  # test: resampled units whose video is not in the calibration draw
                empty_test.append((fold, (testw.sum(1) == 0).mean()))
                frames = offs[calc] + (u * mT[calc]).astype(int)
                Sb = np.zeros((R, G.size))
                for s in range(ke):
                    Sb += LposT[frames[:, s]]
                for st, S in (("fs_a", incl @ LT), ("fs_b", Sb)):
                    for alpha in ALPHAS:
                        j, _ = crc_idx(S, np.full(R, float(ke)), alpha)
                        Lj = LT[:, j].T
                        with np.errstate(invalid="ignore", divide="ignore"):
                            rcl = (testw * Lj).sum(1) / testw.sum(1)
                            rfr = (testw * mT * Lj).sum(1) / (testw * mT).sum(1)
                        for met, v in (("risk_cl", rcl), ("risk_fr", rfr)):
                            boots[(fold, t, alpha, st, met, seed)][1][ch * CHUNK:(ch + 1) * CHUNK] = \
                                np.nanmean(v.reshape(CHUNK, D_BOOT), 1)

        # ---------- R5
        idx_all = pd.read_csv(next(d.glob("index_*.csv"))); preds = pd.read_parquet(next(d.glob("preds_*.parquet")))
        for role in ("val", "target"):
            sub, P = idx_all[idx_all.role == role], preds[preds.role == role]
            a_all = DET["evaluate"](sub.orig_path.tolist(), P)
            a_pos = DET["evaluate"](sub.orig_path[sub.n_gt > 0].tolist(), P)
            assert abs(a_all["AP50"] - pick(dsum, fold=fold, seed=seed, role=role).AP50) < 1e-12; n_assert += 1
            row = dict(fold=fold, seed=seed, role=role, AP50_all=a_all["AP50"], AP50_pos=a_pos["AP50"],
                       n_img=a_all["n_img"], n_pos_img=a_pos["n_img"], neg_share=1 - a_pos["n_img"] / a_all["n_img"])
            if role == "target":
                tp = tgt & pos
                for t in IOUS:
                    poly = (L[t][tp, 0] * K[tp]).sum() / K[tp].sum()
                    mr = 1 - pick(mrec, fold=fold, seed=seed, role="target", iou=t).max_recall
                    assert abs(poly - mr) < 1e-9, (d.name, t, poly, mr); n_assert += 1
                    row.update({f"floor_img_iou{t}": L[t][tp, 0].mean(), f"floor_polyp_iou{t}": mr,
                                f"floor_video_iou{t}": Lc[t][tgtP, 0].mean()})
            r5.append(row)
        print(f"  {d.name}: target pos clusters {nT} (k = {ke}), cal pos frames {len(cp)} in {len(calP)} clusters", flush=True)
    print(f"SANITY: {n_assert} reproductions of existing numbers (pooled, worst_case, pooled_cl, fs_a/fs_b k=20, "
          f"R3 m=all == pooled, AP50, 1-max_recall) - OK")

    # ---------- R1 table (per seed + hierarchical 'all')
    keys = sorted({k[:5] for k in boots})
    for k in keys:
        fold, t, alpha = k[:3]
        seeds = sorted(s for (*kk, s) in boots if tuple(kk) == k)
        for s in seeds:
            p0, bv = boots[(*k, s)]
            r1.append(dict(fold=fold, seed=str(s), iou=t, alpha=alpha, strategy=k[3], metric=k[4], **interval(p0, bv, alpha)))
        pick_s = np.random.default_rng([2026, int(fold[1:]), 5152]).integers(0, len(seeds), B)
        mix = np.array([boots[(*k, seeds[i])][1][b] for b, i in enumerate(pick_s)])
        r1.append(dict(fold=fold, seed="all", iou=t, alpha=alpha, strategy=k[3], metric=k[4],
                       **interval(np.mean([boots[(*k, s)][0] for s in seeds]), mix, alpha)))
    R1 = pd.DataFrame(r1)
    bad = R1[(R1.point < R1.lo - 1e-12) | (R1.point > R1.hi + 1e-12)]
    print(f"R1: {len(R1)} intervals; point outside its 95% interval in {len(bad)} rows")
    if len(bad):
        print(bad.to_string(index=False))
    assert len(bad) == 0, "interval does not contain the point estimate"
    R1.to_csv(EV / "revision_R1_target_bootstrap.csv", index=False)
    et = pd.DataFrame(empty_test, columns=["fold", "f"]).groupby("fold").f.mean()
    print("R1 recal bootstrap: fraction of draws with an empty test set (dropped):", et.round(4).to_dict())

    # ---------- R2: in-domain split distribution vs single pooled deployment
    lo = pd.read_parquet(EV / "conformal_long.parquet", columns=["fold", "seed", "iou", "alpha", "strategy", "risk", "degenerate"])
    lc = pd.read_parquet(EV / "conformal_cluster_long.parquet", columns=["fold", "seed", "iou", "alpha", "strategy", "risk_cl",
                                                                         "risk_fr", "degenerate", "budget", "lam", "fa_neg"])
    ind = pd.concat([lo[lo.strategy == "in_domain"].assign(cmp="pooled", metric="risk_fr"),
                     lc[lc.strategy == "in_domain_cl"].rename(columns={"risk_cl": "risk"}).assign(cmp="pooled_cl", metric="risk_cl")])
    q95 = lambda x: x.quantile(.95)
    r2 = ind.groupby(["fold", "iou", "alpha", "strategy", "cmp", "metric"]).agg(
        ind_mean=("risk", "mean"), ind_q95=("risk", q95), ind_max=("risk", "max"),
        ind_degenerate=("degenerate", "mean"), n_splits=("risk", "size")).reset_index()
    ps = ind.groupby(["fold", "seed", "iou", "alpha", "cmp"]).risk.quantile(.95).rename("q95_seed").reset_index()
    dep = []
    for (f, s, t, a, cmpn), q in ps.set_index(["fold", "seed", "iou", "alpha", "cmp"]).q95_seed.items():
        met = "risk_fr" if cmpn == "pooled" else "risk_cl"
        dep.append(dict(fold=f, seed=s, iou=t, alpha=a, cmp=cmpn, q95_seed=q,
                        dep=pick(R1, fold=f, seed=str(s), iou=t, alpha=a, strategy=cmpn, metric=met).point,
                        dep_lo=pick(R1, fold=f, seed=str(s), iou=t, alpha=a, strategy=cmpn, metric=met).lo))
    dep = pd.DataFrame(dep)
    r2 = r2.merge(dep.groupby(["fold", "iou", "alpha", "cmp"]).agg(
        dep_mean=("dep", "mean"), dep_min=("dep", "min"), dep_max=("dep", "max"),
        seeds_dep_gt_q95=("dep", lambda x: int((x > dep.loc[x.index, "q95_seed"]).sum())),
        seeds_deplo_gt_q95=("dep_lo", lambda x: int((x > dep.loc[x.index, "q95_seed"]).sum()))).reset_index(),
        on=["fold", "iou", "alpha", "cmp"])
    r2.to_csv(EV / "revision_R2_indomain_spread.csv", index=False)

    # ---------- R3 summary
    R3l = pd.concat(r3, ignore_index=True)
    agg3 = dict(risk_fr_mean=("risk_fr", "mean"), risk_fr_q05=("risk_fr", lambda x: x.quantile(.05)),
                risk_fr_q95=("risk_fr", q95), viol_fr=("viol_fr", "mean"), risk_cl_mean=("risk_cl", "mean"),
                viol_cl=("viol_cl", "mean"), degenerate_frac=("degenerate", "mean"), fa_neg_mean=("fa_neg", "mean"),
                n_cal_median=("n_cal", "median"), k_cal_median=("k_cal", "median"), n_rows=("risk_fr", "size"))
    R3l["viol_fr"], R3l["viol_cl"] = R3l.risk_fr > R3l.alpha, R3l.risk_cl > R3l.alpha
    k3 = ["fold", "iou", "alpha", "variant", "cap", "excluded"]
    R3 = pd.concat([R3l.groupby(k3).agg(**agg3).reset_index().assign(seed="all"),
                    R3l.groupby(k3 + ["seed"]).agg(**agg3).reset_index()], ignore_index=True)
    R3.to_csv(EV / "revision_R3_calibration_composition.csv", index=False)

    # ---------- R4: degenerate draws and false alarms (existing per-draw results; any-alarm from N at each draw's lam)
    rec = lc[lc.strategy.isin(["fs_a", "fs_b"]) & lc.budget.isin([10, 20]) | (lc.strategy == "fs_frames")].copy()
    rec["seed_i"] = rec.seed.astype(int)
    fsr = rec[rec.strategy == "fs_frames"].groupby(["fold", "seed", "iou", "alpha", "budget"]).risk_fr.mean()
    for (f, s, t, a, b), v in fsr.items():  # fs_frames == conformal.py few_shot (same stored draws)
        assert abs(v - pick(cs, fold=f, seed=str(s), iou=t, alpha=a, strategy="few_shot", budget=b).risk_mean) < 1e-9
    rec["any_alarm"] = np.nan
    jj = np.rint(rec.lam.to_numpy() * 1000).astype(int) - 1
    for (f, s), (anyv, _) in alarm.items():
        mk = ((rec.fold == f) & (rec.seed_i == s)).to_numpy()
        rec.loc[mk, "any_alarm"] = anyv[jj[mk]]
    rec["ctrl"] = np.where(rec.strategy == "fs_frames", rec.risk_fr, rec.risk_cl)  # quantity the design certifies
    rows4 = []
    for kk, g in rec.groupby(["fold", "iou", "alpha", "strategy", "budget"]):
        a, nd = kk[2], ~g.degenerate.to_numpy()
        r = dict(zip(["fold", "iou", "alpha", "strategy", "budget"], kk), n_draws=len(g),
                 certified_quantity="frame-avg" if kk[3] == "fs_frames" else "video-avg",
                 degenerate_frac=1 - nd.mean())
        for nm, mk in (("all", np.ones_like(nd)), ("nondeg", nd)):
            h = g[mk]
            r.update({f"risk_{nm}": h.ctrl.mean(), f"risk_fr_{nm}": h.risk_fr.mean(), f"risk_cl_{nm}": h.risk_cl.mean(),
                      f"viol_{nm}": (h.ctrl > a).mean(), f"fa_mean_{nm}": h.fa_neg.mean(),
                      f"fa_median_{nm}": h.fa_neg.median(), f"fa_q25_{nm}": h.fa_neg.quantile(.25),
                      f"fa_q75_{nm}": h.fa_neg.quantile(.75), f"any_alarm_median_{nm}": h.any_alarm.median()})
        r.update(outcome_held=(nd & (g.ctrl <= a).to_numpy()).mean(), outcome_violated=(nd & (g.ctrl > a).to_numpy()).mean(),
                 outcome_cannot_certify=1 - nd.mean())
        rows4.append(r)
    R4 = pd.DataFrame(rows4)
    R4.to_csv(EV / "revision_R4_degenerate_fa.csv", index=False)

    # ---------- R5
    R5 = pd.DataFrame(r5); R5.to_csv(EV / "revision_R5_detection_floor.csv", index=False)

    # ---------- compact printouts (IoU 0.1 unless stated)
    f3 = lambda x: f"{x:.3f}"
    for t in (0.1, 0.5):
        for alpha in (0.10, 0.20):
            s = R1[(R1.seed == "all") & (R1.iou == t) & (R1.alpha == alpha)]
            print(f"\nR1 IoU {t}, alpha {alpha:.2f}: point [2.5, 97.5%] P(boot > alpha); seed 'all' = seed drawn per resample")
            tab = {(st, met): {r.fold: f"{r.point:.3f} [{r.lo:.3f},{r.hi:.3f}] {r.frac_above:.2f}" for r in g.itertuples()}
                   for (st, met), g in s.groupby(["strategy", "metric"])}
            print(pd.DataFrame(tab).T.to_string())
    for t in (0.1, 0.5):
        s = r2[(r2.iou == t) & (r2.alpha == 0.10)]
        print(f"\nR2 IoU {t}, alpha 0.10: in-domain splits (600 = 200 x 3 seeds) mean/q95/max vs single deployment "
              f"(mean, min-max over seeds); seeds with deployment > same-seed q95; ... with bootstrap 2.5% bound > q95")
        print(s[["fold", "cmp", "ind_mean", "ind_q95", "ind_max", "ind_degenerate", "dep_mean", "dep_min", "dep_max",
                 "seeds_dep_gt_q95", "seeds_deplo_gt_q95"]].round(3).to_string(index=False))
    for t in (0.1, 0.5):
        for alpha in (0.10, 0.20):
            s = R3[(R3.seed == "all") & (R3.iou == t) & (R3.alpha == alpha)]
            print(f"\nR3 IoU {t}, alpha {alpha:.2f}: pooled image CRC, frame-avg target risk mean (frac > alpha) "
                  f"[video-avg risk]; cap rows over {R_CAP} subsamples x 3 seeds")
            print(s.assign(c=[f"{a:.3f} ({b:.2f}) [{c:.3f}]" for a, b, c in zip(s.risk_fr_mean, s.viol_fr, s.risk_cl_mean)])
                  .pivot(index="variant", columns="fold", values="c").fillna("-").to_string())
    s = R4[(R4.iou == 0.1) & (R4.alpha == 0.10)]
    print("\nR4 IoU 0.1, alpha 0.10: degenerate frac | certified risk all/non-deg (viol non-deg) | FA/neg frame median "
          "[IQR] non-deg | any-alarm median non-deg")
    print(s.assign(c=[f"{r.degenerate_frac:.2f} | {r.risk_all:.3f}/{r.risk_nondeg:.3f} ({r.viol_nondeg:.2f}) | "
                      + (f"{r.fa_median_nondeg:.2f} [{r.fa_q25_nondeg:.2f}-{r.fa_q75_nondeg:.2f}] | {r.any_alarm_median_nondeg:.2f}"
                         if r.fold in NEG_CENTRES else "-") for r in s.itertuples()],
                   d=s.strategy + s.budget.astype(str)).pivot(index="d", columns="fold", values="c").to_string())
    g5 = R5.groupby(["fold", "role"])
    agg5 = g5[["AP50_all", "AP50_pos", "neg_share", "floor_img_iou0.1", "floor_polyp_iou0.1", "floor_video_iou0.1",
               "floor_img_iou0.5", "floor_polyp_iou0.5"]].agg(["mean", "min", "max"])
    print("\nR5 AP50 all vs positive-only (mean, min-max over seeds); target miss floors at lam = 0.001 (mean over seeds)")
    for (f, role), r in agg5.iterrows():
        line = (f"  {f} {role:6s} neg {r[('neg_share', 'mean')]:.2f}  AP50 all {r[('AP50_all', 'mean')]:.3f} "
                f"({r[('AP50_all', 'min')]:.3f}-{r[('AP50_all', 'max')]:.3f})  pos-only {r[('AP50_pos', 'mean')]:.3f} "
                f"({r[('AP50_pos', 'min')]:.3f}-{r[('AP50_pos', 'max')]:.3f})")
        if role == "target":
            line += "  floor IoU0.1 img/polyp/video " + "/".join(f3(r[(c, "mean")]) for c in (
                "floor_img_iou0.1", "floor_polyp_iou0.1", "floor_video_iou0.1"))
            line += "  IoU0.5 img/polyp " + "/".join(f3(r[(c, "mean")]) for c in ("floor_img_iou0.5", "floor_polyp_iou0.5"))
        print(line)
    print("wrote outputs/eval/revision_R1..R5 csv files")


if __name__ == "__main__":
    main()
