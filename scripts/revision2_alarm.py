"""Alarm share of target recalibration on 20 videos with the drawn videos EXCLUDED (review round 2, minor 7).
Stored YOLO11s predictions only. Re-creates the cluster draws of conformal_cluster.py (default_rng([2026, fold, k, 11]))
and takes the per-draw threshold and degenerate flag from conformal_cluster_long.parquet; asserts that the recomputed
false alarms per polyp-free TEST frame equal the stored fa_neg of every draw (so the draws are the same).
Alarm share = share of polyp-free test frames with >= 1 box at the deployed threshold; median over certified
(non-degenerate) draws pooled over the three seeds, as revision_R4 (which used all polyp-free target frames).
Writes outputs/eval/revision2_alarm.csv. Usage: .venv/Scripts/python.exe -I scripts/revision2_alarm.py"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from conformal import clean, G, load_run  # noqa: E402

EV = ROOT / "outputs" / "eval"
K_, R_, A0 = 20, 500, 0.10


def main():
    man = pd.read_csv(ROOT / "data" / "manifest.csv")
    GT = {clean(p): np.asarray(json.loads(gb), float).reshape(-1, 4) for p, gb in zip(man.path, man.gt_boxes)}
    lc = pd.read_parquet(EV / "conformal_cluster_long.parquet",
                         columns=["fold", "seed", "iou", "alpha", "strategy", "budget", "rep", "lam", "degenerate", "fa_neg"])
    lc = lc[(lc.iou == 0.1) & (lc.alpha == A0) & (lc.budget == K_) & lc.strategy.isin(["fs_a", "fs_b"])]
    r4 = pd.read_csv(EV / "revision_R4_degenerate_fa.csv")
    acc, out = {}, []
    for d in sorted((ROOT / "outputs" / "kaggle").glob("C*_seed*")):
        fold, seed = d.name.split("_seed"); seed = int(seed); fi = int(fold[1:])
        if fold not in ("C1", "C4", "C5", "C6"):
            continue
        idx, K, N, L, _ = load_run(d, fold, GT, np.random.default_rng([2026, fi, seed, 99]))
        tgt, pos = (idx.role == "target").to_numpy(), K > 0
        ucl, ci = np.unique(idx.cluster_id.to_numpy(), return_inverse=True); C = len(ucl)
        m = np.bincount(ci, weights=pos, minlength=C)
        c_tgt = np.bincount(ci, weights=tgt, minlength=C) > 0
        tgtP = np.flatnonzero(c_tgt & (m > 0))
        ke = min(K_, len(tgtP) - 10)
        r1 = np.random.default_rng([2026, fi, K_, 11])
        sel = np.array([r1.choice(tgtP, ke, replace=False) for _ in range(R_)])
        neg = tgt & ~pos
        for st in ("fs_a", "fs_b"):
            x = lc[(lc.fold == fold) & (lc.seed == seed) & (lc.strategy == st)].sort_values("rep")
            assert len(x) == R_
            j = np.searchsorted(G, x.lam.to_numpy())
            for r in range(R_):
                test_neg = neg & ~np.isin(ci, sel[r])
                fa = N[test_neg, j[r]].mean()
                assert abs(fa - x.fa_neg.iloc[r]) < 1e-9, (d.name, st, r)
                acc.setdefault((fold, st), []).append((bool(x.degenerate.iloc[r]), (N[test_neg, j[r]] >= 1).mean(),
                                                      (N[neg, j[r]] >= 1).mean()))
        print(d.name, "draws reproduced", flush=True)
    for (fold, st), v in sorted(acc.items()):
        v = np.array(v, float); cert = v[:, 0] == 0
        ref = r4[(r4.fold == fold) & (r4.iou == 0.1) & (r4.alpha == A0) & (r4.strategy == st) & (r4.budget == K_)].iloc[0]
        allneg = np.median(v[cert, 2])
        assert abs(allneg - ref.any_alarm_median_nondeg) < 1e-12, (fold, st, allneg, ref.any_alarm_median_nondeg)
        out.append(dict(fold=fold, strategy=st, n_draws=len(v), n_certified=int(cert.sum()),
                        alarm_share_test_median_cert=np.median(v[cert, 1]), alarm_share_allneg_median_cert=allneg,
                        alarm_share_test_q25=np.quantile(v[cert, 1], .25), alarm_share_test_q75=np.quantile(v[cert, 1], .75)))
    df = pd.DataFrame(out); df.to_csv(ROOT / "outputs" / "eval" / "revision2_alarm.csv", index=False)
    print("R4 all-polyp-free medians reproduced;\n", df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
