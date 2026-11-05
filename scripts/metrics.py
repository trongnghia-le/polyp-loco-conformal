"""Metrics for CRC of polyp miss rate (spec: docs/metric.md). numpy only.
Boxes are [x0, y0, x1, y1] (float, half-open pixel coords), one row per box."""
import numpy as np


def iou(a, b):
    """Pairwise IoU, a: (m,4), b: (n,4) -> (m,n). Degenerate (zero-area) union gives 0."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    ar = lambda x: np.clip(x[:, 2:] - x[:, :2], 0, None).prod(-1)
    union = ar(a)[:, None] + ar(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def miss_fraction(gt, boxes, scores, lam, thr=0.5):
    """Fraction of GT polyps missed at score threshold lam. 0 GT -> 0.0 (by convention;
    such images are excluded from calibration/risk, see docs/metric.md).
    Greedy one-to-one matching in descending score (stable); each box takes the
    still-unmatched GT with highest IoU >= thr."""
    gt = np.asarray(gt, float).reshape(-1, 4)
    if len(gt) == 0:
        return 0.0
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    order = np.argsort(-scores, kind="stable")
    order = order[scores[order] >= lam]
    m = iou(boxes[order], gt)
    free = np.ones(len(gt), bool)
    for row in m:
        row = np.where(free, row, -1.0)
        j = row.argmax()
        if row[j] >= thr:
            free[j] = False
    return float(free.sum()) / len(gt)


def loss_matrix(images, grid):
    """images: list of (gt, boxes, scores) -> (n_images, len(grid)) miss fractions."""
    return np.array([[miss_fraction(g, b, s, lam) for lam in grid] for g, b, s in images])


def crc_lambda(L, alpha, grid):
    """Largest lam in ascending `grid` with (sum_i L_i(lam) + 1) / (n + 1) <= alpha (B = 1).
    Same as Angelopoulos et al. (arXiv 2208.02814) with their lam := -lam, since our loss is
    non-decreasing in lam. Returns (lam_hat, degenerate); if no lam qualifies, returns
    grid[0] (most permissive) with degenerate=True."""
    L = np.asarray(L, float)
    n = len(L)
    ok = (L.sum(0) + 1.0) / (n + 1.0) <= alpha
    if not ok.any():
        return grid[0], True
    return grid[np.flatnonzero(ok)[-1]], False


def false_alarms(boxes_scores_neg, lam):
    """Mean number of boxes with score >= lam per negative frame, and per 100 frames."""
    per = np.array([(np.asarray(s) >= lam).sum() for s in boxes_scores_neg], float)
    return per.mean(), 100 * per.mean()


if __name__ == "__main__":
    g1 = [[0, 0, 10, 10]]
    assert abs(iou(g1, [[0, 0, 10, 5]])[0, 0] - 0.5) < 1e-9 and iou(g1, [[20, 20, 30, 30]])[0, 0] == 0
    assert iou([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0] == 0  # zero area, no NaN

    # one image, two polyps A,B; box 1 (score .9) hits A, box 2 (score .4) hits B, box 3 (.3) is junk
    A, B = [0, 0, 10, 10], [50, 50, 60, 60]
    bx = [[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110]]
    sc = [0.9, 0.4, 0.3]
    assert miss_fraction([A, B], bx, sc, 0.0) == 0.0
    assert miss_fraction([A, B], bx, sc, 0.5) == 0.5
    assert miss_fraction([A, B], bx, sc, 0.95) == 1.0
    assert miss_fraction([], bx, sc, 0.5) == 0.0  # 0 polyps convention
    # one-to-one: a single big box covering both adjacent polyps counts for only one of them
    C, D = [0, 0, 10, 10], [0, 0, 10, 11]
    assert miss_fraction([C, D], [[0, 0, 10, 10.5]], [0.9], 0.5) == 0.5

    # monotone in lam on random data (non-decreasing miss fraction)
    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 101)
    imgs = []
    for _ in range(40):
        k = rng.integers(0, 4)
        gt = rng.uniform(0, 80, (k, 2)); gt = np.hstack([gt, gt + 20])
        nb = rng.integers(0, 6)
        bx = np.vstack([gt + rng.normal(0, 4, gt.shape), rng.uniform(0, 80, (nb, 2)).repeat(2, 1) + [0, 0, 20, 20]])
        imgs.append((gt, bx, rng.uniform(0, 1, len(bx))))
    L = loss_matrix(imgs, grid)
    assert (np.diff(L, axis=1) >= -1e-12).all()

    # CRC: toy L, n=19, alpha=.05 -> needs zero misses; n=18 -> degenerate
    gridt = np.array([0.1, 0.5, 0.9])
    L19 = np.zeros((19, 3)); L19[0, 2] = 1.0  # one miss only at lam=.9
    assert crc_lambda(L19, 0.05, gridt) == (0.5, False)  # (0+1)/20 = .05 <= .05
    assert crc_lambda(np.zeros((18, 3)), 0.05, gridt) == (0.1, True)  # 1/19 > .05
    assert crc_lambda(np.zeros((9, 3)), 0.10, gridt) == (0.9, False)  # n = 1/alpha - 1 suffices
    assert np.allclose(false_alarms([[0.9, 0.2], [], [0.6]], 0.5), (2 / 3, 200 / 3))
    print("ok")
