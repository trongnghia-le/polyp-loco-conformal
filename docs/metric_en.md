# Metric definitions (frozen before running experiments) — English translation

> **Translation note.** This is an English translation, made on 2026-11-17, of [`docs/metric.md`](metric.md), the analysis plan written in Vietnamese. The Vietnamese file is the authoritative record and has not been edited since commit `c89cc74`; its version history is:
> - `f196ed7` (2026-11-05): the plan as first written (IoU ≥ 0.5; α ∈ {0.05, 0.10, 0.20}; polyp = mask component);
> - `4868c6c` (2026-11-06): the change of primary criterion to IoU ≥ 0.1 and the extended α grid (section "Decision 2026-11-06"); the same commit also contains the first conformal analysis;
> - `c89cc74` (2026-11-06): the polyp unit of one official box = one polyp written into section 1, and the IoU note in section 2. That rule was already implemented in `scripts/build_manifest.py` in commit `d2b65ef` (2026-11-06 10:56), together with the training pipeline and before any detector was trained.
>
> Commit dates record when a change was saved to the repository, not when it was decided. Struck-through text is struck through in the original. Text in `code` is quoted verbatim.

All changes made after looking at results must be recorded in a "Changes" section at the end of this file, with the reason. Reference implementation: `scripts/metrics.py` (numpy, with a self-check). The mask binarisation threshold and the minimum area are in `configs/data.yaml`, keys `mask_threshold` and `min_area`.

## 1. Polyp unit
A **polyp** in a frame = one connected component (8-connectivity) of `mask > mask_threshold`, discarding components with area < `min_area` pixels. GT box = the tight bounding box of the component, `[x0, y0, x1+1, y1+1]` (half-open, original-image pixels). ~~GT boxes are always generated from the mask, including the 64 C3 images without a bbox file, so that all six centres use the same rule.~~ **Replaced on 2026-11-06** (`docs/notes/2026-11-05_seq5-and-fragments.md`): where an official bbox file exists, **each official bbox = 1 polyp** and is the GT box (a snare or cutting instrument crossing the polyp breaks the mask into fragments; fragments inside the same bbox belong to the same polyp); a mask fragment outside every bbox counts as a separate polyp and is flagged (3 frames); only the 64 C3 images without a bbox file use mask-derived boxes. Column `gt_boxes` of `data/manifest.csv`. Images with 0 components after filtering are **negative images** (including 5/31/1/81/2/5 empty-mask images at C1..C6 and all `seq*_neg` frames). Predicted boxes are mapped back to original-image pixels before IoU is computed.

## 2. Frame-level detection
For a score threshold λ, the prediction set is P(λ) = {boxes with score ≥ λ}. A polyp g is **detected** if it is matched to a box p ∈ P(λ) with IoU(p, g) ≥ 0.5 (**changed on 2026-11-06: primary criterion IoU ≥ 0.1, IoU ≥ 0.5 is a sensitivity analysis — see the Decision section at the end of the file**).

**Matching rule: one-to-one, greedy by decreasing score** (ties in input order). Visit p in order of decreasing score; p takes the unmatched GT with the highest IoU if that IoU ≥ 0.5, otherwise it matches nothing.
- Why one-to-one: if one large box were allowed to "cover" two adjacent polyps, each polyp would count as detected, giving a falsely low miss rate. One-to-one matches the clinical meaning "each polyp needs its own alarm".
- Monotone in λ: the list is visited in decreasing score order, so the matching decisions at λ' > λ are a prefix of the decisions at λ (removing low-score boxes does not change the decisions of higher-score boxes). Hence the set of polyps detected at λ' is a subset of the set at λ, and the loss in section 3 is **non-decreasing in λ** for every image. CRC needs exactly this property (a loss monotone in the parameter). The rule "any box with IoU ≥ 0.5", without exclusion, is also monotone but allows double coverage; Hungarian matching that maximises the number of pairs is monotone in total but not per polyp, and is more expensive, so it is not used.

## 3. Loss and the risk controlled by CRC
For image i with K_i ≥ 1 polyps:

L_i(λ) = (number of polyps not detected at λ) / K_i ∈ [0, 1], B = 1.

Images with K_i = 0: loss set to 0 in the code but **excluded from calibration and from the risk**. Reason: the share of negative images differs strongly between centres (C2 and C3 have no negative sequences), so mixing them in would make α incomparable; false alarms are reported separately in section 5.

Risk: R(λ) = E[L(λ)] over the distribution of positive images of the target centre. This is a mean of per-image fractions, different from the instance-level miss rate (Σ misses / Σ polyps); the latter is reported as a secondary measure, not guaranteed by CRC.

**Threshold selection** (Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, arXiv:2208.02814, ICLR 2024). The original paper assumes a loss non-increasing in λ; here the loss is non-decreasing in λ, so it is applied with λ := −λ. With n calibration images, R̂_n(λ) = (1/n) Σ L_i(λ):

λ̂ = sup { λ ∈ Λ : (n/(n+1)) · R̂_n(λ) + B/(n+1) ≤ α } = sup { λ : (Σ_i L_i(λ) + 1)/(n+1) ≤ α }

If the set is empty, λ̂ = min Λ (all boxes pass) and the draw is flagged **degenerate**. For a new test image exchangeable with the calibration images: E[L_{n+1}(λ̂)] ≤ α. This is a guarantee **in expectation**, not for each draw; therefore section 7 also reports the frequency of draws whose realised risk is > α. Grid Λ: 0.001..0.999 in steps of 0.001; predictions are stored with a confidence floor ≤ 0.001 (a configuration key set at inference time). The guarantee is meaningful only if the detector at λ = min Λ has R ≤ α; otherwise λ̂ = min Λ and the draw is counted as degenerate (not a failure of CRC).

Minimum sample size: even when R̂ = 0, 1/(n+1) ≤ α is still required, i.e. **n ≥ 1/α − 1**: α = 0.05 → 19, 0.10 → 9, 0.20 → 4. With 20 images and α = 0.05, the condition becomes Σ L_i ≤ 0.05, i.e. almost no image may have a miss.

## 4. Sequence level (C1, C2, C4, C5, C6; C3 has no sequences)
The data have no track IDs. **Assumption A1: each positive `seqK` folder contains exactly one polyp throughout, so track = (sequence folder)**. To be checked before use, with results recorded in `docs/data_audit.md`: (i) the share of frames with exactly 1 component after filtering (should be close to 100%); (ii) ordered by the frame number in the file name, IoU / centre distance between consecutive frames shows no large jumps; (iii) inspect a contact sheet of frames with ≥ 2 components. If A1 fails for a sequence, split tracks by greedy IoU linking between adjacent frames and report the number of sequences affected. Frames are sparsely sampled (e.g. 169, 320), so "frame" here means **annotated frame**.

A track is **detected at sequence level** if, in at least one of the **first N = 5 annotated frames** (by increasing frame number), the polyp is detected according to section 2 at the selected λ̂. Sensitivity: N ∈ {1, 3, 10}. N = 5 is a proposed value that needs the user's confirmation before running. This measure **depends on λ̂ but is not guaranteed by CRC**; it is reported descriptively only (C1 and C5 have only one positive sequence, C2 four, C4 nine folders but 5 source prefixes), with no violation frequency.

## 5. False alarms
On negative frames (section 1): FA(λ) = number of boxes with score ≥ λ in the frame. Report the mean per frame and per 100 negative frames, at λ̂. Negative frames comprise negative sequences and empty-mask single images; **only C1, C4, C5, C6 have negative sequences** (C2 and C3 have only a few empty single images, so FA there is descriptive only, not compared). Consecutive negative frames are strongly correlated; confidence intervals must use a cluster bootstrap (section 6).

## 6. Exchangeable unit: clusters, not frames
Frames from the same video are almost identical, so drawing frames at random mixes calibration and test images from the same video; the two sets are then not independent and the measured guarantee is falsely optimistic. Therefore every calibration/test draw is split by `cluster_id` in `data/clusters.csv` (single images, or groups by prefix; source videos by prefix, not by folder, because seq1/seq2, seq3/seq4, seq6/seq10 and seq7/seq8 share a source). One draw: shuffle the clusters of the target centre and add whole clusters to the calibration set until the number of **positive images** ≥ the budget; all remaining clusters form the test set (cluster-disjoint). The actual number of images n and number of clusters k of each draw are recorded; k is the effective independent sample size. Honest about the limitation: CRC here runs on images while the exchangeable unit is the cluster, so the theoretical guarantee is only approximate; the violation frequency measured in section 7 is the empirical test of that. A budget already exceeded by a single cluster (e.g. a large video cluster at C6) is kept, with k = 1 reported explicitly.

## 7. Reporting per held-out centre
Centres are not pooled. For each centre, α ∈ {0.05, 0.10, 0.20}, budget ∈ {20, 50, 100} positive images, R = 1000 cluster draws (fixed seed):
1. **Realised miss rate** on the test set: mean L_i (positive images), compared with α; mean over draws (should be ≤ α) and the 5–95% percentile range. Secondary: instance-level miss rate.
2. **Violation frequency** = share of draws with test risk > α.
3. **False alarms** at λ̂: per negative frame and per 100 negative frames (C1, C4, C5, C6 only).
4. **Share of degenerate draws**: λ̂ = min Λ (no threshold qualifies, including because n < 1/α − 1 or the detector is insufficient). Also report the share with λ̂ = max Λ (α too loose). At 20 images with α = 0.05, n is only slightly above the minimum of 19, so this share is expected to be high.
5. Sequence level (section 4), descriptive, with the number of sequences.
All numbers are accompanied by the mean n and mean k of the draws.

## Decisions requiring the user's confirmation
- Greedy one-to-one matching (section 2) instead of "any box".
- Negative images excluded from the CRC risk (section 3).
- Assumption A1 (one polyp per sequence) and N = 5 annotated frames (section 4).
- Drawing whole clusters until the image budget is met (the actual n may exceed the budget) (section 6).

---

## Decision 2026-11-06 (after viewing the results of seeds 0–1, before running conformal)

**Primary "detected" criterion: IoU ≥ 0.1** (the predicted box touches the polyp — the clinical meaning of an alarm); **IoU ≥ 0.5 is a sensitivity analysis**. Every other part of the definition (greedy one-to-one matching by score, polyp unit = one official bbox, per-image loss) is unchanged.

**α grid: {0.05, 0.10, 0.15, 0.20, 0.30}.** The paper focuses on 0.10–0.30; α = 0.05 is reported as a limiting case.

**Reason (recorded transparently; the change was made after seeing data):** at the held-out centre, even when every box is kept (conf ≥ 0.001), the share of polyps never boxed correctly (the "miss floor") is 6–25% at IoU ≥ 0.5 (C1: 25%) and 5–10% at IoU ≥ 0.1 (`docs/notes/2026-11-06_detection-heldout.md`). CRC only chooses a score threshold, so it cannot guarantee a miss rate below the floor; with IoU ≥ 0.5 the paper would measure YOLO's localisation ability rather than the reliability of the guarantee. The decision is based only on maximum recall (not on any CRC result) and was confirmed by the user on 2026-11-06. The paper must state this order explicitly and also report IoU ≥ 0.5.
