# Source-video clusters (`data/clusters.csv`)

PolypGen has no patient IDs. `cluster_id` is the best estimate of the source video (or patient, for C1 singles) so that frames of one video never straddle a split. Built by `scripts/build_clusters.py` (~3 min; ends with an assert block: 8037 unique rows, no cluster spans two centers, no cross-center near-duplicates).
Columns: `path` (relative to `PolypGen2021_MultiCenterData_v3/`), `center` (from `docs/data_audit.md`), `cluster_id`, `evidence`.
Note: `data/raw/*.zip` was no longer present when this was built (only the extracted folder); the script reads the zip if it exists, else the extracted folder, read-only. Both give the same 8037 paths.

## Rule per naming pattern
| Pattern | Key identifying the source | Evidence code |
|---|---|---|
| C1 single `136OLCV1_100S0005.jpg` | leading number (`136`) = case/patient ID: 158 distinct values, up to 18 frames each, mixed 1080x1350 and 1048x1232 resolutions inside one ID, different polyps within an ID. 3 files have no number (`100S0001`, ...) | `name_prefix` / `singleton` |
| C4 single and C4 sequences `N_endocv2021_positive_M.jpg` | `N` = source video (0..35 positive). `M` = frame index in that video (<= 1500). Key = `p{N}` | `name_prefix` |
| C4 negatives `EndoCV2021_seqK_neg_N_endocv2021_negative_M` | `N` = negative video (40, 45, 47, 48, 49; C4 single negatives use 35). Key = `n{N}` | `name_prefix` |
| `EndoCV2021_seq3_neg_2_endocv2021_positive_M` | the embedded `2_endocv2021_positive` = positive video 2 -> key `p2` (same as seq6/seq10) | `name_prefix` |
| C6 `seqK_C6_M`, C5 `EndoCV2021_C5_seqK_M`, C2 `EndoCV2021_seqK_M`, C1 seq11/neg 4-7 `Image_N`, C6 neg `seqK_neg_M` | no video id in the name; each sequence folder is one contiguous clip (consecutive frame indices, gap = constant sampling step), so key = folder | `folder` |
| C2 `EndoCV2021_001000`, C3 `C3_EndoCV2021_00110`, C5 `EndoCV2021_C5_5000xxxx`, C6 `EndoCV2021_C6_01000xx` singles | none (numbers are not ordered by video: adjacent-id 32x32 correlation median 0.54-0.75 vs 0.39-0.55 random, <2% of adjacent pairs >= 0.97) | `singleton` unless near-dup |

Then, same center only: groups are merged when any frame pair is a near-duplicate (32x32 grayscale correlation >= 0.99 AND mean abs grey difference <= 3 at 128 px, ignoring the left 30% UI panel) -> `near_dup`. One calibrated weak link is merged: C4 `n40` + `p2` -> `sim_link`. Everything else stays separate. `cluster_id` = name of the largest merged group (e.g. `C4_vp0`, `C2_seq12`, `C3` singles `C3_s_<file stem>`; `C4_vn40` holds seq10_neg + seq6 + seq10 + seq3_neg).
Calibration: inside C4 the names are authoritative; over all 2036 C4 images, corr >= 0.99 gives 1480 same-video pairs and **0 different-video pairs** (0.98: 7; 0.95: 694, almost all n40/p2). 92% of frames have their global nearest neighbour in the same video. Global low-res correlation is therefore a reliable duplicate detector at 0.99 but not a same-video detector.

## Claimed pairs: verification
All pairs share the C4 name prefix, i.e. the same source video, and the frame indices are disjoint segments of it, so no pixel duplicates are expected. Image evidence (contact sheets, colour-histogram nearest neighbours of the endoscope area, burned-in ScopeGuide panel):

| Claim | Name evidence | Image evidence | Verdict |
|---|---|---|---|
| seq1 / seq2 | both video 0; frames 69-327 vs 927-1194 | same bowel region and polyp, same panel glyph; colour-NN seq2->seq1 28/63, seq1->seq2 13/36 (best low-res corr 0.92) | confirmed |
| seq3 / seq4 | both video 1; seq3 ends at frame 343, seq4 starts at 357 | consecutive in the contact sheet (same dark pool and fold at 343/357); colour-NN seq3->seq4 6/15 | confirmed (one continuous clip split in two) |
| seq6 / seq10 | both video 2; frames 1086-1478 and 919-1079 | same two-glyph panel, same yellow debris scenes; colour-NN seq10->seq6 15/25 | confirmed |
| seq7 / seq8 | both video 3; frames 44-282 and 724-1102 | same lumen/polyp scenes; colour-NN seq7->seq8 38/48 | confirmed |
| seq3_neg with seq6/seq10 | seq3_neg files are literally `2_endocv2021_positive_*` (frames 28-371) = video 2 | same panel and debris; colour-NN seq6->seq3_neg 56/91 | confirmed: the "negative" frames are the polyp-free stretch of positive video 2 |

Extra findings:
- seq9 = video 7 alone (no C4 single has prefix 7). C4 singles cover videos 4-6, 8-35 only, so singles and sequences share no video id.
- seq10_neg (negative video 40) is not named like p2, but 140 frame pairs with p2 have corr >= 0.97 (background between any other pair of different C4 videos: about 2) and it shares the two-glyph panel and yellow-debris content. Merged (`sim_link`); one name-independent clue, treat as probable not proven (no pair >= 0.99).
- Not confirmed, kept separate but flagged ("weak"): C1 seq4_neg / seq7_neg (39 pairs >= 0.97, same blue banner; they are Image_18334-22421 vs 745416-753629); C6 seq19_neg / seq21_neg (1107 pairs); C6 seq14_neg, 15, 16, 17, 18, 23 (thousands of pairs but the frames are flat orange mucosa, so correlation is driven by the mask and colour, MAD >= 4.2, not usable as evidence). If the target center is C6 or C1, split those folders as a block.

## Single <-> sequence overlap
- **C4: none.** No name overlap, no pixel overlap (best cross corr 0.93).
- **C1, C3, C5: none** (C1 singles vs seq11 / neg 4-7: best 0.81; C5 singles vs C5 sequences < 0.76; C3 has no sequences).
- **C2: yes.** Singles are frames of the sequences' videos: 001318 and 001320 = seq12 frames (e.g. seq12_968, MAD 1.0; merged into `C2_seq12`), 001346 = seq15 (seq15 contains a frozen stretch: 34 frames identical to it; merged into `C2_seq15`). Borderline, not merged (corr >= 0.99 but MAD 3-6): 001334 ~ seq13_44, 001344 ~ seq14_584, 001140 ~ seq15 frames. So many more C2 singles are probably from seq12-15 videos without being pixel-identical.
- **C6: yes.** 0100111 and 0100112 = seq6_C6 (`C6_seq21`) frames 384/252; 0100091 = seq7_C6 (`C6_seq22`) frames 840-930; 3 more borderline pairs with seq16/seq17.
- **seq5 (decided 2026-11-05: C2, not C5).** C2 singles 001351/001352/001353 are pixel near-duplicates of seq5 frames 12/92/200 (MAD 1.2/2.9/2.9) and seq5 is mapped to C2 (`docs/notes/2026-11-05_seq5-and-fragments.md`), so they now merge into `C2_seq5` by the ordinary same-center near-duplicate rule. The former `xcenter_dup` special case is gone; the script asserts that no cross-center near-duplicate pair remains.

## Counts
Clusters per center (images): C1 166 (914), C2 297 (1416), C3 457 (457), C4 40 (2036), C5 209 (939), C6 103 (2275); total 1272.
Singletons: C1 111, C2 289, C3 457, C4 1, C5 204, C6 85 = 1147 (1147 of 8037 images, 14%).
Size distribution (clusters): 1 image: 1147; 2-5: 65; 6-20: 20; 21-50: 2; 51-100: 14; 101-300: 19; >300: 5 (largest `C4_vn40` 448, C4 n49 315, n45 311, n48 302, `C6_seq14_neg` 301). Median multi-frame cluster: C1 2, C2 183, C4 8, C5 190, C6 101.5.
Evidence column: `folder` 3973 images, `name_prefix` 1732, `singleton` 1147, `folder+near_dup` 718, `name_prefix+sim_link` 448, `near_dup` 19 (singles joined to singles/sequences).

## Remaining uncertainty
1. **Singles with no name key (all of C3; most of C2, C5, C6 singles; C1 without prefix)** are singleton clusters. Only 15 C3 images have any other C3 image at corr >= 0.95, but frames of one video that are not near-identical cannot be linked, so splitting C3 (and C2/C5/C6 singles) still leaks videos/patients to an unknown extent. 1150 "singletons" is an upper bound on independent sources, not an estimate. Safest for calibration/test splits in these centers: split by center, or accept optimistic estimates.
2. C1 prefix is probably patient-level (several polyps per prefix); C1 seq11 and neg 4-7 may belong to the same patients as singles (no pixel evidence either way).
3. C4 single prefixes are video ids (zero cross-video pairs at 0.99) but two videos may share a patient. n40 merge is a similarity judgement.
4. C6 negative folders, C1 neg4/neg7, C6 seq19_neg/seq21_neg possibly shared sources (see above); sequence folders are assumed to be one video each (consecutive frame numbers), except C6 seq14_neg (frames 1-1050 with one 750-frame gap, so two clips) and C6 seq15_neg (one 92-frame gap).
5. (resolved) seq5 is a C2 video; its 250 frames and the 3 matching C2 singles form `C2_seq5`. C5 now has no positive sequence frames (208 singles + 731 negative sequence frames).
6. Thresholds were calibrated on C4 only (low-res correlation); other centers have different frame statistics.
