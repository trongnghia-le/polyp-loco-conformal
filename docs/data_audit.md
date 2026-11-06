# PolypGen v3 data audit (2026-11-05)

Source: `data/raw/PolypGen2021_MultiCenterData_v3.zip`, MD5 `bd0c0b6d70a0c0b1a5a36f6a5a200205` — matches Synapse syn45200214 (version 1, 3.18 GB, created 2022-11-17), verified 2026-11-05. Zip extracted to `data/raw/PolypGen2021_MultiCenterData_v3/`.
Scripts: `scripts/audit_masks.py`, `scripts/audit_resolutions.py` (read straight from the zip).

## Counts
Single frames C1–C6: 256 / 301 / 457 / 227 / 208 / 88 = 1537. Positive sequence frames 2225 (seq1–23), negative sequence frames 4275 (seq1_neg–23_neg). Total 8037, matches the paper.

Empty masks (no polyp) in single frames: 5 / 31 / 1 / 81 / 2 / 5.

## Label issues
- C3: 64 images have a non-empty mask but no bbox file. The dataset's own `fileStructure_all.txt` lists no bbox for them either (e.g. `C3_EndoCV2021_008`), so they are missing in the release, not lost in extraction. Masks exist, so boxes can be derived from masks (`codes/extract_PolypBoxes.py` in the release).
- C3: 1 image name has no matching mask name; 1 empty mask. To resolve by name.
- C2: 1 image marked "no polyp" in CSV and bbox file but its mask has pixels > 127. Inspect visually.
- C6: no `dataDetails` CSV.
- Masks are JPEG: binarise (threshold 127).

## Sequence → center (not stated in filenames; paper Sci Data 2023 gives only per-center totals)
Paper (Europe PMC full text): positive frames per center 484, 1166, 457, 677, 458, 520 (C1–C6); sequence frames "C2 (865), C4 (450), C6 (432)". Subtracting single frames gives C1 228, C3 0, C5 250. Paper: "No mix of centrewise sequence is done". Fig. 7 legend colours encode the center but are not readable as text.

Resolved from the paper's figures (downloaded from media.springernature.com, `41597_2023_1981_Fig7_HTML.png` and `Fig2_HTML.png`): Fig. 7 colours each sequence folder by center (positive list shows seq1–3, 5, 6, 10–13, 16, 23; negative list shows all 23). Fig. 2a gives frame totals per center. Combined with filenames, resolutions and the counts above:

| Positive seq | Frames | Center | Evidence |
|---|---|---|---|
| 1–4, 6–10 | 450 | C4 | Fig. 7 colour (1–3, 6, 10); names like C4 singles; total = text's C4 450 |
| 5 | 250 | **C2** (paper: C5) | decided 2026-11-05: near-duplicate of 3 C2 singles, C2 overlay/resolution; see `docs/notes/2026-11-05_seq5-and-fragments.md` |
| 12–15 | 865 | C2 | Fig. 7 colour (12, 13); resolutions of C2 singles; total = text's C2 865 |
| 16–23 | 432 | C6 | Fig. 7 colour (16, 23); names `seqK_C6_*`; total = text's C6 432 |
| 11 | 228 | **C1** | Burned-in endoscope UI is in French ("ID patient / Nom patient / Sexe / Date naissance") → Paris (C1); C3 (Italy) frames show English Olympus "ID. No / D.O.Birth / SCV-1" or Italian "Nome / Sesso / Data di nasc."; text totals agree (C1 484 = 256 + 228, C3 457 = singles only). Fig. 7's orange (C3) box is a figure error (pixel-checked: rgb 255,127,14). |

| Negative seq | Frames | Center | Evidence |
|---|---|---|---|
| 1–3, 8–10 | 1359 | C4 | Fig. 7 colour |
| 4–7 | 430 | C1 | Fig. 7 colour; all four carry the same French printer banner "Imprim. non connectée E405" (Paris); C4 (Oslo) frames instead show a green ScopeGuide panel |
| 11–13 | 731 | C5 | Fig. 7 colour; names `C5_seq*`; total = Fig. 2a C5 731 |
| 14–23 | 1755 | C6 | Fig. 7 colour; total = Fig. 2a C6 1755 |
All 8,037 frames now have a center. Two apparent conflicts were errors in the paper's figures, settled by the frames themselves (contact sheet: `scripts/audit_center_ui.py`):
- Fig. 7 colours positive seq11 as C3; its French UI says C1 (text totals agree).
- Fig. 2a gives negatives C1 = 223, C4 = 1566; neg seq7 has the C1 French banner, so C1 = 430, C4 = 1359 (Fig. 7 agrees).
- Fig. 2a also labels positive C5 = 520, C6 = 484; text and files give C5 = 458, C6 = 520.

Final frames per center (decision applied 2026-11-05: seq5 -> C2; single | positive seq | negative seq):

| Center | Single | Positive seq | Negative seq | Total |
|---|---|---|---|---|
| C1 | 256 | 228 | 430 | 914 |
| C2 | 301 | 1,115 | 0 | 1,416 |
| C3 | 457 | 0 | 0 | 457 |
| C4 | 227 | 450 | 1,359 | 2,036 |
| C5 | 208 | 0 | 731 | 939 |
| C6 | 88 | 432 | 1,755 | 2,275 |
| all | 1,537 | 2,225 | 4,275 | 8,037 |

**seq5 belongs to C2 (user decision, 2026-11-05).** The paper (Fig. 7 colour and the text totals) says C5, but `scripts/build_clusters.py`, `scripts/audit_seq5.py`, `audit_seq5_deep.py`, `audit_seq5_nn.py` show: C2 singles `EndoCV2021_001351/1352/1353` are near-pixel duplicates of seq5 frames 12/92/200 (mean abs diff 1.8-3.1 grey levels), the single numbers 001318-001353 are taken sequentially from five C2 videos (seq12-15 and seq5), seq5 shares the C2 1440x1064 size and left-side text overlay, and no C5 single resembles any sequence frame. This is a third inconsistency in the paper. Consequence: C2 positive sequence frames 1,115, C5 positive sequence frames 0 (C5 positives are the 206 positive single frames only); the 250 frames and the 3 singles form cluster `C2_seq5`.

Leakage risk (from the LOCO research notes, to verify when building splits): no patient IDs exist, and filename prefixes look like source-video IDs shared across folders: seq1/seq2, seq3/seq4, seq6/seq10, seq7/seq8 share a prefix, and seq3_neg comes from the same source as seq6/seq10. Any split inside a center (validation, target recalibration vs target test) must group by source prefix, not by folder.

C2 and C3 have no negative sequences, so false alarms on negative frames can only be measured at C1, C4, C5, C6 (plus the few empty-mask singles).
