# Pipeline Overview (`fig_pipeline`) and Qualitative Examples (`fig_qualitative`)

Generated on `2026-11-17` to supply the two graphical items missing from the draft:

1. **Manuscript Fig. 1 (`scripts/fig_pipeline.py` -> `results/figures/fig_pipeline.{pdf,eps,png}`):**
   - Double-column (183 × 99 mm), `configs/paper.mplstyle` with Arial TrueType embedded (`pdf.fonttype = 42`), `lw >= 1.0 pt`, 600 dpi PNG preview verified clean with `read_file`.
   - Summarises all four study stages (PolypGen v3 audit & 1,272 video clusters, 6 LOCO folds & YOLO11s/m prediction export floors, source-centre transfer vs target recalibration at $\alpha = 0.10$, target estimands / certification / alarm cost) plus the external CVC-ClinicDB benchmark.
   - Replaces no existing main-text figure and brings the main-manuscript display count to exactly 8 items (Figs 1–4 and Tables 1–4; old Figs 1–3 automatically renumber to Figs 2–4 via `\label{fig:...}`).

2. **Supplementary Fig. S5 (`scripts/fig_qualitative.py` -> `results/figures/fig_qualitative.{pdf,eps,png}`):**
   - Double-column (183 × 159 mm), TrueType Arial embedded, 600 dpi PNG verified clean with `read_file` and zero EPS transparency warnings.
   - All boxes, IoUs, and thresholds are loaded directly and asserted against `data/manifest.csv`, `outputs/kaggle/{C4,C5,C6}_seed0/preds_*_seed0.parquet`, `outputs/eval/revision2_R6a_weighted.csv`, and `outputs/eval/conformal_cluster_long.parquet`:
     - Panel a (`C5`, `EndoCV2021_C5_50000282.jpg`): top box score `0.200`, IoU `0.887` missed by pooled `0.244` and recovered at `0.128` (balanced) / `0.004` (recalibration median).
     - Panel b (`C6`, `EndoCV2021_C6_0100023.jpg`): zero candidate exports `>= 0.001` (below miss floor).
     - Panel c (`C5`, `EndoCV2021_C5_50000193.jpg`): score `0.643 >= 0.244`, centre inside GT, IoU `0.292` (hit at IoU `>= 0.1`, miss at `>= 0.5`).
     - Panel d (`C4`, `EndoCV2021_seq8_neg_45_endocv2021_negative_866.jpg`): polyp-free frame with 0 boxes at pooled (`0.269`) vs 2 cleanly separated non-overlapping (`IoU = 0.0`) false-alarm boxes (`0.041`, `0.007`) at C4 seed-0 certified median recalibration (`0.004`).
   - Placed in `paper/supplementary.tex` with an explicit `[AUTHORS: retain only after confirming CC BY 4.0 permission from the PolypGen authors]` guard matching `paper/sections/methods.tex`.
