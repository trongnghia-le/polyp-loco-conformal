# Figures

| File | Shows | Note |
|---|---|---|
| `fig_dataset_composition` | PolypGen frames per center by source (single / positive sequence / negative sequence) | `docs/notes/2026-11-05_data-audit.md` |
| `fig_splits` | The six LOCO folds: images (a) and positive frames (b) per role (train / val / cal / target) | `docs/notes/2026-11-06_splits.md` |
| `fig_risk_vs_alpha` | Realized target miss rate vs α per held-out centre (IoU ≥ 0.1): in-domain cluster split, pooled source CRC, per-source worst case, target few-shot (50); y = α diagonal; error bars = range over seeds | `docs/notes/2026-11-07_conformal-first-pass.md` |
| `fig_fewshot_budget` | Violation frequency of target few-shot CRC vs budget 20/50/100 per held-out centre, α = 0.10 (a) and 0.20 (b), with the in-domain reference | `docs/notes/2026-11-07_conformal-first-pass.md` |
| `fig_cluster_budget` | Few-shot target recalibration counted in labelled videos: mean cluster-averaged target risk − α vs k labelled positive target clusters per held-out centre (IoU ≥ 0.1), α = 0.10 (top) and 0.20 (bottom); cluster-level CRC, one frame per video, frame-budget few-shot at its median cluster count | `docs/notes/2026-11-07_cluster-level-crc.md` |
