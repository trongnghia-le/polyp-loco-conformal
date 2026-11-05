# polyp-loco-conformal

Leave-one-center-out evaluation on PolypGen, YOLO → SAM2, and conformal risk control of polyp miss rate under center shift. Data audit and metric definitions: [docs/](docs/); working notes: [docs/notes/](docs/notes/).

| Folder | Holds | In git |
|---|---|---|
| `docs/` | data audit, metric definition, decisions, dated notes | yes |
| `data/raw/` | PolypGen zip + extracted copy, never edited | no |
| `splits/` | fixed LOCO split files (public with the paper) | yes |
| `configs/` | one config per experiment (model, seed, α, paths) | yes |
| `scripts/` | data audit, split building, evaluation, conformal (CPU, run locally) | yes |
| `kaggle/` | notebooks that train/infer on Kaggle GPUs and export predictions | yes |
| `outputs/` | checkpoints and predictions downloaded from Kaggle | no |
| `results/` | final tables and figures for the paper | yes |

Compute: GPU work on Kaggle; this laptop has no NVIDIA GPU.
