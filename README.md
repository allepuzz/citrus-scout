# citrus-scout

MLOps pipeline for pest and disease detection in citrus trees from UAV imagery.

## Context

Phytosanitary scouting in citrus is done manually and by sampling: a technician walks the
grove following sampling protocols (GIP-IVIA) and extrapolates. It is slow, expensive, and
never produces a tree-by-tree census.

`citrus-scout` aims to automate that inspection with drone imagery and computer vision,
targeting **cooperatives and ATRIAs** in the Region of Murcia (28,442 ha of lemon trees,
~53% of Spain's national lemon production).

### Two-pass approach

| Pass | Altitude | GSD | Detects |
|---|---|---|---|
| **Screening** | 15-25 m | 0.3-0.7 cm/px | Canopy decline, vigour loss, dead trees |
| **Inspection** | < 2 m | < 0.05 cm/px | Organ-level symptoms (leaf, fruit) on flagged trees |

The screening pass covers the whole plot quickly; the inspection pass only descends over
suspicious trees. That is what keeps the per-hectare cost viable.

## Realistic targets (Region of Murcia)

**Detectable from nadir UAV imagery**, canopy-scale signature:
- Decline caused by *Phytophthora* (gummosis / foot rot)
- Tristeza (CTV)
- Water and nutrient stress

**Not detectable from nadir view**, millimetre-scale symptoms on organs:
California red scale (~2 mm), *Delottococcus aberiae*, citrus leafminer, aphids,
*Ceratitis capitata*. These require the close-range inspection pass.

**Absent from Spain**, no local ground truth possible:
HLB (*Candidatus* Liberibacter spp.) and its vectors. Spain is free of the bacterium and of
*Diaphorina citri*; *Trioza erytreae* is present in the Canary Islands and the Cantabrian
coast, but not in the Mediterranean Levante. **The entire UAV-based HLB detection literature
is therefore not reproducible here.**

## Status

Phase 0 complete: the pipeline trains, evaluates, calibrates, quantifies uncertainty
and reports where the model looks. A baseline run on the public leaf datasets reaches
PR-AUC 1.0 on its test split, which measures the task being easy rather than the problem
being solved, and is why the evaluation tooling exists.

The binding constraint now is data, not code: real UAV imagery over Murcian groves with
agronomic ground truth. Nothing in this repository improves until that exists.

## Documentation

Complete explanation of the system, every design decision, and every limitation, in
`docs/` (Spanish, written from zero assumed knowledge of machine learning):

| Document | Contents |
|---|---|
| [Index](docs/00-indice.md) | Overview and the real numbers from the baseline run |
| [1. Concepts](docs/01-conceptos.md) | Every term from scratch: specificity, PPV, PR-AUC, calibration, uncertainty, Grad-CAM, with the maths |
| [2. The data](docs/02-datos.md) | What these images actually are, the 17 classes, the 80.3% prevalence, the filtering funnel |
| [3. Training](docs/03-entrenamiento.md) | Every decision in the order it happens, explaining the baseline run line by line |
| [4. Evaluation](docs/04-evaluacion.md) | The four tools and which question each one answers |
| [5. Decisions and bugs](docs/05-decisiones.md) | Master decision table, the seven bugs found, and every limitation |
| [6. The full flow](docs/06-flujo.md) | One image from disk to an alert, and where each step lives |

Start with document 2. The nature of the data sets the ceiling on everything else.

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/allepuzz/citrus-scout.git
cd citrus-scout
uv sync --extra dev
uv pip install -e . --no-deps
```

### Kaggle credentials

Public datasets are downloaded through the Kaggle API. Create a token under
*Kaggle > Settings > API* and place it in your Kaggle config directory:

```
~/.kaggle/                  # Linux / macOS
%USERPROFILE%\.kaggle\      # Windows
```

Recent Kaggle versions issue an `access_token` file; older ones a `kaggle.json` holding
`username` and `key`. The client accepts either, trying the access token first. Never
commit these (both are covered by `.gitignore`).

## Usage

```bash
# Download the public leaf datasets (about 5.5 GB)
uv run citrus-scout data download

# Inspect what was assembled
uv run citrus-scout data summary
uv run citrus-scout data relevance

# Package for upload to Colab: 5.5 GB and 14,000 files become one 174 MB archive
uv run citrus-scout data package

# Train
uv run citrus-scout train --config configs/leaf_baseline.yaml

# Evaluate, reporting PPV at real field prevalence plus calibration
uv run citrus-scout evaluate --checkpoint runs/leaf_baseline/best.pt

# Uncertainty and triage: which trees a technician should actually visit
uv run citrus-scout uncertainty --checkpoint runs/leaf_baseline/best.pt

# Per-class recall and confusions, weighted by what occurs in Murcia
uv run citrus-scout per-class --checkpoint runs/leaf_baseline/best.pt

# Where the model looks, flagging background shortcuts
uv run citrus-scout attention --checkpoint runs/leaf_baseline/best.pt --save-to runs/cam
```

### Reading the evaluation

Four commands, answering four different questions:

| Command | Question | What to watch |
|---|---|---|
| `evaluate` | How good is it? | PPV at field prevalence, not PR-AUC |
| `uncertainty` | Where should a human look? | Whether uncertainty is higher on errors |
| `per-class` | What does it confuse? | Recall on locally present vs absent diseases |
| `attention` | Is it cheating? | Share of images explained by the frame border |

`attention` is the one that checks the others. Every metric scores *what* the model
answers; only the heatmap asks *where it looked*. With a near-perfect score those two
hypotheses, genuinely easy task versus background shortcut, predict identical numbers
everywhere else.

### Training on Colab

Training on a laptop GPU is slow and heats the machine, so the intended path is
Colab. `notebooks/colab_train.ipynb` clones this repo, installs it, and runs the same
CLI shown above; nothing that affects a result is defined in a notebook cell.

1. Locally: `uv run citrus-scout data package`
2. Upload `data/processed/leaf_dataset.zip` to Drive
3. Open the notebook in Colab, set the runtime to a T4 GPU, run the cells

The archive carries its own split assignment, so a Colab run trains on exactly the
partition your machine produced rather than recomputing one that might differ.

## Layout

```
src/citrus_scout/
├── data/          # datasets, download, transforms
├── models/        # architectures and backbone factory
├── training/      # training loop, callbacks
├── evaluation/    # metrics, calibration, uncertainty, per-class, Grad-CAM
└── utils/         # config, seeding, logging
configs/           # experiment configs (YAML)
scripts/           # standalone utilities
notebooks/         # exploration and Colab
```

## Metrics

This problem has **severe class imbalance** (prevalence of affected trees is typically
2-5%). Accuracy is misleading and is not used.

Headline metrics:
- **PR-AUC** (area under the precision-recall curve)
- **F1** and sensitivity at fixed specificity
- **PPV at real prevalence**. At 2% prevalence with 90% sensitivity and 90% specificity,
  the positive predictive value is 15.5%: roughly 6 out of 7 alerts would be false.
  Raising specificity to 99% lifts it to ~65%.

The operating point is tuned towards **high specificity**: the cost of skipping a healthy
tree is low, but flooding the technician with false positives makes the system useless.

## Data

Data is **not versioned in git**. It is managed with DVC.

Public datasets used in Phase 0 (close-range leaf images, controlled background):
PlantVillage (orange), Kaggle citrus collections, the MDPI 649-leaf dataset.

Note: these cover mostly exotic diseases (canker, HLB) and are close-range leaf shots, **not
aerial imagery**. They are useful to pre-train the classifier for the inspection pass and to
validate the pipeline, not as production data.

## Regulatory notes

Operations fall under EU Regulations 2019/947 and 2019/945 plus Spanish RD 517/2024:
- UAS operator registration with AESA (mandatory, free)
- A1/A3 training (online, free)
- Mandatory geographic-zone check on **ENAIRE Drones** before every flight
- Maximum altitude 120 m in the Open category
- Insurance not mandatory in A1/A3 under 20 kg (RD 517/2024 art. 8), but recommended

## License

MIT. See [LICENSE](LICENSE).
