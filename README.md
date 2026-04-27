# Explainability & Reliability in Medical AI Systems

MSc thesis repository. Bone fracture detection on musculoskeletal X-rays using deep
learning, with a focus on model explainability and clinical reliability.

The core thesis argument: clinical AI should **support prediction with evidence**, not
explain after the fact. Every model component is evaluated against the question —
*"How does this help a clinician make a better decision?"*

**Dataset:** FracAtlas (Abedeen et al., 2023) — 4,083 X-ray images, 717 fractured,
annotated for classification, localization, and segmentation.

---

## Phases

| Phase | Model | Task | Primary Metric | Status |
|-------|-------|------|----------------|--------|
| 1 | ResNet-18 | Binary fracture classification (E-series) | F1 (fractured class) | Complete |
| 1 | ResNet-18 | CAALMIX augmentation ablation (E5/E6/E7/E8) | F1 (fractured class) | Complete — E6 champion (68.9%); E8 isolates XRayAugMix |
| 1 | DenseNet-169 | Binary fracture classification (D-series) | F1 (fractured class) | Complete — D1 champion (72.4%) |
| 1 | DenseNet-169 | CAALMIX augmentation ablation (D3/D4/D5) | F1 (fractured class) | D3 done (CLAHE hurts −6.2pp); D4/D5 skipped |
| 1 | EfficientNet-B3 | Binary fracture classification (F-series) | F1 (fractured class) | Complete — F1 baseline (67.1% val / 56.3% test) |
| 2 | YOLOv8s / YOLOv8s-seg / YOLOv8m | Localization & segmentation | mAP@0.5 | Complete |
| 3 | CBM + Prototypes + Counterfactuals | XAI — three-pillar architecture | Task-specific | Pending |

Phase 3 is the core thesis contribution: integrating clinically-grounded attribute
explanations (Pillar 1), precedent-based example retrieval (Pillar 2), and contrastive
counterfactual explanations (Pillar 3) in a single system for fracture detection.

---

## Repository Structure

```
/
├── main.py                   # Entry point — --config, --task, --seed, --debug, --no-plot
├── configs/                  # Experiment config files (YAML)
│   ├── resnet_E4.yaml        # E4 classification experiments (E4a / E4i / E4e / E4h)
│   ├── resnet_E5.yaml        # E5 CAALMIX step 1 — CLAHE only (done, F1=66.7%)
│   ├── resnet_E6.yaml        # E6 CAALMIX step 2 — +AlbumentationsDelta (done, F1=68.9% ★)
│   ├── resnet_E7.yaml        # E7 CAALMIX step 3 — +XRayAugMix (done, F1=65.2%, regressed)
│   ├── resnet_E8.yaml        # E8 XRayAugMix standalone — no CLAHE, no Albu (done, F1=63.6%, isolates XRayAugMix harm)
│   ├── densenet_D1.yaml      # D1 DenseNet-169 baseline (flat LR, no dropout)
│   ├── densenet_D2.yaml      # D2 DenseNet-169 cosine warmup + dropout (worse — D1 is champion)
│   ├── densenet_D3.yaml      # D3 CAALMIX step 1 — CLAHE only (done, F1=66.2%, −6.2pp vs D1 — CLAHE hurts DenseNet)
│   ├── densenet_D4.yaml      # D4 CAALMIX step 2 — +AlbumentationsDelta (skipped — D3 regression unrecoverable)
│   ├── densenet_D5.yaml      # D5 CAALMIX step 3 — +XRayAugMix (skipped — D3 regression unrecoverable)
│   ├── efficientnet_F1.yaml  # F1 EfficientNet-B3 baseline (compound scaling, plateau scheduler)
│   ├── yolo_baseline.yaml    # Original Y0 runs (fractured-only, mixed optimizers)
│   ├── yolo_Y0.yaml          # Three-way reproduction: Y0A / Y0B / Y0C
│   ├── yolo_Y1.yaml          # Extended training: Y1A (patience=10) / Y1B (patience=50)
│   ├── yolo_Y2.yaml          # Resolution ablation: imgsz=640 (COCO-standard)
│   ├── yolo_Y3.yaml          # Resolution ablation: imgsz=800, batch=8 (VRAM limit)
│   ├── yolo_Y4.yaml          # Capacity ablation: YOLOv8m (25.9M params)
│   └── yolo_Y5.yaml          # Negative sampling ablation: 1:1 ratio
├── data/                     # Data preparation scripts
│   ├── prepare_classification.py  # Builds ImageFolder split dirs for ResNet/DenseNet
│   └── prepare_yolo.py            # Builds YOLO detection / segmentation datasets
├── models/
│   ├── classification/       # ResNet-18 (E-series), DenseNet-169 (D-series), EfficientNet-B3 (F-series)
│   └── yolo/                 # YOLO localization & segmentation (Y-series)
├── inference/                # FracAssist clinical decision support system
│   ├── config.py             # Fixed hyperparameters, weight paths, CUDA auto-detect
│   ├── predict.py            # GEL ensemble + GradCAM
│   └── app.py                # Flask: GET /, GET /health, POST /predict
├── index.html                # FracAssist web UI (three tabs: Assist / Model Status / Config)
├── style.css                 # Dark theme — bone-gradient plates, teal/red accents
├── scripts.js                # UI logic — fetch /predict, overlay toggle, drag-drop, zoom
├── xai/                      # XAI pillar implementations (Phase 3 — pending)
├── utils/
│   ├── logger.py             # Experiment logging
│   ├── plot.py               # Training curves, metric plots
│   ├── gradcam.py            # GradCAM — compute_overlay / to_base64 / save
│   ├── augmentations.py      # CAALMIX augmentation blocks — AlbumentationsDelta, XRayAugMix
│   ├── eval_resnet.py        # Evaluate all ResNet-18 checkpoints on val/test set
│   ├── eval_densenet.py      # Evaluate all DenseNet-169 checkpoints on val/test set
│   ├── eval_efficientnet.py  # Evaluate all EfficientNet-B3 checkpoints on val/test set
│   └── eval_gel.py           # Evaluate GEL ensemble on val/test — threshold sweep + baselines
├── results/                  # Saved metrics and plots
│   ├── experiments_yolo.csv      # All YOLO experiments — hyperparams + metrics
│   ├── experiments_resnet.csv    # All ResNet-18 experiments — hyperparams + metrics
│   ├── experiments_densenet.csv      # All DenseNet-169 experiments — hyperparams + metrics
│   ├── experiments_efficientnet.csv  # All EfficientNet-B3 experiments — hyperparams + metrics
│   ├── gel_eval_results.txt          # GEL evaluation output — both splits, baselines vs ensemble
│   └── plots/                    # Training curves (gitignored)
└── weights/                  # Saved model weights (gitignored)
```

All experiments are implemented as plain Python scripts.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Supported Models

| Model | Task | Phase |
|-------|------|-------|
| YOLOv8s | Fracture localization (detect) | 2 |
| YOLOv8s-seg | Fracture segmentation | 2 |
| YOLOv8m | Fracture localization (detect) — capacity ablation | 2 |
| YOLOv8m-seg | Fracture segmentation — capacity ablation | 2 |
| ResNet-18 | Binary classification (E-series) | 1 |
| DenseNet-169 | Binary classification (D-series) | 1 |
| EfficientNet-B3 | Binary classification (F-series) — compound scaling | 1 |
| CBM / Prototypes / Counterfactuals | XAI explainability | 3 |

---

## Key Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--config` | path to YAML | Experiment config file |
| `--task` | key in config / `all` | Which experiment(s) to run |
| `--seed` | int (default: 42) | Global random seed for all runs |
| `--debug` | flag | Override epochs=1 — tests pipeline without full training |
| `--no-plot` | flag | Disable plot generation |

> **Note:** `--split` is an argument to `utils/eval_resnet.py`, `utils/eval_densenet.py`, and `utils/eval_efficientnet.py`, not `main.py`.

---

## Usage

### 1. Prepare datasets

```bash
# Classification (Phase 1) — build ImageFolder split dirs once
python data/prepare_classification.py
# Output: data/dataset_cls/train|val|test / Fractured|Non_fractured/
# Fractured: official train/valid/test.csv splits
# Non-fractured: proportional 80/12/8, seed=42
```

| Flag | Description |
|------|-------------|
| `--out_dir` | Custom output directory (default: `data/dataset_cls`) |
| `--clean` | Wipe and rebuild from scratch |

```bash
# Y0A — paper stated splits (574 fractured only)
python data/prepare_yolo.py --out_dir data/dataset_yolo_Y0A --clean
python data/prepare_yolo.py --seg --out_dir data/dataset_yolo_seg_Y0A --clean

# Y0B — author notebook splits (574 + 61 test leak = 635)
python data/prepare_yolo.py --include_test --out_dir data/dataset_yolo_Y0B --clean
python data/prepare_yolo.py --seg --include_test --out_dir data/dataset_yolo_seg_Y0B --clean

# Y0C — full negative sampling (574 fractured + 3,366 non-fractured = 3,940)
python data/prepare_yolo.py --n_neg -1 --out_dir data/dataset_yolo_Y0C --clean
python data/prepare_yolo.py --seg --n_neg -1 --out_dir data/dataset_yolo_seg_Y0C --clean

# Y5 — balanced 1:1 negative sampling (635 frac + 635 neg train, 82 frac + 82 neg val)
python data/prepare_yolo.py --include_test --n_neg 635 --n_neg_val 82 \
    --out_dir data/dataset_yolo_Y5 --clean
```

Key flags for `prepare_yolo.py`:

| Flag | Description |
|------|-------------|
| `--seg` | Build segmentation dataset (COCO polygon labels) |
| `--n_neg -1` | Include all 3,366 non-fractured images as train negatives |
| `--n_neg N` | Include N randomly sampled non-fractured images (seed=42) |
| `--n_neg_val N` | Add N non-fractured images to the validation split (sampled jointly with `--n_neg` — no overlap) |
| `--include_test` | Append test.csv to train — reproduces author notebook behaviour (Y0B only) |
| `--out_dir` | Custom output directory |
| `--clean` | Wipe and rebuild from scratch |

### 2. Test pipeline (debug mode)

```bash
python main.py --config configs/yolo_Y0.yaml --task Y0A_localization --debug
```

`--debug` overrides epochs to 1 for a fast end-to-end pipeline check (~1 min).

### 3. Train

```bash
# DenseNet-169 (D-series — baseline)
python main.py --config configs/densenet_D1.yaml --task D1
python main.py --config configs/densenet_D2.yaml --task D2

# DenseNet-169 CAALMIX ablation (D3→D4→D5 sequentially)
python main.py --config configs/densenet_D3.yaml --task D3   # CLAHE only
python main.py --config configs/densenet_D4.yaml --task D4   # +AlbumentationsDelta
python main.py --config configs/densenet_D5.yaml --task D5   # +XRayAugMix
```

```bash
# ResNet-18 (E-series — baseline)
python main.py --config configs/resnet_E4.yaml --task E4a_m050
python main.py --config configs/resnet_E4.yaml --task E4a_m075
python main.py --config configs/resnet_E4.yaml --task E4i_d03
python main.py --config configs/resnet_E4.yaml --task E4i_d05
python main.py --config configs/resnet_E4.yaml --task E4e
python main.py --config configs/resnet_E4.yaml --task E4h_g1
python main.py --config configs/resnet_E4.yaml --task E4h_g2

# All E4 experiments sequentially
python main.py --config configs/resnet_E4.yaml --task all

# ResNet-18 CAALMIX ablation (E5→E6→E7 sequentially)
python main.py --config configs/resnet_E5.yaml --task E5   # CLAHE only
python main.py --config configs/resnet_E6.yaml --task E6   # +AlbumentationsDelta (champion)
python main.py --config configs/resnet_E7.yaml --task E7   # +XRayAugMix

# ResNet-18 isolation experiment
python main.py --config configs/resnet_E8.yaml --task E8   # XRayAugMix standalone (no CLAHE, no Albu)
```

```bash
# EfficientNet-B3 (F-series)
python main.py --config configs/efficientnet_F1.yaml --task F1   # compound scaling baseline
```

```bash
# YOLO
python main.py --config configs/yolo_Y0.yaml --task Y0A_localization
python main.py --config configs/yolo_Y0.yaml --task Y0B_localization
python main.py --config configs/yolo_Y0.yaml --task all
python main.py --config configs/yolo_Y5.yaml --task Y5_localization
```

### 4. Evaluate

```bash
# YOLO
python models/yolo/evaluate.py \
    --weights weights/Y0A_detect_best.pt \
    --data    data/dataset_yolo_Y0A/data.yaml \
    --task    detect \
    --imgsz   600

# ResNet-18 — evaluate all checkpoints, ranked by F1
python utils/eval_resnet.py              # test set (default)
python utils/eval_resnet.py --split val  # val set

# DenseNet-169 — evaluate all D-series checkpoints, ranked by F1
python utils/eval_densenet.py              # test set (default)
python utils/eval_densenet.py --split val  # val set

# EfficientNet-B3 — evaluate all F-series checkpoints, ranked by F1
python utils/eval_efficientnet.py              # test set (default)
python utils/eval_efficientnet.py --split val  # val set

# GEL — evaluate ensemble on both splits (val then test, val-optimal threshold transferred)
python utils/eval_gel.py
```

All eval scripts perform a post-hoc threshold sweep (0.05–0.95, step 0.025) and report
per-model baselines alongside the ensemble. Use `--split val` for tuning; `--split test`
for final reporting. `eval_gel.py` always runs both splits to enable val→test threshold transfer.

> **Note — CAALMIX checkpoints:** `eval_resnet.py` applies standard transforms (no CLAHE).
> For E5/E6/E7 and D3/D4/D5 checkpoints, use the post-training sweep output from the
> training log — those runs apply CLAHE correctly on the val set.

### Config format — classification

```yaml
E4e:
  experiment_id  : "E4e"
  task           : "classify"
  data_dir       : "data/dataset_cls"
  epochs         : 30
  batch_size     : 32
  img_size       : 224
  device         : "0"
  dropout_p      : 0.3           # 0.0 = no dropout
  weight_mult    : 0.5           # 0.0 = flat, 1.0 = natural imbalance ratio
  loss           : "weighted_ce" # "weighted_ce" or "focal"
  gamma          : 1.0           # focal loss only (gamma=1 mild, gamma=2 strong)
  scheduler      : "cosine_warmup" # "plateau" or "cosine_warmup"
  warmup_epochs  : 3
  lr_backbone    : 1.0e-5
  lr_head        : 1.0e-3
  val_threshold  : 0.5           # starting threshold; post-training sweep refines and saves optimal
  plot           : true
```

**CAALMIX augmentation keys** (E5/E6/E7 and D3/D4/D5):

```yaml
E6:
  experiment_id       : "E6"
  task                : "classify"        # "classify_densenet" for DenseNet
  data_dir            : "data/dataset_cls"
  epochs              : 30
  batch_size          : 32
  img_size            : 224
  device              : "0"
  dropout_p           : 0.0
  weight_mult         : 0.5
  loss                : "weighted_ce"
  scheduler           : "plateau"
  val_threshold       : 0.5
  use_clahe           : true   # CLAHE applied to ALL splits (train+val+test)
  use_albu            : true   # AlbumentationsDelta applied to TRAIN only
  use_augmix          : false  # XRayAugMix applied to TRAIN only (E7/D5 only)
  early_stop_patience : 15     # stop if no val F1 gain for N epochs; 0 = disabled
  plot                : true
```

Post-training, each run performs a val-set threshold sweep (0.05–0.95, step 0.025).
The optimal threshold is saved directly into the checkpoint (`val_threshold` key).

### Config format — EfficientNet-B3

Same structure as DenseNet-169. Key differences: `task: classify_efficientnet`, backbone uses differential LR (strong ImageNet pretraining → lower backbone LR).

```yaml
F1:
  experiment_id       : "F1"
  task                : "classify_efficientnet"   # routes to models/classification/efficientnet.py
  data_dir            : "data/dataset_cls"
  epochs              : 50
  batch_size          : 32
  img_size            : 224
  device              : "0"
  dropout_p           : 0.0
  weight_mult         : 0.5
  loss                : "weighted_ce"
  scheduler           : "plateau"
  lr_backbone         : 1.0e-5    # low — compound-scaled backbone well-pretrained
  lr_head             : 1.0e-3    # high — head re-initialised for binary task
  val_threshold       : 0.5
  early_stop_patience : 15
  plot                : true
```

After training, update `inference/config.py`:
- `"efficientnet_threshold"` → val_threshold from checkpoint
- `"gel_f1_efficientnet"` → val F1 from `eval_efficientnet.py --split val`

### Config format — DenseNet-169

Same structure as ResNet-18. Only `task` and the model-specific defaults differ:

```yaml
D4:
  experiment_id       : "D4"
  task                : "classify_densenet"   # routes to models/classification/densenet.py
  data_dir            : "data/dataset_cls"
  epochs              : 50
  batch_size          : 32
  img_size            : 224
  device              : "0"
  dropout_p           : 0.0
  weight_mult         : 0.5
  loss                : "weighted_ce"
  scheduler           : "plateau"
  lr_backbone         : 1.0e-4
  lr_head             : 1.0e-4
  val_threshold       : 0.5
  use_clahe           : true
  use_albu            : true
  early_stop_patience : 15
  plot                : true
```

D1 uses `scheduler: plateau` and flat LR (`lr_backbone: lr_head: 1e-4`) as a clean baseline.
D2 mirrors the ResNet-18 E4e champion config (cosine warmup, differential LR, dropout=0.3).
D3/D4/D5 add CAALMIX augmentation steps sequentially.
All D-series share the same ImageFolder split as E-series (`data/dataset_cls`).

### Config format — YOLO

```yaml
Y0A_localization:
  experiment_id : "Y0A_detect"
  task          : "detect"
  model_weights : "yolov8s.pt"
  data_yaml     : "data/dataset_yolo_Y0A/data.yaml"
  epochs        : 30
  patience      : 50          # early stopping; omit to use YOLO default (50)
  imgsz         : 600
  batch         : 16          # optional — omit to use YOLO default (16)
  device        : "0"         # GPU index, or "cpu"
  optimizer     : "SGD"       # explicit for Y0A paper reproduction; omit for auto
  lr0           : 0.01        # optional — only set if overriding auto optimizer
  momentum      : 0.937       # optional — only set if overriding auto optimizer
  plot          : true
```

`batch`, `lr0`, and `momentum` are optional — omit them to let YOLO's auto optimizer
select appropriate values. Only set these when explicitly reproducing a specific
configuration (e.g., Y0A SGD reproduction).

Each top-level key is a runnable task. Add new experiments by adding new keys.

---

## Dataset Setup

FracAtlas is not committed to this repository. Download from Figshare and place at the
repo root:

**Download:** https://doi.org/10.6084/m9.figshare.22363012

```
FracAtlas/
├── images/
│   ├── Fractured/
│   └── Non_fractured/
├── Annotations/
│   ├── YOLO/
│   ├── COCO JSON/
│   └── PASCAL VOC/
├── Utilities/
│   └── Fracture Split/
│       ├── train.csv    # 574 fractured
│       ├── valid.csv    # 82 fractured
│       └── test.csv     # 61 fractured
└── dataset.csv
```

> All YOLO experiments use the official Fracture Split CSVs — not a random split — to
> enable benchmark comparison with the original paper.

---

## Phase 1 — Classification Results

Baseline from E3 (Week 3–4, weighted CE multiplier=1.0):
F1 = 57.7% | Recall = 77.0% | Precision = 46.1% | AUC = 0.857

### E4 Experiment Series

| ID | Change vs previous | Key idea |
|----|-------------------|----------|
| E4a_m050 | weight_mult 1.0 → 0.5 | Reduce fracture over-emphasis; ratio ~2.4:1 |
| E4a_m075 | weight_mult 1.0 → 0.75 | Moderate reduction; ratio ~3.4:1 |
| E4i_d03 | +dropout=0.3 on E4a winner | Combat train/val loss gap (~0.27 in E3) |
| E4i_d05 | +dropout=0.5 on E4a winner | Stronger regularisation |
| E4a_m050 ★ | weight_mult=0.5 — highest val F1 | **Inference champion** (threshold=0.375) |
| E4e | +cosine warmup on E4a+E4i winner | Smoother LR convergence |
| E4h_g1 | +focal loss γ=1 | Focus on hard mis-classified fractures |
| E4h_g2 | +focal loss γ=2 | Stronger focusing (Lin et al. 2017 default) |

All runs use ResNet-18 (ImageNet pretrained), full fine-tune, Adam, differential LR
(backbone 1e-5 / head 1e-3), img_size=224, batch=32, seed=42.
Post-training threshold sweep on val saved per checkpoint automatically.

### Results (Val set — optimal threshold per experiment)

| Experiment | Threshold | F1 | Recall | Precision | Accuracy | AUC |
|------------|-----------|-----|--------|-----------|----------|-----|
| **E4a_m050 ★** | **0.375** | **65.8%** | **64.6%** | **67.1%** | **88.7%** | **0.883** |
| E4a_m075 | — | 58.0% | — | — | — | — |
| E4i_d03 | 0.275 | 63.3% | 68.3% | 69.7% | 88.5% | 0.881 |
| E4i_d05 | — | 60.9% | — | — | — | — |
| E4e | 0.425 | 63.6% | 59.8% | 68.1% | 88.5% | 0.875 |
| E4h_g1 | 0.550 | 59.6% | 58.5% | 65.8% | 87.8% | 0.866 |
| E4h_g2 | 0.500 | 60.4% | 69.5% | 53.5% | 82.0% | 0.861 |

### Final model — E4a_m050 ★ (inference champion)

E4a_m050 chosen as inference champion: highest val F1 (65.8%) at threshold=0.375.
Weights: `weights/E4a_m050_best.pth`

---

### CAALMIX Augmentation Ablation — ResNet-18 (E5/E6/E7)

CAALMIX is a custom augmentation pipeline for low-contrast, class-imbalanced X-ray datasets:

| Component | Applied to | Description |
|-----------|-----------|-------------|
| CLAHE | All splits | Local contrast enhancement — reduces scanner-to-scanner variability |
| AlbumentationsDelta | Train only | Affine (±5% shift, 90–110% scale, ±10° rotation) + ElasticTransform + GaussNoise |
| XRayAugMix | Train only | Domain-specific AugMix: CLAHE-vary, gamma, contrast, blur — Dirichlet/Beta mixing |

The three steps are tested as an additive ablation (E5 → E6 → E7).
All runs: weight_mult=0.5, plateau scheduler, no dropout, seed=42.
Early stopping at patience=15 — stops training if val F1 does not improve for 15 consecutive epochs.
TTA tested and rejected on all experiments (consistent negative results, −1 to −4pp).

#### Results (Val set — post-training threshold sweep)

| Experiment | Pipeline | F1 | Recall | Precision | AUC | Threshold | Runtime | Stopped at |
|------------|----------|----|--------|-----------|-----|-----------|---------|------------|
| E4a (baseline) | Standard aug | 65.8% | 64.6% | 67.1% | 0.883 | 0.375 | — | ep20 |
| E5 | +CLAHE | 66.7% | 62.2% | 71.8% | 0.875 | 0.475 | 26m39s | ep30 |
| **E6 ★** | **+AlbumentationsDelta** | **68.9%** | **62.2%** | **77.3%** | **0.889** | **0.525** | **31m15s** | **ep30** |
| E7 | +XRayAugMix | 65.2% | 54.9% | 80.4% | 0.878 | 0.625 | 67m48s | ep51 (early stop) |
| E8 | XRayAugMix standalone | 63.6% | 59.8% | 68.1% | 0.869 | 0.500 | 40m30s | ep31 (early stop) |

**E6 is the CAALMIX champion for ResNet-18** — +3.1pp F1 over E4a baseline, +2.2pp over CLAHE-only (E5).

Key findings:
- CLAHE (E5) provides a moderate +0.9pp gain. The primary gain comes from AlbumentationsDelta (E6, +2.2pp over E5).
- XRayAugMix (E7) **regresses** −3.7pp vs E6. E8 (XRayAugMix standalone, no CLAHE, no Albu) confirms the regression is intrinsic to XRayAugMix — not a CLAHE conflict artifact. E8 scores −2.2pp vs E4a baseline, ruling out pipeline-order confounds.
- E7 (65.2%) > E8 (63.6%): CLAHE+Albu context partially offsets XRayAugMix damage — geometric augmentation provides countervailing regularization.
- AUC peaks at E6 (0.889). XRayAugMix regresses AUC in both E7 and E8.

---

### D-series — DenseNet-169

DenseNet-169 (ImageNet pretrained), full fine-tune, Adam. Same ImageFolder split as E-series.

| ID | Scheduler | LR backbone / head | Dropout | Key idea |
|----|-----------|-------------------|---------|----------|
| D1 | Plateau | 1e-4 / 1e-4 (flat) | 0.0 | Clean baseline — matches E4a structure |
| D2 | Cosine warmup (3ep) | 1e-5 / 1e-3 | 0.3 | Mirrors E4e champion config (cosine warmup + differential LR) |
| D3 | Plateau | 1e-4 / 1e-4 | 0.0 | +CLAHE all splits (mirrors E5) |
| D4 | Plateau | 1e-4 / 1e-4 | 0.0 | +AlbumentationsDelta train-only (mirrors E6) |
| D5 | Plateau | 1e-4 / 1e-4 | 0.0 | +XRayAugMix train-only (mirrors E7) |

### Results — D-series (val-optimal threshold per experiment, confirmed via eval_densenet.py)

| Split | Experiment | Threshold | F1 | Recall | Precision | Acc | AUC | Status |
|-------|------------|-----------|-----|--------|-----------|-----|-----|--------|
| Val | **D1 ★** | **0.175** | **72.4%** | **72.0%** | **72.8%** | **90.7%** | **0.844** | Done |
| Val | D2 | 0.875 | 63.1% | 50.0% | 85.4% | — | 0.854 | Done |
| Val | D3 | 0.500 | 66.2% | 59.8% | 74.2% | — | 0.861 | Done — CLAHE hurts (−6.2pp vs D1) |
| Val | D4 | — | — | — | — | — | — | Skipped — D3 regression unrecoverable |
| Val | D5 | — | — | — | — | — | — | Skipped — D3 regression unrecoverable |
| Test | D1 ★ | 0.350 | 68.4% | 65.6% | 71.4% | 88.9% | 0.847 | Done |
| Test | D2 | 0.075 | 58.7% | 63.9% | 54.2% | 83.4% | 0.852 | Done |

**D1 is the approved DenseNet-169 baseline.** Inference threshold: 0.175 (val-sweep optimal).

Key findings:
- D1 beats ResNet-18 E4a by +6.6pp F1 on val (72.4% vs 65.8%) and +5.2pp on test (68.4% vs 63.2%).
- D2's cosine warmup + dropout=0.3 **hurts** DenseNet: DenseNet's dense connections already act as implicit regularisation; additional dropout collapses threshold stability and destabilises recall. D1's flat LR clean baseline is strictly better.
- D1 best checkpoint at epoch 13; model then overfits (train loss → 0.01, val loss → 0.5+). Early stopping with patience=15 is applied for D3/D4/D5.
- TTA hurts D1 (−3.95pp on val) — all DenseNet inference uses single forward pass.
- D3 (CLAHE only) **regresses −6.2pp** vs D1 (66.2% vs 72.4%). DenseNet's dense connections already propagate low-level contrast features across all layers — CLAHE disrupts the learned input distribution without adding discriminative information. D4/D5 skipped; −6pp is unrecoverable by adding augmentation.
- **CAALMIX is architecture-selective:** CLAHE+Albu helps ResNet-18 (+3.1pp, E6) but hurts DenseNet-169 (−6.2pp, D3). This is the primary CAALMIX empirical finding.

Weights: `weights/D1_best.pth`

---

### F-series — EfficientNet-B3

EfficientNet-B3 (ImageNet pretrained), compound scaling (φ=1.2, ~12M params), full fine-tune,
Adam, differential LR (backbone 1e-5 / head 1e-3). Same ImageFolder split as E/D-series.

| ID | Epochs | Dropout | Scheduler | Key idea |
|----|--------|---------|-----------|----------|
| F1 | 50 (early stop ep36, best ep21) | 0.3 | Plateau | Compound-scaled baseline — 3rd GEL classifier |

Early stopping patience=15. TTA tested: −3.59pp (skip — consistent with all models).

#### Results

| Split | Threshold | F1 | Recall | Precision | Acc | AUC |
|-------|-----------|-----|--------|-----------|-----|-----|
| Val | 0.525 | 67.1% | 68.3% | 65.9% | 88.7% | 0.883 |
| Test | 0.325 | 56.3% | 65.6% | 49.4% | 81.3% | 0.818 |

Val→test generalization gap: −10.7pp F1 (larger than ResNet −2.6pp and DenseNet −4.0pp).
GEL F1 anchor: 0.671 (val-based, consistent with ResNet and DenseNet anchors).

Weights: `weights/F1_best.pth`

---

### GEL — Gated Ensemble Logic Results

GEL combines ResNet-18 (E4a_m050), DenseNet-169 (D1), and EfficientNet-B3 (F1) via
performance-weighted aggregation with a direction-aware disagreement penalty. The
implementation adapts automatically to 2 or 3 loaded classifiers. Evaluated via
`utils/eval_gel.py` (threshold sweep 0.05–0.95, step 0.025). Val-optimal threshold
transferred to test.

**Hyperparameters:** τ=0.35 (BVG gate), δ=0.40 (OAM disagreement limit),
k_high=0.30 (HIGH outlier — lenient, lone fracture signal), k_low=0.10 (LOW outlier — aggressive, lone no-fracture dissenter)

| Split | Model | Threshold | F1 | Recall | Precision | AUC |
|-------|-------|-----------|-----|--------|-----------|-----|
| Val | ResNet-18 (E4a) | 0.550 | 65.4% | 61.0% | 70.4% | 0.883 |
| Val | DenseNet-169 (D1) | 0.175 | 72.4% | 71.9% | 72.8% | 0.844 |
| Val | EfficientNet-B3 (F1) | 0.525 | 67.1% | 68.3% | 65.9% | 0.883 |
| Val | **GEL ★** | **0.525** | **70.1%** | **65.9%** | **75.0%** | **0.900** |
| Test | ResNet-18 (E4a) | 0.225 | 63.2% | 70.5% | 57.3% | 0.840 |
| Test | DenseNet-169 (D1) | 0.350 | 68.4% | 65.6% | 71.4% | 0.847 |
| Test | EfficientNet-B3 (F1) | 0.325 | 56.3% | 65.6% | 49.4% | 0.818 |
| Test | **GEL ★** | **0.325** | **67.7%** | **68.9%** | **66.7%** | **0.854** |

GEL achieves the **best AUC on both splits** (0.900 val / 0.854 test), exceeding all
individual classifiers on both splits. The thesis reliability claim rests on AUC improvement
and BVG bbox authentication architecture, not F1 (DenseNet wins test F1 alone by +0.6pp).
OAM trigger rate: ~1–3% per model — rare safety mechanism, rarely active.
BVG gate pass rate: 16.5% val / 17.8% test — matches fracture prevalence (~18%).

Full results: `results/gel_eval_results.txt`

---

## Phase 2 — YOLO Baseline Results

### Paper targets (Abedeen et al., 2023 — Ultralytics 8.0.49, SGD)

| Task | Model | Box P | Box R | mAP@0.5 | Mask P | Mask R | Mask mAP@0.5 |
|------|-------|-------|-------|---------|--------|--------|--------------|
| Localization | YOLOv8s | 0.807 | 0.473 | 0.562 | — | — | — |
| Segmentation | YOLOv8s-seg | 0.718 | 0.607 | 0.627 | 0.830 | 0.499 | 0.589 |

Paper notebook trained on 635 (detect) / 635 (seg) images — not 574 as stated in the
paper text. The test split (61 images) was included in training.

### Y0A — Paper stated splits (Ultralytics 8.4.27, SGD, 574 train)

| Task | Box P | Box R | mAP@0.5 | Mask P | Mask R | Mask mAP@0.5 |
|------|-------|-------|---------|--------|--------|--------------|
| Localization | 0.597 | 0.484 | 0.484 | — | — | — |
| Segmentation | 0.692 | 0.516 | 0.531 | 0.677 | 0.505 | 0.495 |

### Y0B — Author notebook splits (Ultralytics 8.4.27, AdamW, 635 train)

| Task | Box P | Box R | mAP@0.5 | Mask P | Mask R | Mask mAP@0.5 |
|------|-------|-------|---------|--------|--------|--------------|
| Localization | 0.669 | 0.512 | 0.547 | — | — | — |
| Segmentation | 0.627 | 0.582 | 0.560 | 0.602 | 0.548 | 0.507 |

### Y0C — Full negatives (Ultralytics 8.4.27, AdamW, 3,940 train)

| Task | Box P | Box R | mAP@0.5 | Mask P | Mask R | Mask mAP@0.5 |
|------|-------|-------|---------|--------|--------|--------------|
| Localization | 0.333 | 0.297 | 0.290 | — | — | — |
| Segmentation | — | — | — | — | — | — |

Config: `epochs=30`, `imgsz=600`, COCO pre-trained weights, `seed=42`.

### Y1 — Extended training (epochs=200, patience=50, Y0B split)

| Run | Epochs (stopped) | Box mAP@0.5 | Mask mAP@0.5 | P | R |
|-----|-----------------|-------------|--------------|---|---|
| Y1A detect | 62 | 0.508 | — | 0.674 | 0.440 |
| Y1B detect | 200 | **0.651** | — | 0.761 | 0.595 |
| Y1A seg | 58 | 0.580 | 0.518 | 0.691 | 0.466 |
| Y1B seg | 179 | **0.608** | **0.546** | 0.802 | 0.516 |

Y1A uses `patience=10` (aggressive); Y1B uses `patience=50`. Y1A stopped early in both
tasks — patience=10 is too aggressive for this dataset. Y1B is the best baseline going
forward. Y1B detect is +8.9pp above the paper; Y1B seg mask mAP is −4.3pp below paper.

### Y2 / Y3 / Y4 — Ablations (complete)

All ablations: `epochs=200`, `patience=50`, `optimizer=auto`, `seed=42`, Y0B split.

#### Resolution ablation — Localization

| Exp | imgsz | Best epoch | mAP@0.5 | P | R |
|-----|-------|-----------|---------|---|---|
| Y1B (reference) | 600 | 170 | **0.651** | 0.761 | 0.595 |
| Y2 | 640 | 132 | 0.607 | 0.683 | 0.582 |
| Y3 | 800 | 123 | 0.601 | 0.693 | 0.549 |

#### Resolution ablation — Segmentation

| Exp | imgsz | Best epoch | Box mAP@0.5 | Mask mAP@0.5 | P | R |
|-----|-------|-----------|-------------|--------------|---|---|
| Y1B (reference) | 600 | 169 | **0.608** | **0.546** | 0.802 | 0.516 |
| Y2 | 640 | 114 | 0.559 | 0.489 | 0.711 | 0.451 |
| Y3 | 800 | 124 | 0.561 | 0.500 | 0.704 | 0.522 |

600px is optimal for FracAtlas on both tasks. Resolution increase monotonically hurts performance. COCO-alignment hypothesis (640px) rejected.

#### Negative sampling ablation — Localization

| Exp | Neg ratio | Train size | Val size | mAP@0.5 | P | R |
|-----|-----------|-----------|----------|---------|---|---|
| Y1B (reference) | 0 (frac-only) | 635 | 82 | **0.651** | 0.761 | 0.595 |
| Y5 | 1:1 | 1270 | 164 | 0.536 | 0.727 | 0.505 |
| Y0C | 1:6 | 3940 | 82 | 0.290 | 0.335 | 0.297 |

Negative sampling monotonically degrades YOLO fracture detection on FracAtlas. The 1:1
ratio (Y5) partially recovers from Y0C's collapse but remains −11.5pp below the
fractured-only baseline. Y1B confirmed as champion.

#### Capacity ablation — Localization (YOLOv8m vs YOLOv8s)

| Exp | Model | imgsz | Best epoch | mAP@0.5 | P | R |
|-----|-------|-------|-----------|---------|---|---|
| Y1B (reference) | YOLOv8s | 600 | 170 | **0.651** | 0.761 | 0.595 |
| Y4 | YOLOv8m | 600 | 119 | 0.613 | 0.705 | 0.538 |

YOLOv8m underperforms YOLOv8s by −3.8pp. Larger model overfits on the small dataset (~635 train images). Y4 segmentation not run — detect result sufficient to confirm the pattern.

#### Summary — Best results

| Task | Model | imgsz | mAP@0.5 | Mask mAP@0.5 | vs Paper |
|------|-------|-------|---------|--------------|---------|
| Localization | YOLOv8s (Y1B) | 600 | **0.651** | — | +8.9pp |
| Segmentation | YOLOv8s-seg (Y1B) | 600 | 0.608 | **0.546** | −4.3pp |

> **Reproduction notes:**
> - Version drift between Ultralytics 8.0.49 (paper) and 8.4.27 introduces new
>   augmentation defaults (`erasing=0.4`, `rle=1.0`) and changes optimizer auto-selection
>   from SGD to AdamW. Y0A uses explicit SGD to match the paper; Y0B/C use AdamW.
> - Y0B reproduces the author notebook's test-set leak (635 train). This is a
>   methodological flaw in the paper — documented here for transparency.
> - Y0C confirms that flooding training with 3,366 negatives at a 6:1 ratio severely
>   hurts localization recall when the validation set contains only fractured images.
> - Y5 (1:1 ratio, balanced val) shows partial recovery vs Y0C but still −11.5pp below
>   Y1B. Negative sampling monotonically degrades detection on this dataset.

---

## FracAssist — Inference System

A local web app for clinical decision support. Runs entirely offline; no data leaves the machine.

```bash
# Weights required — place in weights/ before starting:
#   Y1B_detect_best.pt     (required — YOLO detector)
#   E4a_m050_best.pth      (required for GEL — ResNet-18 classifier)
#   D1_best.pth            (required for GEL — DenseNet-169 classifier)
#   F1_best.pth            (optional — EfficientNet-B3 classifier, GEL adapts if absent)

python inference/app.py
# → http://127.0.0.1:5000
```

### GEL — Gated Ensemble Logic (primary mode)

The default inference mode is **GEL**, a three-stage reliability architecture:

```
Upload X-ray (JPG / PNG)
        │
        ▼
  All models run in parallel:
  ┌──────────────────────────────────────────────────────┐
  │  YOLO · Y1B · conf ≥ 0.25                           │
  │  ResNet-18 · E4a · threshold = 0.375                │
  │  DenseNet-169 · D1 · threshold = 0.175              │
  │  EfficientNet-B3 · F1 · threshold = 0.525 (optional) │
  └──────────────────────────────────────────────────────┘
        │
        ▼
  OAM — Direction-Aware Asymmetric Outlier-Aware Modification
  (|p_i − μ| > 0.40 triggers penalty; direction determines severity)
  HIGH outlier (p_i > μ, lone fracture signal)      → k_high = 0.30  lenient  — preserve fracture signal
  LOW  outlier (p_i < μ, no-frac dissenter)         → k_low  = 0.10  aggressive — protect fracture consensus
        │
        ▼
  PDWF — Performance-weighted Decision Fusion
  (F1-weighted average of OAM-adjusted classifiers — adapts to 2 or 3)
        │
        ▼
   p_final ──► BVG gate: p_final ≥ τ=0.35 → YOLO bbox authenticated and shown
               p_final < τ=0.35 → bbox suppressed (p_final still reported)
        │
        ▼
   fracture_probability  label  xray_with_box  gradcam_image (DenseNet-169 denseblock4)
```

OAM uses direction-aware asymmetric penalties: a HIGH outlier (lone fracture signal) receives a lenient penalty (k=0.30) to preserve the fracture signal; a LOW outlier (lone no-fracture dissenter against a fracture consensus) receives an aggressive penalty (k=0.10) to prevent P_final being dragged toward a missed fracture. Both directions protect against missed fractures from opposite sides. A symmetric balanced reference (k=0.20) is preserved in config for comparison.

BVG uses `p_final` — the same OAM-adjusted PDWF output that becomes the fracture probability — rather than a separate pre-OAM intermediate. The gate and the clinical output are derived from an identical, calibrated estimate.

**Inference modes** (selectable via UI dropdown):

| Mode | Key | Description |
|------|-----|-------------|
| GEL | `gel` | Default — all 3 models + BVG/OAM/PDWF |
| YOLOv8s only | `yolo` | Detection only; no classification |
| ResNet-18 + DenseNet-169 | `resnet` | Classifiers only; no localization |

### API

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Serve `index.html` |
| `GET` | `/health` | `{"status": "ok", "device": "cuda:0"}` |
| `POST` | `/predict` | `multipart/form-data` with `image` field → inference JSON |

---

## Reproducibility

- Global seed: `42` (applied to Python `random`, NumPy, PyTorch, and CUDA via `--seed`).
- Validation set is used for all tuning decisions; test set is used once per phase.
- Experiment IDs are stable: `E-series` (ResNet-18), `D-series` (DenseNet-169), `Y-series` (YOLO).
- `--debug` flag overrides `epochs=1` for fast pipeline validation without touching configs.
- All classification training logs saved to `results/logs/`.
- Post-training threshold sweep (0.05–0.95, step 0.025) runs automatically and saves the optimal threshold into the checkpoint.

---

## Citation

```bibtex
@article{abedeen2023fracatlas,
  title={FracAtlas: A Dataset for Fracture Classification, Localization and
         Segmentation of Musculoskeletal Radiographs},
  author={Abedeen, Ifra and others},
  journal={Scientific Data},
  publisher={Nature Portfolio},
  year={2023},
  doi={10.1038/s41597-023-02432-4}
}
```
