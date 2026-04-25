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
| 1 | DenseNet-169 | Binary fracture classification (D-series) | F1 (fractured class) | Complete |
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
│   ├── densenet_D1.yaml      # D1 DenseNet-169 baseline (flat LR, no dropout)
│   ├── densenet_D2.yaml      # D2 DenseNet-169 cosine warmup + dropout + threshold sweep
│   ├── yolo_baseline.yaml    # Original Y0 runs (fractured-only, mixed optimizers)
│   ├── yolo_Y0.yaml          # Three-way reproduction: Y0A / Y0B / Y0C
│   ├── yolo_Y1.yaml          # Extended training: Y1A (patience=10) / Y1B (patience=50)
│   ├── yolo_Y2.yaml          # Resolution ablation: imgsz=640 (COCO-standard)
│   ├── yolo_Y3.yaml          # Resolution ablation: imgsz=800, batch=8 (VRAM limit)
│   ├── yolo_Y4.yaml          # Capacity ablation: YOLOv8m (25.9M params)
│   └── yolo_Y5.yaml          # Negative sampling ablation: 1:1 ratio (635 neg train + 82 neg val)
├── data/                     # Data preparation scripts
│   ├── prepare_classification.py  # Builds ImageFolder split dirs for ResNet
│   └── prepare_yolo.py            # Builds YOLO detection / segmentation datasets
├── models/
│   ├── classification/       # ResNet-18 (E-series) and DenseNet-169 (D-series)
│   └── yolo/                 # YOLO localization & segmentation (Y-series)
├── inference/                # FracAssist clinical decision support system
│   ├── config.py             # Fixed hyperparameters, weight paths, CUDA auto-detect
│   ├── predict.py            # Selective ensemble logic + GradCAM
│   └── app.py                # Flask: GET /, GET /health, POST /predict
├── index.html                # FracAssist web UI (three tabs: Assist / Health / Config)
├── style.css                 # Dark theme — bone-gradient plates, teal/red accents
├── scripts.js                # UI logic — fetch /predict, overlay toggle, drag-drop
├── xai/                      # XAI pillar implementations (Phase 3)
├── utils/
│   ├── logger.py             # Experiment logging
│   ├── plot.py               # Training curves, metric plots
│   ├── gradcam.py            # GradCAM — compute_overlay / to_base64 / save
│   ├── eval_resnet.py        # Evaluate all ResNet-18 checkpoints on val/test set
│   └── eval_densenet.py      # Evaluate all DenseNet-169 checkpoints on val/test set
├── results/                  # Saved metrics and plots
│   ├── experiments_yolo.csv      # All YOLO experiments — hyperparams + metrics
│   ├── experiments_resnet.csv    # All ResNet-18 experiments — hyperparams + metrics
│   ├── experiments_densenet.csv  # All DenseNet-169 experiments — hyperparams + metrics
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
| `--weights` | path to `.pt` | Weights for standalone evaluation |
| `--split` | `val` / `test` | Eval split — use `test` only for final reporting |

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
# DenseNet-169 (D-series)
python main.py --config configs/densenet_D1.yaml --task D1
python main.py --config configs/densenet_D2.yaml --task D2
```

```bash
# Classification — ResNet-18 (E-series)
python main.py --config configs/resnet_E4.yaml --task E4a_m050
python main.py --config configs/resnet_E4.yaml --task E4a_m075
python main.py --config configs/resnet_E4.yaml --task E4i_d03
python main.py --config configs/resnet_E4.yaml --task E4i_d05
python main.py --config configs/resnet_E4.yaml --task E4e
python main.py --config configs/resnet_E4.yaml --task E4h_g1
python main.py --config configs/resnet_E4.yaml --task E4h_g2

# All E4 experiments sequentially
python main.py --config configs/resnet_E4.yaml --task all
```

```bash
# Individual tasks
python main.py --config configs/yolo_Y0.yaml --task Y0A_localization
python main.py --config configs/yolo_Y0.yaml --task Y0A_segmentation
python main.py --config configs/yolo_Y0.yaml --task Y0B_localization
python main.py --config configs/yolo_Y0.yaml --task Y0B_segmentation
python main.py --config configs/yolo_Y0.yaml --task Y0C_localization
python main.py --config configs/yolo_Y0.yaml --task Y0C_segmentation

# All tasks sequentially
python main.py --config configs/yolo_Y0.yaml --task all

# Y5 — negative sampling ablation
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
```

Both eval scripts perform a post-hoc threshold sweep (0.05–0.95, step 0.025) and rank
checkpoints by F1. Use `--split val` for tuning decisions; `--split test` for final reporting.

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

Post-training, each run performs a val-set threshold sweep (0.05–0.95, step 0.025).
The optimal threshold is saved directly into the checkpoint (`val_threshold` key).

### Config format — DenseNet-169

Same structure as ResNet-18. Only `task` and the model-specific defaults differ:

```yaml
D2:
  experiment_id  : "D2"
  task           : "classify_densenet"   # routes to models/classification/densenet.py
  data_dir       : "data/dataset_cls"
  epochs         : 50
  batch_size     : 32
  img_size       : 224
  device         : "0"
  dropout_p      : 0.3
  weight_mult    : 0.5
  loss           : "weighted_ce"
  scheduler      : "cosine_warmup"
  warmup_epochs  : 3
  lr_backbone    : 1.0e-5
  lr_head        : 1.0e-3
  val_threshold  : 0.5
  plot           : true
```

D1 uses `scheduler: plateau` and flat LR (`lr_backbone: lr_head: 1e-4`) as a clean baseline.
D2 mirrors the ResNet-18 E4e champion config (cosine warmup, differential LR, dropout=0.3).
Both share the same ImageFolder split as E-series (`data/dataset_cls`).

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
E4e (cosine warmup) has lower val F1 (63.6%) but remains documented as the most principled
LR schedule. Test set audit pending — val metrics used above.

Weights: `weights/E4a_m050_best.pth`

### D-series — DenseNet-169

DenseNet-169 (ImageNet pretrained), full fine-tune, Adam. Same ImageFolder split as E-series.

| ID | Scheduler | LR backbone / head | Dropout | Key idea |
|----|-----------|-------------------|---------|----------|
| D1 | Plateau | 1e-4 / 1e-4 (flat) | 0.0 | Clean baseline — matches E4a structure |
| D2 | Cosine warmup (3ep) | 1e-5 / 1e-3 | 0.3 | Mirrors E4e champion config (cosine warmup + differential LR) |

### Results — D-series (val-optimal threshold per experiment, confirmed via eval_densenet.py)

| Split | Experiment | Threshold | F1 | Recall | Precision | Acc | AUC |
|-------|------------|-----------|-----|--------|-----------|-----|-----|
| Val | **D1 ★** | **0.175** | **72.4%** | **72.0%** | **72.8%** | **90.7%** | **0.844** |
| Val | D2 | 0.475 | 64.3% | 56.1% | 75.4% | 89.5% | 0.858 |
| Test | D1 ★ | 0.350 | 68.4% | 65.6% | 71.4% | 88.9% | 0.847 |
| Test | D2 | 0.075 | 58.7% | 63.9% | 54.2% | 83.4% | 0.852 |

**D1 is the approved DenseNet-169 baseline.** Inference threshold: 0.175 (val-sweep optimal).

Key findings:
- D1 beats ResNet-18 E4a by +6.6pp F1 on val (72.4% vs 65.8%) and +5.2pp on test (68.4% vs 63.2%).
- D2's cosine warmup + dropout=0.3 **hurts** DenseNet: DenseNet's dense connections already act as implicit regularisation; additional dropout collapses the threshold and destabilises recall (0.075 on test vs 0.175 on val — over 4× gap). D1's flat LR clean baseline is strictly better on all splits.
- D1 best checkpoint at epoch 13; model then overfits (train loss → 0.01, val loss → 0.5+). TTA hurts D1 (−3.95pp) — inference uses single forward pass.

Weights: `weights/D1_best.pth`

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
#   E4a_m050_best.pth      (optional — enables CLASSIFIER-LED path + GradCAM)
#   D1_best.pth            (optional — enables DenseNet-169 secondary output)

python inference/app.py
# → http://127.0.0.1:5000
```

### Selective Cascade

The system implements a **YOLO-first cascade with ResNet-18 fallback**. YOLO and ResNet
never combine scores — YOLO has first say; ResNet only runs when YOLO fires nothing.
DenseNet-169 D1 runs in parallel on both paths as a secondary classifier.

```
Upload X-ray (JPG / PNG)
        │
        ▼
  YOLO · Y1B · conf ≥ 0.25
        │
   Box detected?
   ┌────┴────┐
  Yes        No
   │          │
YOLO-LED   CLASSIFIER-LED
   │          │
   │     ResNet-18 · E4a
   │     threshold = 0.375
   │          │
   └────┬─────┘
        │                     DenseNet-169 · D1 (parallel, both paths)
        │                     threshold = 0.175
        ▼
   fracture_probability        densenet_probability (secondary)
   label  (Fractured / Non-Fractured)
   xray_with_box  (base64 PNG)
   gradcam_image  (base64 PNG — ResNet layer4[-1])
```

- **YOLO-LED**: YOLO fired a box → fracture probability = YOLO confidence. ResNet and DenseNet run in parallel (if loaded). Clinician can toggle between bounding-box and GradCAM overlays.
- **CLASSIFIER-LED**: YOLO found no box → ResNet-18 classifies the full image (primary decision). DenseNet-169 provides a secondary probability. GradCAM generated on ResNet `layer4[-1]`.
- **YOLO-only mode**: Works without ResNet/DenseNet; GradCAM and secondary classifier unavailable until weights are placed in `weights/`.

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
- Experiment IDs are stable: `E-series` (classification), `Y-series` (YOLO).
- `--debug` flag overrides `epochs=1` for fast pipeline validation without touching configs.

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
