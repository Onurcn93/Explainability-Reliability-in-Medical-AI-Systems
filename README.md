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
| 1 | ResNet-18 | Binary fracture classification | F1 (fractured class) | Complete |
| 2 | YOLOv8s / YOLOv8s-seg / YOLOv8m | Localization & segmentation | mAP@0.5 | Complete |
| 3 | CBM + Prototypes + Counterfactuals | XAI — three-pillar architecture | Task-specific | Pending |

Phase 3 is the core thesis contribution: integrating clinically-grounded attribute
explanations (Pillar 1), precedent-based example retrieval (Pillar 2), and contrastive
counterfactual explanations (Pillar 3) in a single system for fracture detection.

---

## Repository Structure

```
/
├── configs/                  # Experiment config files (YAML)
│   ├── yolo_baseline.yaml    # Original Y0 runs (fractured-only, mixed optimizers)
│   ├── yolo_Y0.yaml          # Three-way reproduction: Y0A / Y0B / Y0C
│   ├── yolo_Y1.yaml          # Extended training: Y1A (patience=10) / Y1B (patience=50)
│   ├── yolo_Y2.yaml          # Resolution ablation: imgsz=640 (COCO-standard)
│   ├── yolo_Y3.yaml          # Resolution ablation: imgsz=800, batch=8 (VRAM limit)
│   └── yolo_Y4.yaml          # Capacity ablation: YOLOv8m (25.9M params)
├── data/                     # Data preparation scripts
├── models/
│   ├── classification/       # ResNet-18 experiments (E-series)
│   └── yolo/                 # YOLO localization & segmentation (Y-series)
├── xai/                      # XAI pillar implementations (Phase 3)
├── utils/
│   ├── logger.py             # Experiment logging
│   └── plot.py               # Training curves, metric plots
├── results/                  # Saved metrics and plots (gitignored)
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
| ResNet-18 | Binary classification | 1 |
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
# Y0A — paper stated splits (574 fractured only)
python data/prepare_yolo.py --out_dir data/dataset_yolo_Y0A --clean
python data/prepare_yolo.py --seg --out_dir data/dataset_yolo_seg_Y0A --clean

# Y0B — author notebook splits (574 + 61 test leak = 635)
python data/prepare_yolo.py --include_test --out_dir data/dataset_yolo_Y0B --clean
python data/prepare_yolo.py --seg --include_test --out_dir data/dataset_yolo_seg_Y0B --clean

# Y0C — full negative sampling (574 fractured + 3,366 non-fractured = 3,940)
python data/prepare_yolo.py --n_neg -1 --out_dir data/dataset_yolo_Y0C --clean
python data/prepare_yolo.py --seg --n_neg -1 --out_dir data/dataset_yolo_seg_Y0C --clean
```

Key flags for `prepare_yolo.py`:

| Flag | Description |
|------|-------------|
| `--seg` | Build segmentation dataset (COCO polygon labels) |
| `--n_neg -1` | Include all 3,366 non-fractured images as train negatives |
| `--n_neg N` | Include N randomly sampled non-fractured images (seed=42) |
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
# Individual tasks
python main.py --config configs/yolo_Y0.yaml --task Y0A_localization
python main.py --config configs/yolo_Y0.yaml --task Y0A_segmentation
python main.py --config configs/yolo_Y0.yaml --task Y0B_localization
python main.py --config configs/yolo_Y0.yaml --task Y0B_segmentation
python main.py --config configs/yolo_Y0.yaml --task Y0C_localization
python main.py --config configs/yolo_Y0.yaml --task Y0C_segmentation

# All tasks sequentially
python main.py --config configs/yolo_Y0.yaml --task all
```

### 4. Evaluate

```bash
python models/yolo/evaluate.py \
    --weights weights/Y0A_detect_best.pt \
    --data    data/dataset_yolo_Y0A/data.yaml \
    --task    detect \
    --imgsz   600
```

Use `--split val` during development (default). Use `--split test` only for
final per-phase reporting.

### Config format

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

## Phase 1 — Classification Results (Complete)

| Model | Threshold | F1 (Fractured) | Recall | Accuracy | AUC |
|-------|-----------|----------------|--------|----------|-----|
| ResNet-18 (E4e, cosine warmup) | 0.425 | 65.81% | 64.66% | 88.75% | 0.8884 |

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