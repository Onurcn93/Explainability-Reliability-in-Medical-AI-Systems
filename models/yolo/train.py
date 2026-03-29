"""
models/yolo/train.py

YOLO training wrapper. Runs Ultralytics model.train(), then parses the
results.csv saved by YOLO to feed metrics into our Logger and plot utilities.

Supports both localization (task=detect) and segmentation (task=segment).
One variable changed at a time — experiment config drives everything.

Results CSV column mapping
--------------------------
Detect:
    BoxLoss  ← train/box_loss
    mAP@0.5  ← metrics/mAP50(B)
    P        ← metrics/precision(B)
    R        ← metrics/recall(B)

Segment (additional):
    MaskLoss ← train/seg_loss
    MaskMAP  ← metrics/mAP50(M)
"""

import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from ultralytics import YOLO

from utils.logger import Logger
from utils.plot import plot_training_curves

# ------------------------------------------------------------------ #
# CSV column definitions per task
# ------------------------------------------------------------------ #

DETECT_COL_MAP = {
    "BoxLoss": "train/box_loss",
    "mAP@0.5": "metrics/mAP50(B)",
    "P":        "metrics/precision(B)",
    "R":        "metrics/recall(B)",
}

SEGMENT_COL_MAP = {
    "BoxLoss":  "train/box_loss",
    "MaskLoss": "train/seg_loss",
    "mAP@0.5":  "metrics/mAP50(B)",
    "MaskMAP":  "metrics/mAP50(M)",
    "P":         "metrics/precision(B)",
    "R":         "metrics/recall(B)",
}

DETECT_PLOT = dict(
    groups  = [["BoxLoss"], ["mAP@0.5", "P", "R"]],
    titles  = ["Training Loss", "Validation Metrics"],
    ylabels = ["Loss", "Value"],
)

SEGMENT_PLOT = dict(
    groups  = [["BoxLoss", "MaskLoss"], ["mAP@0.5", "MaskMAP", "P", "R"]],
    titles  = ["Training Losses", "Validation Metrics"],
    ylabels = ["Loss", "Value"],
)


# ------------------------------------------------------------------ #

def _parse_results_csv(csv_path: Path, col_map: Dict[str, str]) -> Dict[str, List[float]]:
    """Parse Ultralytics results.csv into a dict of metric_name -> per-epoch values."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    series = {}
    for metric_name, col_name in col_map.items():
        if col_name in df.columns:
            series[metric_name] = df[col_name].tolist()
        else:
            print(f"[warn] Column '{col_name}' not found in results.csv — skipping {metric_name}")
    return series


def run_training(config: dict) -> Path:
    """Run YOLO training from a config dict.

    Args:
        config: Experiment config loaded from YAML. Expected keys:
            experiment_id  — e.g. "Y0"
            task           — "detect" or "segment"
            model_weights  — e.g. "yolov8s.pt" or "yolov8s-seg.pt"
            data_yaml      — path to data.yaml
            epochs         — int
            imgsz          — int
            device         — e.g. "0" (GPU index) or "cpu"
            plot           — bool

    Returns:
        Path to best.pt copied into weights/.
    """
    exp_id   = config["experiment_id"]
    task     = config["task"]
    is_seg   = (task == "segment")
    col_map  = SEGMENT_COL_MAP if is_seg else DETECT_COL_MAP
    plot_cfg = SEGMENT_PLOT    if is_seg else DETECT_PLOT
    metrics  = list(col_map.keys())

    experiment = (
        f"{exp_id} | {'YOLOv8s-seg' if is_seg else 'YOLOv8s'} | {task}"
        f" | epochs={config['epochs']} | imgsz={config['imgsz']}"
    )

    logger = Logger(
        experiment     = experiment,
        metrics        = metrics,
        primary_metric = "mAP@0.5",
    )

    logger.log_start({
        "Task"   : task,
        "Data"   : config["data_yaml"],
        "Epochs" : f"{config['epochs']}  |  imgsz: {config['imgsz']}  |  Device: {config['device']}",
        "Weights": config["model_weights"],
    })

    # ── Train ────────────────────────────────────────────────────── #
    model = YOLO(config["model_weights"])
    model.train(
        task      = task,
        data      = config["data_yaml"],
        epochs    = config["epochs"],
        imgsz     = config["imgsz"],
        optimizer = config.get("optimizer", "auto"),
        lr0       = config.get("lr0", 0.01),
        momentum  = config.get("momentum", 0.937),
        device    = config["device"],
        seed      = config.get("seed", 42),
        name      = exp_id,
        exist_ok  = True,
    )

    # ── Parse results.csv ────────────────────────────────────────── #
    results_csv = Path("runs") / task / exp_id / "results.csv"
    if not results_csv.exists():
        logger.log_message(f"[warn] results.csv not found at {results_csv} — skipping post-training log")
        logger.close()
        return Path()

    series = _parse_results_csv(results_csv, col_map)

    # Replay epoch rows through our logger
    n_epochs = len(next(iter(series.values())))
    best_map  = 0.0
    best_epoch = 0

    for i in range(n_epochs):
        epoch_metrics = {m: series[m][i] for m in metrics if m in series}
        logger.log_epoch(i + 1, epoch_metrics)

        current_map = epoch_metrics.get("mAP@0.5", 0.0)
        if current_map > best_map:
            best_map   = current_map
            best_epoch = i + 1

    # ── Copy best weights ────────────────────────────────────────── #
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    src_weights = Path("runs") / task / exp_id / "weights" / "best.pt"
    dst_weights = weights_dir / f"{exp_id}_best.pt"

    if src_weights.exists():
        shutil.copy2(src_weights, dst_weights)
    else:
        logger.log_message(f"[warn] best.pt not found at {src_weights}")

    logger.log_best(best_map, str(dst_weights))
    logger.log_complete(best_map, str(dst_weights))
    logger.close()

    # ── Plot ─────────────────────────────────────────────────────── #
    if config.get("plot", True):
        plot_training_curves(
            series     = series,
            experiment = experiment,
            **plot_cfg,
        )

    return dst_weights