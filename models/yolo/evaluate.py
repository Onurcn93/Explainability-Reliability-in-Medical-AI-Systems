"""
models/yolo/evaluate.py

Post-training evaluation for YOLO models.

Runs validation on the saved best.pt weights and prints a clean metrics
summary. Intended to be called from main.py after training, or standalone
for re-evaluation without retraining.

Usage (standalone):
    python models/yolo/evaluate.py \
        --weights weights/Y0_best.pt \
        --data    data/dataset_yolo/data.yaml \
        --task    detect \
        --imgsz   600

Metrics reported
----------------
Localization (detect):  Box P, Box R, mAP@0.5, mAP@0.5:0.95
Segmentation (segment): Box P, Box R, Box mAP@0.5 + Mask P, Mask R, Mask mAP@0.5
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

SEP  = "=" * 72
LINE = "─" * 72


def run_evaluation(
    weights: str,
    data:    str,
    task:    str,
    imgsz:   int  = 600,
    device:  str  = "0",
    split:   str  = "val",
) -> dict:
    """Run YOLO validation and return a metrics dict.

    Args:
        weights: Path to best.pt.
        data:    Path to data.yaml.
        task:    "detect" or "segment".
        imgsz:   Inference image size.
        device:  CUDA device index or "cpu".
        split:   Dataset split to evaluate on — "val" during development,
                 "test" only for final reporting.

    Returns:
        Dict of metric_name -> float value.
    """
    model   = YOLO(weights)
    results = model.val(data=data, imgsz=imgsz, device=device, split=split)

    metrics = {}

    if task == "detect":
        metrics["Box P"]       = float(results.box.mp)
        metrics["Box R"]       = float(results.box.mr)
        metrics["mAP@0.5"]     = float(results.box.map50)
        metrics["mAP@0.5:0.95"]= float(results.box.map)

    elif task == "segment":
        metrics["Box P"]       = float(results.box.mp)
        metrics["Box R"]       = float(results.box.mr)
        metrics["Box mAP@0.5"] = float(results.box.map50)
        metrics["Mask P"]      = float(results.seg.mp)
        metrics["Mask R"]      = float(results.seg.mr)
        metrics["Mask mAP@0.5"]= float(results.seg.map50)

    _print_metrics(weights, task, split, metrics)
    return metrics


def _print_metrics(weights: str, task: str, split: str, metrics: dict) -> None:
    print()
    print(SEP)
    print(f"  Evaluation — {task} | split={split}")
    print(f"  Weights    : {weights}")
    print(LINE)
    for name, value in metrics.items():
        print(f"  {name:<20}: {value:.4f}")
    print(SEP)
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data",    type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--task",    type=str, required=True, choices=["detect", "segment"])
    parser.add_argument("--imgsz",   type=int, default=600)
    parser.add_argument("--device",  type=str, default="0")
    parser.add_argument("--split",   type=str, default="val", choices=["val", "test"],
                        help="Use 'val' during development. 'test' only for final reporting.")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        print(f"[error] Weights not found: {args.weights}")
        return

    run_evaluation(
        weights = args.weights,
        data    = args.data,
        task    = args.task,
        imgsz   = args.imgsz,
        device  = args.device,
        split   = args.split,
    )


if __name__ == "__main__":
    main()