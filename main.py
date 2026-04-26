"""
main.py

Entry point for all experiments.

Usage:
    # YOLO baseline reproduction
    python main.py --config configs/yolo_baseline.yaml --task localization
    python main.py --config configs/yolo_baseline.yaml --task segmentation

    # Both in one go
    python main.py --config configs/yolo_baseline.yaml --task all
"""

import argparse
import random
import sys
from pathlib import Path

import yaml


def load_config(config_path: str, task: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    if task not in full:
        print(f"[error] Task '{task}' not found in {config_path}")
        print(f"        Available: {list(full.keys())}")
        sys.exit(1)
    return full[task]


def run_yolo(config: dict) -> None:
    from models.yolo.train import run_training
    run_training(config)


def run_resnet(config: dict) -> None:
    from models.classification.resnet import run_training
    run_training(config)


def run_densenet(config: dict) -> None:
    from models.classification.densenet import run_training
    run_training(config)


def set_global_seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[main] Global seed set to {seed}")


def main():
    parser = argparse.ArgumentParser(description="Explainability & Reliability in Medical AI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML (e.g. configs/yolo_baseline.yaml)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task key inside the config file, or 'all' to run every task",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for all runs (default: 42)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: override epochs=1 for all tasks (tests pipeline without full training)",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] Config not found: {config_path}")
        sys.exit(1)

    if args.debug:
        print("[main] DEBUG mode: epochs overridden to 1 for all tasks")

    # Resolve tasks to run
    if args.task == "all":
        with open(config_path, encoding="utf-8") as f:
            full = yaml.safe_load(f)
        tasks = list(full.keys())
    else:
        tasks = [args.task]

    for task in tasks:
        print(f"\n{'='*72}")
        print(f"  Task: {task}")
        print(f"{'='*72}\n")

        config = load_config(str(config_path), task)

        if args.no_plot:
            config["plot"] = False
        if args.debug:
            config["epochs"] = 1
        config["seed"] = args.seed

        # Route to the right training module based on config
        yolo_tasks = {"detect", "segment", "localization", "segmentation"}
        if config.get("task") in yolo_tasks or config.get("model_weights", "").startswith("yolo"):
            run_yolo(config)
        elif config.get("task") == "classify":
            run_resnet(config)
        elif config.get("task") == "classify_densenet":
            run_densenet(config)
        else:
            print(f"[error] No training module found for config task '{config.get('task')}'")
            sys.exit(1)


if __name__ == "__main__":
    main()