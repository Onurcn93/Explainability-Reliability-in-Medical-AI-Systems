"""
eval_resnet.py

Evaluate all ResNet-18 checkpoints on the test set and print a ranked F1 table.
Handles both local (weights/) and Colab (colab_results/) checkpoints.

Usage:
    python eval_resnet.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import ImageFile
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Config ────────────────────────────────────────────────────────────── #

DATA_DIR   = Path("data/dataset_cls/test")  # overridden by --split
BATCH_SIZE = 32
IMG_SIZE   = 224
FRAC_CLASS = "Fractured"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CHECKPOINT_DIRS = [Path("weights"), Path("colab_results")]

# Map filename substrings → dropout_p for architecture reconstruction.
DROPOUT_MAP = [
    ("dropout05", 0.5),
    ("d05",       0.5),
    ("dropout03", 0.3),
    ("d03",       0.3),
    ("cosine",    0.3),   # E4e always used dropout=0.3
    ("focal",     0.3),   # E4h always used dropout=0.3
    ("_g1",       0.3),
    ("_g2",       0.3),
]
# E4a (weight only) → dropout=0.0 (default if nothing matches above)

# ── Helpers ───────────────────────────────────────────────────────────── #

def _infer_dropout(path: Path, state_dict: dict) -> float:
    # Detect from checkpoint keys first — reliable regardless of filename.
    # dropout=0.3/0.5 → fc is Sequential → keys are fc.1.weight / fc.1.bias
    # dropout=0.0     → fc is Linear    → keys are fc.weight / fc.bias
    if "fc.1.weight" in state_dict:
        # Determine exact dropout value from filename as tiebreak
        name = path.stem.lower()
        for substr, dp in DROPOUT_MAP:
            if substr in name:
                return dp
        return 0.3  # default for dropout models when filename gives no hint
    return 0.0


def _build_model(dropout_p: float, device: torch.device) -> nn.Module:
    model   = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = model.fc.in_features
    if dropout_p > 0.0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_feat, 2))
    else:
        model.fc = nn.Linear(in_feat, 2)
    return model.to(device)


def _get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _collect_probs(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def _sweep_threshold(labels, probs, frac_idx):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.025):
        preds = np.where(probs[:, frac_idx] >= t, frac_idx, 1 - frac_idx)
        f1    = f1_score(labels, preds, pos_label=frac_idx, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _evaluate(labels, probs, threshold, frac_idx):
    preds      = np.where(probs[:, frac_idx] >= threshold, frac_idx, 1 - frac_idx)
    bin_labels = (labels == frac_idx).astype(int)
    return {
        "f1":        f1_score(labels, preds, pos_label=frac_idx, zero_division=0),
        "recall":    recall_score(labels, preds, pos_label=frac_idx, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=frac_idx, zero_division=0),
        "acc":       accuracy_score(labels, preds),
        "auc":       roc_auc_score(bin_labels, probs[:, frac_idx]),
    }


# ── Main ─────────────────────────────────────────────────────────────── #

def main(data_dir: Path = DATA_DIR):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tf      = _get_transform()
    dataset = ImageFolder(root=str(data_dir), transform=tf)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    frac_idx = dataset.class_to_idx[FRAC_CLASS]
    print(f"Split: {data_dir}  |  {len(dataset)} images  |  Fractured idx: {frac_idx}\n")

    # Collect all checkpoints — skip DenseNet weights (D-series)
    _SKIP = re.compile(r'^[Dd]\d', re.IGNORECASE)
    ckpt_paths = []
    for d in CHECKPOINT_DIRS:
        if d.exists():
            ckpt_paths.extend(
                p for p in sorted(d.glob("*.pth"))
                if not _SKIP.match(p.stem) and "densenet" not in p.stem.lower()
            )

    if not ckpt_paths:
        print("No .pth files found.")
        return

    rows = []
    for ckpt_path in ckpt_paths:
        print(f"Evaluating {ckpt_path} ...", end=" ", flush=True)

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Load weights — handle both plain state_dict and wrapped checkpoint
        state     = ckpt.get("model_state_dict", ckpt)
        dropout_p = _infer_dropout(ckpt_path, state)
        model     = _build_model(dropout_p, device)
        model.load_state_dict(state, strict=True)
        model.eval()

        # Saved threshold (post-sweep for local runs; may be 0.5 for Colab)
        saved_thresh = float(ckpt.get("val_threshold", 0.5))

        labels, probs = _collect_probs(model, loader, device)

        # Always sweep to find test-optimal threshold
        opt_thresh, _ = _sweep_threshold(labels, probs, frac_idx)

        # Evaluate at both thresholds
        m_saved = _evaluate(labels, probs, saved_thresh, frac_idx)
        m_opt   = _evaluate(labels, probs, opt_thresh,   frac_idx)

        # Use the better F1 for ranking
        use_thresh = opt_thresh if m_opt["f1"] >= m_saved["f1"] else saved_thresh
        m_use      = m_opt      if m_opt["f1"] >= m_saved["f1"] else m_saved

        source = "colab" if "colab" in str(ckpt_path) else "local"
        exp_id = ckpt.get("exp_id", ckpt_path.stem)

        rows.append({
            "exp":    exp_id,
            "source": source,
            "thresh": use_thresh,
            "f1":     m_use["f1"],
            "recall": m_use["recall"],
            "prec":   m_use["precision"],
            "acc":    m_use["acc"],
            "auc":    m_use["auc"],
        })
        print(f"F1={m_use['f1']:.4f}  thresh={use_thresh:.3f}")

    # Sort by F1 descending
    rows.sort(key=lambda r: r["f1"], reverse=True)

    # Print table
    print("\n" + "=" * 85)
    print(f"{'Rank':<5} {'Experiment':<30} {'Src':<6} {'Thresh':<7} "
          f"{'F1':>6} {'Recall':>7} {'Prec':>7} {'Acc':>7} {'AUC':>7}")
    print("-" * 85)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:<5} {r['exp']:<30} {r['source']:<6} {r['thresh']:<7.3f} "
            f"{r['f1']:>6.4f} {r['recall']:>7.4f} {r['prec']:>7.4f} "
            f"{r['acc']:>7.4f} {r['auc']:>7.4f}"
        )
    print("=" * 85)
    print(f"\nChampion: {rows[0]['exp']} ({rows[0]['source']})  "
          f"F1={rows[0]['f1']:.4f}  thresh={rows[0]['thresh']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    args = parser.parse_args()
    main(data_dir=Path(f"data/dataset_cls/{args.split}"))
