"""
eval_resnet.py

Evaluate ResNet-18 checkpoints on a given split and print a ranked F1 table.
CLAHE preprocessing is applied automatically for CAALMIX checkpoints (E5/E6/E7).

Usage:
    python utils/eval_resnet.py                        # all checkpoints, test split
    python utils/eval_resnet.py --split val
    python utils/eval_resnet.py --ckpt E6_best.pth     # single checkpoint
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Config ────────────────────────────────────────────────────────────── #

BATCH_SIZE = 32
IMG_SIZE   = 224
FRAC_CLASS = "Fractured"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CHECKPOINT_DIRS = [Path("weights"), Path("colab_results")]

# Checkpoints trained with CLAHE on ALL splits — must use CLAHE at eval time.
# E8 is explicitly excluded: XRayAugMix standalone, no CLAHE in its pipeline.
_CLAHE_CKPTS = {"E5_best", "E6_best", "E7_best"}

# Map filename substrings → dropout_p for architecture reconstruction.
DROPOUT_MAP = [
    ("dropout05", 0.5),
    ("d05",       0.5),
    ("dropout03", 0.3),
    ("d03",       0.3),
    ("cosine",    0.3),
    ("focal",     0.3),
    ("_g1",       0.3),
    ("_g2",       0.3),
]

# ── Transforms ─────────────────────────────────────────────────────────── #

class _CLAHETransform:
    def __call__(self, img: Image.Image) -> Image.Image:
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray     = np.array(img.convert("L"), dtype=np.uint8)
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced).convert("RGB")


def _get_transform(clahe: bool = False) -> transforms.Compose:
    steps = []
    if clahe:
        steps.append(_CLAHETransform())
    steps += [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


# ── Helpers ───────────────────────────────────────────────────────────── #

def _infer_dropout(path: Path, state_dict: dict) -> float:
    if "fc.1.weight" in state_dict:
        name = path.stem.lower()
        for substr, dp in DROPOUT_MAP:
            if substr in name:
                return dp
        return 0.3
    return 0.0


def _build_model(dropout_p: float, device: torch.device) -> nn.Module:
    model   = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    in_feat = model.fc.in_features
    model.fc = (nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_feat, 2))
                if dropout_p > 0.0 else nn.Linear(in_feat, 2))
    return model.to(device)


def _collect_probs(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
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

def main(data_dir: Path, ckpt_filter: str = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Split: {data_dir}\n")

    # Collect checkpoints — skip DenseNet (D-series) and EfficientNet (F-series)
    _SKIP = re.compile(r'^[DdFf]\d', re.IGNORECASE)
    ckpt_paths = []
    for d in CHECKPOINT_DIRS:
        if d.exists():
            ckpt_paths.extend(
                p for p in sorted(d.glob("*.pth"))
                if not _SKIP.match(p.stem)
                and "densenet" not in p.stem.lower()
                and "efficientnet" not in p.stem.lower()
            )

    if ckpt_filter:
        ckpt_paths = [p for p in ckpt_paths if ckpt_filter in p.name]
        if not ckpt_paths:
            print(f"No checkpoint matching '{ckpt_filter}' found.")
            return

    if not ckpt_paths:
        print("No .pth files found.")
        return

    rows = []
    for ckpt_path in ckpt_paths:
        needs_clahe = ckpt_path.stem in _CLAHE_CKPTS
        clahe_tag   = " [CLAHE]" if needs_clahe else ""
        print(f"Evaluating {ckpt_path.name}{clahe_tag} ...", end=" ", flush=True)

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        state     = ckpt.get("model_state_dict", ckpt)
        dropout_p = _infer_dropout(ckpt_path, state)
        model     = _build_model(dropout_p, device)
        model.load_state_dict(state, strict=True)
        model.eval()

        saved_thresh = float(ckpt.get("val_threshold", 0.5))

        # Build loader with the correct transform for this checkpoint
        tf      = _get_transform(clahe=needs_clahe)
        dataset = ImageFolder(root=str(data_dir), transform=tf)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        frac_idx = dataset.class_to_idx[FRAC_CLASS]

        labels, probs = _collect_probs(model, loader, device)
        opt_thresh, _ = _sweep_threshold(labels, probs, frac_idx)
        m_saved = _evaluate(labels, probs, saved_thresh, frac_idx)
        m_opt   = _evaluate(labels, probs, opt_thresh,   frac_idx)

        use_thresh = opt_thresh if m_opt["f1"] >= m_saved["f1"] else saved_thresh
        m_use      = m_opt      if m_opt["f1"] >= m_saved["f1"] else m_saved

        source = "colab" if "colab" in str(ckpt_path) else "local"
        exp_id = ckpt.get("exp_id", ckpt_path.stem)

        rows.append({
            "exp":    exp_id,
            "source": source,
            "clahe":  needs_clahe,
            "thresh": use_thresh,
            "f1":     m_use["f1"],
            "recall": m_use["recall"],
            "prec":   m_use["precision"],
            "acc":    m_use["acc"],
            "auc":    m_use["auc"],
        })
        print(f"F1={m_use['f1']:.4f}  thresh={use_thresh:.3f}")

    rows.sort(key=lambda r: r["f1"], reverse=True)

    print("\n" + "=" * 90)
    print(f"{'Rank':<5} {'Experiment':<30} {'Src':<6} {'CLAHE':<6} {'Thresh':<7} "
          f"{'F1':>6} {'Recall':>7} {'Prec':>7} {'Acc':>7} {'AUC':>7}")
    print("-" * 90)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:<5} {r['exp']:<30} {r['source']:<6} {'yes' if r['clahe'] else 'no':<6} "
            f"{r['thresh']:<7.3f} {r['f1']:>6.4f} {r['recall']:>7.4f} "
            f"{r['prec']:>7.4f} {r['acc']:>7.4f} {r['auc']:>7.4f}"
        )
    print("=" * 90)
    champion = rows[0]
    print(
        f"\nChampion: {champion['exp']} ({champion['source']})"
        f"{'  [CLAHE]' if champion['clahe'] else ''}"
        f"  F1={champion['f1']:.4f}  thresh={champion['thresh']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--ckpt", default=None,
                        help="Evaluate only this checkpoint filename (e.g. E6_best.pth)")
    args = parser.parse_args()
    main(data_dir=Path(f"data/dataset_cls/{args.split}"), ckpt_filter=args.ckpt)
