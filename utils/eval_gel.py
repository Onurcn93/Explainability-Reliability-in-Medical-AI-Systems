"""
eval_gel.py

Evaluate the Gated Ensemble Logic (GEL) pipeline on val and/or test splits.

GEL pipeline: RC init -> Asymmetric OAM -> PDWF -> P_final -> BVG gate.
Supports 2-model (ResNet + DenseNet) or 3-model (+ EfficientNet-B3) automatically.
YOLO bbox authentication is UI-only; it does not affect p_final and is
therefore not included in classification metrics here.

Usage:
    python utils/eval_gel.py                  # val sweep -> apply to test (default)
    python utils/eval_gel.py --split val
    python utils/eval_gel.py --split test
"""

import argparse
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

# ── Constants ──────────────────────────────────────────────────────────── #

BATCH_SIZE    = 32
IMG_SIZE      = 224
FRAC_CLASS    = "Fractured"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RESNET_WEIGHTS       = Path("weights/E6_best.pth")
DENSENET_WEIGHTS     = Path("weights/D1_best.pth")
EFFICIENTNET_WEIGHTS = Path("weights/F1_best.pth")

# GEL hyperparameters — must match inference/config.py GEL_CONFIG exactly
GEL_F1_RESNET          = 0.689   # E6 val F1 anchor (CAALMIX champion)
GEL_F1_DENSENET        = 0.724   # D1 val F1 anchor
GEL_F1_EFFICIENTNET    = 0.671   # F1 val F1 anchor (confirmed 2026-04-28)
GEL_TAU                = 0.35    # BVG gate threshold
GEL_DISAGREE_LIM       = 0.40    # OAM outlier disagreement limit
GEL_PENALTY_K_LOW      = 0.10    # Direction-aware OAM — LOW outlier (aggressive: lone no-frac dissenter)
GEL_PENALTY_K_HIGH     = 0.30    # Direction-aware OAM — HIGH outlier (lenient: lone fracture signal)
GEL_PENALTY_K_STANDARD = 0.20    # Symmetric balanced reference (not used in inference)

RESNET_THRESHOLD       = 0.525   # E6 val-optimal (individual comparison)
DENSENET_THRESHOLD     = 0.175   # D1 val-optimal (individual comparison)
EFFICIENTNET_THRESHOLD = 0.525   # F1 val-optimal (individual comparison)

# ── Model loading ─────────────────────────────────────────────────────── #

def _load_resnet(device):
    if not RESNET_WEIGHTS.exists():
        raise FileNotFoundError(f"ResNet-18 weights not found: {RESNET_WEIGHTS}")
    ckpt     = _safe_load(RESNET_WEIGHTS, device)
    state    = ckpt.get("model_state_dict", ckpt)
    frac_idx = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    has_dropout = "fc.1.weight" in state
    m = tv_models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2)) if has_dropout else nn.Linear(512, 2)
    m.load_state_dict(state)
    m.eval()
    return m.to(device), frac_idx


def _load_densenet(device):
    if not DENSENET_WEIGHTS.exists():
        raise FileNotFoundError(f"DenseNet-169 weights not found: {DENSENET_WEIGHTS}")
    ckpt     = _safe_load(DENSENET_WEIGHTS, device)
    state    = ckpt.get("model_state_dict", ckpt)
    frac_idx = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    has_dropout = "classifier.1.weight" in state
    m = tv_models.densenet169(weights=None)
    in_feat = m.classifier.in_features  # 1664
    m.classifier = (
        nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 2))
        if has_dropout else nn.Linear(in_feat, 2)
    )
    m.load_state_dict(state)
    m.eval()
    return m.to(device), frac_idx


def _load_efficientnet(device):
    if not EFFICIENTNET_WEIGHTS.exists():
        print(f"[INFO] EfficientNet-B3 weights not found — running 2-model GEL.")
        return None, None
    ckpt     = _safe_load(EFFICIENTNET_WEIGHTS, device)
    state    = ckpt.get("model_state_dict", ckpt)
    frac_idx = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    has_dropout = "classifier.1.weight" in state
    m = tv_models.efficientnet_b3(weights=None)
    in_feat = m.classifier[1].in_features  # 1536
    m.classifier = (
        nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 2))
        if has_dropout else nn.Linear(in_feat, 2)
    )
    m.load_state_dict(state)
    m.eval()
    return m.to(device), frac_idx


def _safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


# ── Preprocessing ─────────────────────────────────────────────────────── #

class _CLAHETransform:
    """CLAHE preprocessing for ResNet-18 E6 (clip=2.0, tile=8×8 — training default)."""
    def __call__(self, img: Image.Image) -> Image.Image:
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray     = np.array(img.convert("L"), dtype=np.uint8)
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced).convert("RGB")


def _get_transform(clahe: bool = False):
    steps = ([_CLAHETransform()] if clahe else []) + [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


# ── Inference ─────────────────────────────────────────────────────────── #

def _collect_single(model, frac_idx, loader, device):
    """Collect fracture probabilities for one model over a full DataLoader."""
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            probs.extend(torch.softmax(model(imgs.to(device)), dim=1)[:, frac_idx].cpu().numpy())
    return np.array(probs)


def _collect_labels(loader):
    """Collect ground-truth labels in DataLoader order (shuffle=False assumed)."""
    labels = []
    for _, lbl in loader:
        labels.extend(lbl.numpy())
    return np.array(labels)


# ── GEL — vectorized ─────────────────────────────────────────────────── #

def _apply_gel(p_r, p_d, p_e=None):
    """Vectorized GEL: RC init -> Asymmetric OAM -> PDWF -> P_final -> BVG gate.

    Asymmetric OAM:
      HIGH outlier (p_i > mu, lone fracture signal)   -> k_high=0.30 lenient  — preserve fracture signal
      LOW  outlier (p_i < mu, lone no-frac dissenter) -> k_low =0.10 aggressive — protect fracture consensus
    Both directions protect against missed fractures from opposite sides.

    Returns (p_final, gate_passed) — both shape (N,).
    """
    use_e = p_e is not None

    if use_e:
        total_f1 = GEL_F1_RESNET + GEL_F1_DENSENET + GEL_F1_EFFICIENTNET
        probs = [p_r, p_d, p_e]
        f1s   = [GEL_F1_RESNET, GEL_F1_DENSENET, GEL_F1_EFFICIENTNET]
    else:
        total_f1 = GEL_F1_RESNET + GEL_F1_DENSENET
        probs = [p_r, p_d]
        f1s   = [GEL_F1_RESNET, GEL_F1_DENSENET]

    n = len(probs)

    # Step 2 — RC initialisation
    rcs = [np.full_like(p, f1 / total_f1) for p, f1 in zip(probs, f1s)]

    # Step 3 — Asymmetric OAM
    mu = sum(probs) / n
    new_rcs = []
    for p, rc in zip(probs, rcs):
        outlier = np.abs(p - mu) > GEL_DISAGREE_LIM
        k = np.where(p < mu, GEL_PENALTY_K_LOW, GEL_PENALTY_K_HIGH)
        new_rcs.append(np.where(outlier, rc * k, rc))
    rcs = new_rcs

    # Step 4 — PDWF
    total_rc = sum(rcs)
    p_final  = sum(p * rc for p, rc in zip(probs, rcs)) / total_rc

    # Step 5 — BVG gate using P_final (post-OAM)
    gate_passed = p_final >= GEL_TAU

    return p_final, gate_passed


# ── Threshold helpers ─────────────────────────────────────────────────── #

def _sweep_threshold(labels, scores, frac_idx):
    binary = (labels == frac_idx).astype(int)
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.025):
        preds = (scores >= t).astype(int)
        f1    = f1_score(binary, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def _evaluate(labels, scores, threshold, frac_idx):
    preds  = (scores >= threshold).astype(int)
    binary = (labels == frac_idx).astype(int)
    return {
        "f1":        f1_score(binary,        preds, pos_label=1, zero_division=0),
        "recall":    recall_score(binary,    preds, pos_label=1, zero_division=0),
        "precision": precision_score(binary, preds, pos_label=1, zero_division=0),
        "acc":       accuracy_score(binary,  preds),
        "auc":       roc_auc_score(binary,   scores),
    }


# ── Reporting helpers ─────────────────────────────────────────────────── #

def _print_metrics_row(label, m):
    print(
        f"  {label:<32}  "
        f"F1={m['f1']:.4f}  Recall={m['recall']:.4f}  "
        f"Prec={m['precision']:.4f}  Acc={m['acc']:.4f}  AUC={m['auc']:.4f}"
    )


def _gel_diagnostics(probs, names, gate_passed):
    n  = len(probs)
    mu = sum(probs) / n
    for p, name in zip(probs, names):
        trigger_pct = (np.abs(p - mu) > GEL_DISAGREE_LIM).mean() * 100
        print(f"  OAM {name:<22} trigger: {trigger_pct:.1f}%  (delta={GEL_DISAGREE_LIM})")
    print(f"  BVG gate pass rate (P_final>=tau): {gate_passed.mean()*100:.1f}%  (tau={GEL_TAU})")


# ── Per-split evaluation ───────────────────────────────────────────────── #

def eval_split(resnet, densenet, r_fi, d_fi, data_dir, device, label, val_thresh=None,
               efficientnet=None, e_fi=None):
    """
    Run GEL eval on one split.
    val_thresh — if provided, apply as fixed threshold (test mode);
                 if None, sweep and report both 0.5 and optimal (val mode).
    Returns opt_thresh found on this split.
    """
    # ResNet E6 requires CLAHE; DenseNet D1 and EfficientNet F1 use standard pipeline.
    # Two separate DataLoaders — shuffle=False preserves order so indices match.
    ds_clahe = ImageFolder(root=str(data_dir), transform=_get_transform(clahe=True))
    ds_std   = ImageFolder(root=str(data_dir), transform=_get_transform(clahe=False))
    loader_clahe = DataLoader(ds_clahe, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    loader_std   = DataLoader(ds_std,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    frac_idx = ds_std.class_to_idx[FRAC_CLASS]

    use_e    = efficientnet is not None
    n_models = 3 if use_e else 2
    print(f"\n{'=' * 80}")
    print(f"  Split: {label}  ({data_dir})  —  {len(ds_std)} images  |  frac_idx={frac_idx}  |  {n_models}-model GEL")
    print(f"{'=' * 80}")

    labels = _collect_labels(loader_std)
    p_r    = _collect_single(resnet,  r_fi, loader_clahe, device)
    p_d    = _collect_single(densenet, d_fi, loader_std,  device)
    p_e    = _collect_single(efficientnet, e_fi, loader_std, device) if use_e else None
    p_final, gate_passed = _apply_gel(p_r, p_d, p_e)

    probs = [p_r, p_d] + ([p_e] if use_e else [])
    names = ["ResNet-18", "DenseNet-169"] + (["EfficientNet-B3"] if use_e else [])

    print("\nGEL Diagnostics:")
    _gel_diagnostics(probs, names, gate_passed)
    print(
        f"  p_final range         : [{p_final.min():.3f}, {p_final.max():.3f}]  "
        f"mean={p_final.mean():.3f}  std={p_final.std():.3f}"
    )

    # Individual model metrics
    r_opt_t, _ = _sweep_threshold(labels, p_r, frac_idx)
    d_opt_t, _ = _sweep_threshold(labels, p_d, frac_idx)
    m_resnet   = _evaluate(labels, p_r, r_opt_t, frac_idx)
    m_densenet = _evaluate(labels, p_d, d_opt_t, frac_idx)

    # GEL metrics
    opt_thresh, _ = _sweep_threshold(labels, p_final, frac_idx)
    m_gel_05  = _evaluate(labels, p_final, 0.5,        frac_idx)
    m_gel_opt = _evaluate(labels, p_final, opt_thresh, frac_idx)

    print("\nMetrics:")
    _print_metrics_row(f"ResNet-18   (thr={r_opt_t:.3f} sweep)", m_resnet)
    _print_metrics_row(f"DenseNet-169 (thr={d_opt_t:.3f} sweep)", m_densenet)
    if use_e:
        e_opt_t, _ = _sweep_threshold(labels, p_e, frac_idx)
        m_efficientnet = _evaluate(labels, p_e, e_opt_t, frac_idx)
        _print_metrics_row(f"EfficientNet-B3 (thr={e_opt_t:.3f} sweep)", m_efficientnet)
    print(f"  {'-' * 78}")
    _print_metrics_row("GEL  (thr=0.500 fixed)",        m_gel_05)
    _print_metrics_row(f"GEL  (thr={opt_thresh:.3f} sweep-opt)", m_gel_opt)

    if val_thresh is not None:
        m_gel_val = _evaluate(labels, p_final, val_thresh, frac_idx)
        _print_metrics_row(f"GEL  (thr={val_thresh:.3f} val-optimal)", m_gel_val)

    print(f"\n  {label}-sweep optimal threshold: {opt_thresh:.3f}")
    return opt_thresh


# ── Main ─────────────────────────────────────────────────────────────── #

def main(split):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ResNet-18    weights : {RESNET_WEIGHTS}")
    print(f"DenseNet-169 weights : {DENSENET_WEIGHTS}")
    print(f"EfficientNet weights : {EFFICIENTNET_WEIGHTS}")

    resnet,       r_fi = _load_resnet(device)
    densenet,     d_fi = _load_densenet(device)
    efficientnet, e_fi = _load_efficientnet(device)

    if efficientnet is not None:
        print(f"Models loaded  (r_fi={r_fi}, d_fi={d_fi}, e_fi={e_fi})  — 3-model GEL")
    else:
        print(f"Models loaded  (r_fi={r_fi}, d_fi={d_fi})  — 2-model GEL")

    if split == "both":
        val_thresh = eval_split(
            resnet, densenet, r_fi, d_fi,
            Path("data/dataset_cls/val"), device, label="VAL",
            efficientnet=efficientnet, e_fi=e_fi,
        )
        eval_split(
            resnet, densenet, r_fi, d_fi,
            Path("data/dataset_cls/test"), device, label="TEST",
            val_thresh=val_thresh,
            efficientnet=efficientnet, e_fi=e_fi,
        )
    else:
        eval_split(
            resnet, densenet, r_fi, d_fi,
            Path(f"data/dataset_cls/{split}"), device, label=split.upper(),
            efficientnet=efficientnet, e_fi=e_fi,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="both",
        help="Split to evaluate (default: both — sweeps val then applies to test)",
    )
    args = parser.parse_args()
    main(args.split)
