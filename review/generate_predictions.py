"""
generate_predictions.py

Per-image prediction CSVs for all splits (train / val / test / all).
Loads champion weights: ResNet-18 E6 (CLAHE), DenseNet-169 D1, EfficientNet-B3 F1.
Runs GEL on each image using the 3-model pipeline.

Two DataLoader passes per split are required because ResNet-18 E6 was trained
with CLAHE preprocessing while DenseNet-169 D1 and EfficientNet-B3 F1 were not.

Output: review/train.csv, review/val.csv, review/test.csv, review/all.csv

Columns:
    split, image_id, true_label,
    resnet_probability, resnet_label,
    densenet_probability, densenet_label,
    efficientnet_probability, efficientnet_label,
    gel_probability, gel_label

Run from repo root:
    python review/generate_predictions.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Paths ──────────────────────────────────────────────────────────────── #

REPO_ROOT  = Path(__file__).resolve().parent.parent
REVIEW_DIR = REPO_ROOT / "review"
DATA_ROOT  = REPO_ROOT / "data" / "dataset_cls"
WEIGHT_DIR = REPO_ROOT / "weights"

# Champion weights
RESNET_WEIGHTS       = WEIGHT_DIR / "E6_best.pth"        # CAALMIX champion (+3.11pp vs E4a)
DENSENET_WEIGHTS     = WEIGHT_DIR / "D1_best.pth"
EFFICIENTNET_WEIGHTS = WEIGHT_DIR / "F1_best.pth"

# ── Hyperparameters (val-optimal thresholds) ───────────────────────────── #

IMG_SIZE      = 224
FRAC_CLASS    = "Fractured"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
BATCH_SIZE    = 32

RESNET_THRESHOLD       = 0.525   # E6 val-optimal
DENSENET_THRESHOLD     = 0.175   # D1 val-optimal
EFFICIENTNET_THRESHOLD = 0.525   # F1 val-optimal
GEL_THRESHOLD          = 0.525   # GEL val-optimal

# GEL PDWF anchors — val F1 for each champion
GEL_F1_RESNET       = 0.689   # E6 val F1 (CAALMIX champion)
GEL_F1_DENSENET     = 0.724   # D1 val F1
GEL_F1_EFFICIENTNET = 0.671   # F1 val F1

GEL_DISAGREE_LIM = 0.40
GEL_PENALTY_K_LOW  = 0.10
GEL_PENALTY_K_HIGH = 0.30

# ── CLAHE preprocessing (E6 was trained with this on all splits) ───────── #

class CLAHETransform:
    """CLAHE local contrast enhancement. clip_limit=2.0, tile=(8,8) — medical imaging standard."""

    def __call__(self, img: Image.Image) -> Image.Image:
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray     = np.array(img.convert("L"), dtype=np.uint8)
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced).convert("RGB")


# ── Transforms ─────────────────────────────────────────────────────────── #

def get_transform_clahe():
    """For ResNet-18 E6 — CLAHE first, then standard pipeline."""
    return transforms.Compose([
        CLAHETransform(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_transform():
    """For DenseNet-169 D1 and EfficientNet-B3 F1 — standard pipeline, no CLAHE."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Model loading ──────────────────────────────────────────────────────── #

def _safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def load_resnet(device):
    ckpt     = _safe_load(RESNET_WEIGHTS, device)
    state    = ckpt.get("model_state_dict", ckpt)
    frac_idx = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    m        = tv_models.resnet18(weights=None)
    m.fc     = (nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
                if "fc.1.weight" in state else nn.Linear(512, 2))
    m.load_state_dict(state)
    m.eval()
    return m.to(device), frac_idx


def load_densenet(device):
    ckpt     = _safe_load(DENSENET_WEIGHTS, device)
    state    = ckpt.get("model_state_dict", ckpt)
    frac_idx = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    m        = tv_models.densenet169(weights=None)
    in_feat  = m.classifier.in_features
    m.classifier = (nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 2))
                    if "classifier.1.weight" in state else nn.Linear(in_feat, 2))
    m.load_state_dict(state)
    m.eval()
    return m.to(device), frac_idx


def load_efficientnet(device):
    if not EFFICIENTNET_WEIGHTS.exists():
        print("[WARN] EfficientNet weights not found — efficientnet columns will be empty.")
        return None, None
    ckpt     = _safe_load(EFFICIENTNET_WEIGHTS, device)
    state    = ckpt.get("model_state_dict", ckpt)
    frac_idx = int(ckpt.get("frac_idx", 0)) if isinstance(ckpt, dict) else 0
    m        = tv_models.efficientnet_b3(weights=None)
    in_feat  = m.classifier[1].in_features
    m.classifier = (nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 2))
                    if "classifier.1.weight" in state else nn.Linear(in_feat, 2))
    m.load_state_dict(state)
    m.eval()
    return m.to(device), frac_idx


# ── GEL (matches eval_gel.py _apply_gel exactly) ──────────────────────── #

def apply_gel(p_r, p_d, p_e=None):
    use_e = p_e is not None
    if use_e:
        total_f1 = GEL_F1_RESNET + GEL_F1_DENSENET + GEL_F1_EFFICIENTNET
        probs    = [p_r, p_d, p_e]
        f1s      = [GEL_F1_RESNET, GEL_F1_DENSENET, GEL_F1_EFFICIENTNET]
    else:
        total_f1 = GEL_F1_RESNET + GEL_F1_DENSENET
        probs    = [p_r, p_d]
        f1s      = [GEL_F1_RESNET, GEL_F1_DENSENET]

    rcs = [np.full_like(p, f1 / total_f1) for p, f1 in zip(probs, f1s)]
    mu  = sum(probs) / len(probs)
    new_rcs = []
    for p, rc in zip(probs, rcs):
        outlier = np.abs(p - mu) > GEL_DISAGREE_LIM
        k       = np.where(p < mu, GEL_PENALTY_K_LOW, GEL_PENALTY_K_HIGH)
        new_rcs.append(np.where(outlier, rc * k, rc))

    total_rc = sum(new_rcs)
    return sum(p * rc for p, rc in zip(probs, new_rcs)) / total_rc


# ── Inference helpers ──────────────────────────────────────────────────── #

def _collect_probs(model, frac_idx, loader, device):
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            probs.extend(torch.softmax(model(imgs.to(device)), dim=1)[:, frac_idx].cpu().numpy())
    return np.array(probs)


def _make_loader(split_dir, transform):
    dataset = ImageFolder(root=str(split_dir), transform=transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return dataset, loader


# ── Per-split inference ─────────────────────────────────────────────────── #

def run_split(split_name, resnet, r_fi, densenet, d_fi, efficientnet, e_fi, device):
    split_dir = DATA_ROOT / split_name
    if not split_dir.exists():
        print(f"[SKIP] {split_dir} not found.")
        return []

    # Pass 1 — ResNet with CLAHE (E6 training distribution)
    ds_clahe, loader_clahe = _make_loader(split_dir, get_transform_clahe())
    frac_idx    = ds_clahe.class_to_idx[FRAC_CLASS]
    image_ids   = [Path(p).name for p, _ in ds_clahe.samples]
    true_labels = [ds_clahe.classes[l] for _, l in ds_clahe.samples]
    p_r = _collect_probs(resnet, r_fi, loader_clahe, device)

    # Pass 2 — DenseNet + EfficientNet without CLAHE
    _, loader_std = _make_loader(split_dir, get_transform())
    p_d = _collect_probs(densenet, d_fi, loader_std, device)
    p_e = _collect_probs(efficientnet, e_fi, loader_std, device) if efficientnet is not None else None

    p_gel = apply_gel(p_r, p_d, p_e)

    rows = []
    for i, (img_id, true_label) in enumerate(zip(image_ids, true_labels)):
        r_prob   = float(p_r[i])
        d_prob   = float(p_d[i])
        e_prob   = float(p_e[i]) if p_e is not None else None
        gel_prob = float(p_gel[i])

        rows.append({
            "split":                    split_name,
            "image_id":                 img_id,
            "true_label":               true_label,
            "resnet_probability":       f"{r_prob:.4f}",
            "resnet_label":             "Fractured" if r_prob >= RESNET_THRESHOLD else "Non_fractured",
            "densenet_probability":     f"{d_prob:.4f}",
            "densenet_label":           "Fractured" if d_prob >= DENSENET_THRESHOLD else "Non_fractured",
            "efficientnet_probability": f"{e_prob:.4f}" if e_prob is not None else "",
            "efficientnet_label":       ("Fractured" if e_prob >= EFFICIENTNET_THRESHOLD else "Non_fractured") if e_prob is not None else "",
            "gel_probability":          f"{gel_prob:.4f}",
            "gel_label":                "Fractured" if gel_prob >= GEL_THRESHOLD else "Non_fractured",
        })

    print(f"[{split_name:5s}] {len(rows)} images processed.")
    return rows


# ── CSV writing ────────────────────────────────────────────────────────── #

FIELDNAMES = [
    "split", "image_id", "true_label",
    "resnet_probability", "resnet_label",
    "densenet_probability", "densenet_label",
    "efficientnet_probability", "efficientnet_label",
    "gel_probability", "gel_label",
]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"         -> {path.name}  ({len(rows)} rows)")


# ── Main ────────────────────────────────────────────────────────────────── #

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"ResNet-18    : {RESNET_WEIGHTS}  [CLAHE preprocessing]")
    print(f"DenseNet-169 : {DENSENET_WEIGHTS}")
    print(f"EfficientNet : {EFFICIENTNET_WEIGHTS}\n")

    resnet,       r_fi = load_resnet(device)
    densenet,     d_fi = load_densenet(device)
    efficientnet, e_fi = load_efficientnet(device)

    n_models = 3 if efficientnet is not None else 2
    print(f"Loaded {n_models}-model GEL  (r_fi={r_fi}, d_fi={d_fi}, e_fi={e_fi})\n")

    REVIEW_DIR.mkdir(exist_ok=True)

    all_rows = []
    for split in ("train", "val", "test"):
        rows = run_split(split, resnet, r_fi, densenet, d_fi, efficientnet, e_fi, device)
        if rows:
            write_csv(REVIEW_DIR / f"{split}.csv", rows)
            all_rows.extend(rows)

    if all_rows:
        write_csv(REVIEW_DIR / "all.csv", all_rows)

    print("\nDone.")


if __name__ == "__main__":
    main()
