"""
models/classification/densenet.py

DenseNet-169 training module for FracAtlas binary fracture classification.

Follows the same interface as models/classification/resnet.py:
    run_training(config: dict) -> Path

Supported experiments
---------------------
    D1  — Full fine-tune, plateau scheduler, flat LR, 50 epochs
    D2  — Cosine warmup + post-training threshold sweep (mirrors E4e)
    D3  — CLAHE all splits (mirrors E5)
    D4  — CLAHE + AlbumentationsDelta train-only (mirrors E6)
    D5  — CLAHE + AlbumentationsDelta + XRayAugMix train-only (mirrors E7)

Config keys (from YAML)
-----------------------
    experiment_id   str     e.g. "D1"
    task            str     must be "classify_densenet"
    data_dir        str     path to ImageFolder split root (train/ val/ test/)
    epochs          int     training epochs
    batch_size      int     default 32
    img_size        int     default 224
    device          str     "0" for GPU 0, or "cpu"
    dropout_p       float   0.0 = no dropout
    weight_mult     float   class weight multiplier (0.0 → flat, 1.0 → natural ratio)
    loss            str     "weighted_ce" or "focal"
    gamma           float   focal loss gamma (only used when loss=focal, default 1.0)
    scheduler       str     "plateau" or "cosine_warmup"
    warmup_epochs   int     warmup epochs for cosine_warmup (default 3)
    lr_backbone     float   backbone learning rate (default 1e-4)
    lr_head         float   head learning rate (default 1e-4)
    val_threshold   float   decision threshold used during checkpoint selection (default 0.5)
    use_clahe           bool    CLAHE local contrast enhancement on all splits (default False)
    use_albu            bool    AlbumentationsDelta on train only (default False)
    use_augmix          bool    XRayAugMix on train only (default False)
    early_stop_patience int     stop if no val F1 improvement for N epochs; 0=disabled (default 0)
    plot                bool    whether to save training curves

Prerequisite
------------
    Run data/prepare_classification.py once to build the ImageFolder split dirs.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _worker_init(worker_id):  # noqa: ARG001
    from PIL import ImageFile as _IF
    _IF.LOAD_TRUNCATED_IMAGES = True


from utils.logger import Logger
from utils.plot import plot_training_curves
from utils.augmentations import AlbumentationsDelta, XRayAugMix

# ── Constants ─────────────────────────────────────────────────────── #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
FRAC_CLASS    = "Fractured"
WEIGHTS_DIR   = Path("weights")


# ── Loss ──────────────────────────────────────────────────────────── #

class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        fl = (1.0 - pt) ** self.gamma * ce
        return fl.mean() if self.reduction == "mean" else fl.sum()


# ── Scheduler ─────────────────────────────────────────────────────── #

class WarmupCosineScheduler:
    """Linear LR warmup followed by cosine decay.

    Manages two separate param groups (backbone and head) independently.
    Call scheduler.step() once per epoch.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr_backbone: float = 1e-5,
        base_lr_head: float = 1e-3,
        min_lr: float = 1e-7,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_bb       = base_lr_backbone
        self.base_head     = base_lr_head
        self.min_lr        = min_lr
        self._epoch        = 0

    def step(self) -> None:
        self._epoch += 1
        e, we, te = self._epoch, self.warmup_epochs, self.total_epochs

        if e <= we:
            frac_bb   = self.base_bb   * e / we
            frac_head = self.base_head * e / we
        else:
            progress  = (e - we) / max(te - we, 1)
            cos_val   = 0.5 * (1.0 + math.cos(math.pi * progress))
            frac_bb   = self.min_lr + (self.base_bb   - self.min_lr) * cos_val
            frac_head = self.min_lr + (self.base_head - self.min_lr) * cos_val

        self.optimizer.param_groups[0]["lr"] = frac_bb
        self.optimizer.param_groups[1]["lr"] = frac_head


# ── Model builder ─────────────────────────────────────────────────── #

def _build_model(dropout_p: float, device: torch.device) -> nn.Module:
    """DenseNet-169 (ImageNet pretrained), all layers unfrozen, optional dropout.

    DenseNet-169 head is model.classifier (Linear 1664→1000).
    Replaced with Linear 1664→2 (binary: Fractured / Non_fractured).
    """
    model   = tv_models.densenet169(weights=tv_models.DenseNet169_Weights.IMAGENET1K_V1)
    in_feat = model.classifier.in_features  # 1664

    if dropout_p > 0.0:
        model.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_feat, 2))
    else:
        model.classifier = nn.Linear(in_feat, 2)

    for p in model.parameters():
        p.requires_grad = True

    return model.to(device)


# ── Class weights ─────────────────────────────────────────────────── #

def _compute_class_weights(
    data_dir: Path, weight_mult: float, device: torch.device
) -> torch.Tensor:
    ds        = ImageFolder(root=str(data_dir / "train"))
    frac_idx  = ds.class_to_idx[FRAC_CLASS]
    labels    = [lbl for _, lbl in ds.samples]
    n_total   = len(labels)
    n_frac    = labels.count(frac_idx)
    n_nonfrac = n_total - n_frac

    base_frac    = n_total / (2 * n_frac)
    base_nonfrac = n_total / (2 * n_nonfrac)

    w_frac    = 1.0 + (base_frac    - 1.0) * weight_mult
    w_nonfrac = 1.0 + (base_nonfrac - 1.0) * weight_mult

    weights             = torch.zeros(2, dtype=torch.float32)
    weights[frac_idx]   = w_frac
    weights[1 - frac_idx] = w_nonfrac
    return weights.to(device)


# ── Transforms ────────────────────────────────────────────────────── #

class CLAHETransform:
    """CLAHE local contrast enhancement for X-ray images. Applied to ALL splits (preprocessing).

    Converts to grayscale, applies CLAHE, returns PIL RGB so downstream
    Grayscale(3ch) and normalisation remain identical to the non-CLAHE pipeline.
    clip_limit=2.0 and tile_grid=(8,8) are the standard medical-imaging defaults.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)):
        self._clip_limit = clip_limit
        self._tile_grid  = tile_grid

    def __call__(self, img: Image.Image) -> Image.Image:
        clahe    = cv2.createCLAHE(clipLimit=self._clip_limit, tileGridSize=self._tile_grid)
        gray     = np.array(img.convert("L"), dtype=np.uint8)
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced).convert("RGB")


def _get_transforms(
    img_size: int,
    use_clahe: bool = False,
    use_albu: bool = False,
    use_augmix: bool = False,
):
    """Return (train_tf, val_tf, tta_transforms_list)."""
    clahe_step  = [CLAHETransform()] if use_clahe else []
    albu_step   = [AlbumentationsDelta()] if use_albu else []
    augmix_step = [XRayAugMix()] if use_augmix else []

    val_tf = transforms.Compose(
        clahe_step + [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_tf = transforms.Compose(
        clahe_step + [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ] + albu_step + augmix_step + [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    # NOTE: TTA passes 1-3 intentionally omit CLAHE. When use_clahe=True,
    # this creates an input-distribution mismatch (CLAHE-trained model sees
    # non-CLAHE images). This is why TTA regresses on D3. TTA is
    # disabled in all experiments — do not use it.
    tta_tfs = [
        val_tf,   # original (includes CLAHE if use_clahe=True)
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    ]
    return train_tf, val_tf, tta_tfs


# ── Training ──────────────────────────────────────────────────────── #

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(dim=1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


# ── Evaluation ────────────────────────────────────────────────────── #

@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    frac_idx: int,
    threshold: float,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, total = 0.0, 0
    all_labels: List[int]   = []
    all_probs:  List[float] = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(out, dim=1)[:, frac_idx].cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)
        total += imgs.size(0)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    preds      = np.where(all_probs >= threshold, frac_idx, 1 - frac_idx)
    bin_labels = (all_labels == frac_idx).astype(int)

    recall = recall_score(all_labels, preds, pos_label=frac_idx, zero_division=0)
    prec   = precision_score(all_labels, preds, pos_label=frac_idx, zero_division=0)
    f1     = f1_score(all_labels, preds, pos_label=frac_idx, zero_division=0)
    auc    = roc_auc_score(bin_labels, all_probs) if len(np.unique(bin_labels)) > 1 else 0.0

    return {
        "loss": total_loss / total,
        "f1": f1, "recall": recall, "precision": prec, "auc": auc,
    }


@torch.no_grad()
def _evaluate_tta(
    model: nn.Module,
    split_path: Path,
    tta_tfs: list,
    frac_idx: int,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    ref_ds     = ImageFolder(root=str(split_path), transform=tta_tfs[0])
    n          = len(ref_ds)
    all_labels = np.array([lbl for _, lbl in ref_ds.samples])
    tta_probs  = np.zeros((n, len(tta_tfs)), dtype=np.float32)

    for ti, tf in enumerate(tta_tfs):
        ds     = ImageFolder(root=str(split_path), transform=tf)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, worker_init_fn=_worker_init)
        idx = 0
        for imgs, _ in loader:
            imgs  = imgs.to(device)
            probs = torch.softmax(model(imgs), dim=1)[:, frac_idx].cpu().numpy()
            tta_probs[idx : idx + len(probs), ti] = probs
            idx += len(probs)

    avg_probs  = tta_probs.mean(axis=1)
    preds      = np.where(avg_probs >= threshold, frac_idx, 1 - frac_idx)
    bin_labels = (all_labels == frac_idx).astype(int)

    recall = recall_score(all_labels, preds, pos_label=frac_idx, zero_division=0)
    prec   = precision_score(all_labels, preds, pos_label=frac_idx, zero_division=0)
    f1     = f1_score(all_labels, preds, pos_label=frac_idx, zero_division=0)
    auc    = roc_auc_score(bin_labels, avg_probs) if len(np.unique(bin_labels)) > 1 else 0.0

    return {"f1": f1, "recall": recall, "precision": prec, "auc": auc}


# ── Threshold sweep ───────────────────────────────────────────────── #

@torch.no_grad()
def _threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    frac_idx: int,
    device: torch.device,
) -> Tuple[float, float]:
    """Sweep thresholds 0.05–0.95 on the val set. Returns (best_thresh, best_f1)."""
    model.eval()
    all_labels: List[int]   = []
    all_probs:  List[float] = []

    for imgs, labels in loader:
        probs = torch.softmax(model(imgs.to(device)), dim=1)[:, frac_idx].cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    best_thresh, best_f1 = 0.5, 0.0

    for t in np.arange(0.05, 0.95, 0.025):
        preds = np.where(all_probs >= t, frac_idx, 1 - frac_idx)
        f1    = f1_score(all_labels, preds, pos_label=frac_idx, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(t)

    return best_thresh, best_f1


# ── Public entry point ────────────────────────────────────────────── #

def run_training(config: dict) -> Path:
    """Run DenseNet-169 classification training from a config dict.

    Args:
        config: Experiment config loaded from YAML. See module docstring.

    Returns:
        Path to best checkpoint saved in weights/.
    """
    exp_id      = config["experiment_id"]
    data_dir    = Path(config["data_dir"])
    epochs      = config["epochs"]
    batch_size  = config.get("batch_size",    32)
    img_size    = config.get("img_size",      224)
    dropout_p   = config.get("dropout_p",     0.0)
    weight_mult = config.get("weight_mult",   1.0)
    loss_type   = config.get("loss",          "weighted_ce")
    gamma       = config.get("gamma",         1.0)
    sched_type  = config.get("scheduler",     "plateau")
    warmup_ep   = config.get("warmup_epochs", 3)
    lr_bb       = config.get("lr_backbone",   1e-4)
    lr_head     = config.get("lr_head",       1e-4)
    val_thresh  = config.get("val_threshold", 0.5)
    use_clahe           = config.get("use_clahe",             False)
    use_albu            = config.get("use_albu",              False)
    use_augmix          = config.get("use_augmix",            False)
    early_stop_patience = config.get("early_stop_patience",   0)

    # ── Device ───────────────────────────────────────────────────── #
    dev_str = str(config.get("device", "cpu"))
    if dev_str != "cpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{dev_str}")
    else:
        device = torch.device("cpu")

    # ── Experiment label ──────────────────────────────────────────── #
    experiment = (
        f"{exp_id} | DenseNet-169 | classify"
        f" | epochs={epochs}"
        f" | loss={loss_type}"
        + (f"(γ={gamma})" if loss_type == "focal" else "")
        + f" | w={weight_mult}"
        + (f" | drop={dropout_p}" if dropout_p > 0.0 else "")
        + (" | CLAHE" if use_clahe else "")
        + (" | Albu" if use_albu else "")
        + (" | AugMix" if use_augmix else "")
        + f" | {sched_type}"
    )

    # ── Logger ───────────────────────────────────────────────────── #
    metrics_keys = ["TrLoss", "VaLoss", "VaF1", "VaRec", "VaPrec", "VaAUC"]
    logger = Logger(
        experiment     = experiment,
        metrics        = metrics_keys,
        primary_metric = "VaF1",
    )

    logger.log_start({
        "Task"     : "classify  (binary: Fractured / Non_fractured)",
        "Data"     : str(data_dir),
        "Epochs"   : f"{epochs}  |  batch: {batch_size}  |  imgsz: {img_size}  |  Device: {device}",
        "Model"    : f"DenseNet-169 (ImageNet pretrained)  |  in_features: 1664  |  dropout: {dropout_p}",
        "Loss"     : loss_type + (f"  gamma={gamma}" if loss_type == "focal" else ""),
        "Weights"  : f"class weight multiplier: {weight_mult}",
        "Scheduler": sched_type + (f"  warmup={warmup_ep} ep" if sched_type == "cosine_warmup" else ""),
        "LR"       : f"backbone={lr_bb}  |  head={lr_head}",
    })

    # ── Data ─────────────────────────────────────────────────────── #
    if not (data_dir / "train").exists():
        logger.log_message(
            f"[error] data_dir '{data_dir}' not found. "
            "Run: python data/prepare_classification.py"
        )
        logger.close()
        return Path()

    train_tf, val_tf, tta_tfs = _get_transforms(img_size, use_clahe=use_clahe, use_albu=use_albu, use_augmix=use_augmix)
    train_ds = ImageFolder(root=str(data_dir / "train"), transform=train_tf)
    val_ds   = ImageFolder(root=str(data_dir / "val"),   transform=val_tf)
    frac_idx = train_ds.class_to_idx[FRAC_CLASS]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, worker_init_fn=_worker_init,
    )

    labels_all = [lbl for _, lbl in train_ds.samples]
    n_frac     = labels_all.count(frac_idx)
    n_nonfrac  = len(labels_all) - n_frac
    logger.log_message(
        f"[data] Train: {len(train_ds)} images  "
        f"({n_frac} Fractured, {n_nonfrac} Non-fractured, ratio {n_nonfrac/n_frac:.2f}:1)"
        f"  |  Val: {len(val_ds)} images  |  Fractured class idx: {frac_idx}"
    )

    # ── Model ────────────────────────────────────────────────────── #
    model = _build_model(dropout_p, device)

    # ── Loss function ────────────────────────────────────────────── #
    class_weights = _compute_class_weights(data_dir, weight_mult, device)
    if loss_type == "focal":
        criterion = FocalLoss(gamma=gamma, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.log_message(
        f"[loss] Fractured weight: {class_weights[frac_idx]:.3f}  |  "
        f"Non-fractured weight: {class_weights[1 - frac_idx]:.3f}  |  "
        f"Ratio: {class_weights[frac_idx] / class_weights[1 - frac_idx]:.2f}:1"
    )

    # ── Optimizer (differential LR: low for backbone, higher for head) ── #
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params     = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": lr_bb},
        {"params": head_params,     "lr": lr_head},
    ])

    # ── Scheduler ────────────────────────────────────────────────── #
    if sched_type == "cosine_warmup":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_ep,
            total_epochs=epochs,
            base_lr_backbone=lr_bb,
            base_lr_head=lr_head,
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

    # ── Training loop ────────────────────────────────────────────── #
    WEIGHTS_DIR.mkdir(exist_ok=True)
    ckpt_path        = WEIGHTS_DIR / f"{exp_id}_best.pth"
    best_val_f1      = 0.0
    epochs_no_improve = 0

    series: Dict[str, List[float]] = {k: [] for k in metrics_keys}

    for epoch in range(1, epochs + 1):
        tr_loss, _ = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_m      = _evaluate(model, val_loader, criterion, frac_idx, val_thresh, device)

        if sched_type == "cosine_warmup":
            scheduler.step()
        else:
            scheduler.step(val_m["loss"])

        ep = {
            "TrLoss": tr_loss,
            "VaLoss": val_m["loss"],
            "VaF1":   val_m["f1"],
            "VaRec":  val_m["recall"],
            "VaPrec": val_m["precision"],
            "VaAUC":  val_m["auc"],
        }
        for k, v in ep.items():
            series[k].append(v)
        logger.log_epoch(epoch, ep)

        if val_m["f1"] > best_val_f1:
            best_val_f1       = val_m["f1"]
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics":          val_m,
                    "exp_id":               exp_id,
                    "val_threshold":        val_thresh,
                    "frac_idx":             frac_idx,
                },
                ckpt_path,
            )
            logger.log_best(best_val_f1, str(ckpt_path))
        else:
            epochs_no_improve += 1
            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                logger.log_message(
                    f"[early stop] No val F1 improvement for {early_stop_patience} epochs — stopping at epoch {epoch}."
                )
                break

    # ── Post-training threshold sweep on val ─────────────────────── #
    # Load the saved best checkpoint so the sweep runs on the best epoch's
    # weights, not the last epoch's weights (which may have overfit past the best).
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    logger.log_message("")
    logger.log_message("[sweep] Threshold sweep on val set (0.05 → 0.95, step 0.025)...")
    opt_thresh, opt_f1 = _threshold_sweep(model, val_loader, frac_idx, device)
    logger.log_message(
        f"[sweep] Optimal threshold: {opt_thresh:.3f}  |  "
        f"Val F1 @ optimal: {opt_f1:.4f}  "
        f"(trained with threshold={val_thresh:.2f})"
    )

    val_opt = _evaluate(model, val_loader, criterion, frac_idx, opt_thresh, device)
    logger.log_message(
        f"[sweep] Val @ opt threshold  —  "
        f"F1={val_opt['f1']:.4f}  Recall={val_opt['recall']:.4f}  "
        f"Prec={val_opt['precision']:.4f}  AUC={val_opt['auc']:.4f}"
    )

    # ── TTA benefit check on val ──────────────────────────────────── #
    logger.log_message("")
    logger.log_message("[tta] Checking TTA benefit on val set (4 passes)...")
    tta_m    = _evaluate_tta(model, data_dir / "val", tta_tfs, frac_idx,
                             opt_thresh, batch_size, device)
    tta_gain = tta_m["f1"] - val_opt["f1"]
    logger.log_message(
        f"[tta] Without TTA: F1={val_opt['f1']:.4f}  |  "
        f"With TTA: F1={tta_m['f1']:.4f}  |  Gain: {tta_gain:+.4f}"
    )
    logger.log_message(
        "[tta] Recommendation: "
        + ("USE TTA at test time" if tta_m["f1"] > val_opt["f1"] else "Skip TTA — no improvement on val")
    )

    # Patch checkpoint with swept optimal threshold for direct use in inference.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["val_threshold"] = opt_thresh
    torch.save(ckpt, ckpt_path)
    logger.log_message(f"[sweep] Checkpoint updated: val_threshold → {opt_thresh:.3f}")

    logger.log_complete(best_val_f1, str(ckpt_path))
    logger.close()

    # ── Training curves ───────────────────────────────────────────── #
    if config.get("plot", True):
        plot_training_curves(
            series     = series,
            groups     = [["TrLoss", "VaLoss"], ["VaF1", "VaRec", "VaPrec", "VaAUC"]],
            titles     = ["Loss per Epoch", "Val Medical Metrics"],
            ylabels    = ["Loss", "Score"],
            experiment = experiment,
        )

    return ckpt_path
