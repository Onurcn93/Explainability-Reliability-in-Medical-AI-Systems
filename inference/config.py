import os
import torch

# Repo root is one level above this file (inference/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Weight paths — place files in repo_root/weights/
    "yolo_weights":     os.path.join(_ROOT, "weights", "Y1B_detect_best.pt"),
    "resnet_weights":   os.path.join(_ROOT, "weights", "E4a_m050_best.pth"),
    "densenet_weights":     os.path.join(_ROOT, "weights", "D1_best.pth"),
    "efficientnet_weights": os.path.join(_ROOT, "weights", "F1_best.pth"),

    # YOLO inference — fixed from Y1B training
    "yolo_conf_threshold": 0.25,
    "yolo_iou_threshold":  0.5,
    "yolo_imgsz":          600,

    # ResNet-18 inference — fixed from E4a_m050 (val threshold 0.375, optimal sweep)
    "resnet_threshold":  0.375,
    "resnet_input_size": 224,
    "resnet_resize":     256,

    # DenseNet-169 inference — D1 val-sweep optimal threshold (0.175)
    "densenet_threshold":  0.175,
    "densenet_input_size": 224,

    # EfficientNet-B3 inference — F1 training complete (2026-04-28)
    # Val F1=0.6707, threshold=0.525, AUC=0.8825 (early stop ep36, best ep21)
    "efficientnet_threshold":  0.525,
    "efficientnet_input_size": 224,

    # GradCAM — DenseNet-169 D1 (final dense block before global avg pool)
    "gradcam_layer": "features.denseblock4",

    # Device: auto-upgrade to CUDA if available
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ImageNet normalisation (val_test_transforms)
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std":  [0.229, 0.224, 0.225],

    # Served static file
    "index_html": os.path.join(_ROOT, "index.html"),
}

# ---------------------------------------------------------------------------
# GEL — Gated Ensemble Logic hyperparameters
#
# Performance anchors (F1): empirical val-sweep results, 2026-04-25.
#   YOLO is intentionally excluded — detection mAP is not comparable to
#   classification F1 and must not enter the PDWF weighted sum.
#
# Pipeline order: RC init -> OAM -> PDWF -> P_final -> BVG gate.
#
# OAM (Outlier-Aware Modification) — Direction-Aware Asymmetric:
#   mu = mean of loaded classifier probabilities
#   HIGH outlier (p_i > mu): lone fracture signal  -> RC_i_adj = RC_i * gel_penalty_k_high (lenient)
#   LOW  outlier (p_i < mu): lone no-frac dissenter -> RC_i_adj = RC_i * gel_penalty_k_low  (aggressive)
#   Both directions protect against missed fractures from opposite sides.
#   gel_penalty_k_standard (0.20) preserved as symmetric / balanced reference.
#
# PDWF (Performance-Driven Weighted Fusion):
#   P_final = sum(P_i * RC_i_adj) / sum(RC_i_adj)   [loaded classifiers only]
#
# BVG (Binary Verification Gate) — uses P_final, not a separate pre-OAM average:
#   If P_final < gel_tau -> gate fails -> bbox suppressed (P_final still shown).
#   Gate and fracture probability both derive from the same OAM-adjusted estimate.
#
# GEL adapts to 2 or 3 loaded classifiers automatically — see predict.py _run_gel().
# ---------------------------------------------------------------------------

GEL_CONFIG = {
    # Performance anchors — update when champion weights change
    "gel_f1_resnet":       0.658,   # E4a_m050 val F1
    "gel_f1_densenet":     0.724,   # D1 val F1
    "gel_f1_efficientnet": 0.671,   # F1 val F1 (confirmed 2026-04-28 — val F1=0.6707 @ thr=0.525)

    # BVG gate threshold — below this, YOLO bbox is suppressed
    "gel_tau":          0.35,

    # OAM — disagreement limit and direction-aware asymmetric penalty factors
    "gel_disagree_lim":      0.40,
    "gel_penalty_k_low":     0.10,   # LOW outlier  — aggressive: lone no-frac dissenter against fracture consensus
    "gel_penalty_k_high":    0.30,   # HIGH outlier — lenient:    lone fracture signal against no-frac consensus
    "gel_penalty_k_standard": 0.20,  # Balanced symmetric reference (not used in OAM — preserved for comparison)
}

CONFIG.update(GEL_CONFIG)
