"""
inference/predict.py — FracAssist inference module.

Selective cascade logic:
  YOLO-LED       : YOLO fires a box (conf >= 0.25) → fracture confirmed by detector.
  CLASSIFIER-LED : YOLO no box → ResNet-18 decides (if loaded); else defaults Non-Fractured.

Both ResNet-18 (E4a) and DenseNet-169 (D1) run in parallel when loaded.
ResNet-18 drives the cascade decision; DenseNet-169 provides a secondary probability.
GradCAM is generated on the ResNet-18 branch when ResNet is loaded.

ResNet-18 / DenseNet-169 weights are optional — the module degrades gracefully if
checkpoints are absent.
"""

import os

import cv2
import numpy as np


def _imread(path: str) -> "np.ndarray | None":
    """cv2.imread that works on Windows paths containing spaces or Unicode."""
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tv_models, transforms
from ultralytics import YOLO

import utils.gradcam as gradcam_utils

# ---------------------------------------------------------------------------
# Module-level model handles — loaded once at startup, reused per request
# ---------------------------------------------------------------------------
_yolo_model        = None

_resnet_model      = None
_resnet_loaded     = False
_resnet_frac_idx   = 0
_resnet_threshold  = 0.375   # E4a_m050 optimal val threshold

_densenet_model     = None
_densenet_loaded    = False
_densenet_frac_idx  = 0
_densenet_threshold = 0.175  # D1 val-sweep optimal; overridden from config at load time


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(config):
    """Load YOLO, ResNet-18, and DenseNet-169 (latter two optional/graceful).
    Called once at app startup. Raises FileNotFoundError if YOLO weights missing."""
    global _yolo_model
    global _resnet_model, _resnet_loaded, _resnet_frac_idx, _resnet_threshold
    global _densenet_model, _densenet_loaded, _densenet_frac_idx, _densenet_threshold

    device = config["device"]

    # --- YOLO (required) ---
    yolo_path = config["yolo_weights"]
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(
            f"YOLO weights not found: {yolo_path}\n"
            "Place Y1B_detect_best.pt in the weights/ directory."
        )
    _yolo_model = YOLO(yolo_path)
    print(f"[INFO] YOLO loaded: {yolo_path}")

    # --- ResNet-18 (optional — E4a_m050) ---
    resnet_path = config["resnet_weights"]
    if os.path.exists(resnet_path):
        ckpt = torch.load(resnet_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state              = ckpt["model_state_dict"]
            _resnet_frac_idx   = ckpt.get("frac_idx",      0)
            _resnet_threshold  = ckpt.get("val_threshold",  config["resnet_threshold"])
        else:
            state              = ckpt
            _resnet_frac_idx   = 0
            _resnet_threshold  = config["resnet_threshold"]

        has_dropout = "fc.1.weight" in state
        m = tv_models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2)) if has_dropout else nn.Linear(512, 2)
        m.load_state_dict(state)
        m.eval()
        _resnet_model  = m.to(device)
        _resnet_loaded = True
        print(
            f"[INFO] ResNet-18 loaded: {resnet_path} "
            f"(threshold={_resnet_threshold:.3f}, dropout={has_dropout})"
        )
    else:
        print(f"[WARN] ResNet-18 weights not found: {resnet_path} — YOLO-only mode.")
        _resnet_loaded = False

    # --- DenseNet-169 (optional — D1) ---
    densenet_path = config.get("densenet_weights", "")
    if densenet_path and os.path.exists(densenet_path):
        ckpt = torch.load(densenet_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state               = ckpt["model_state_dict"]
            _densenet_frac_idx  = ckpt.get("frac_idx", 0)
        else:
            state               = ckpt
            _densenet_frac_idx  = 0
        _densenet_threshold = config.get("densenet_threshold", 0.175)

        # Detect dropout head: Sequential(Dropout, Linear) → "classifier.1.weight"
        has_dropout = "classifier.1.weight" in state
        m = tv_models.densenet169(weights=None)
        in_feat = m.classifier.in_features  # 1664
        m.classifier = (
            nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 2))
            if has_dropout else nn.Linear(in_feat, 2)
        )
        m.load_state_dict(state)
        m.eval()
        _densenet_model  = m.to(device)
        _densenet_loaded = True
        print(
            f"[INFO] DenseNet-169 loaded: {densenet_path} "
            f"(threshold={_densenet_threshold:.3f}, dropout={has_dropout})"
        )
    else:
        print(f"[INFO] DenseNet-169 weights not found — will show as pending in UI.")
        _densenet_loaded = False

    return _yolo_model, _resnet_model, _densenet_model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image_path, input_size, config):
    """Val/test transform — Resize, Grayscale 3ch, ToTensor, ImageNet norm."""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["imagenet_mean"], std=config["imagenet_std"]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def run_yolo(model, image_path, config):
    """Run YOLO detection. Returns list of {confidence, bbox} dicts, or []."""
    results = model.predict(
        source=image_path,
        imgsz=config["yolo_imgsz"],
        conf=config["yolo_conf_threshold"],
        iou=config["yolo_iou_threshold"],
        verbose=False,
    )
    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "confidence": float(box.conf[0]),
                "bbox": [float(v) for v in box.xyxy[0]],
            })
    return detections


def run_resnet(tensor, config):
    """ResNet-18 forward pass. Returns (label, probability)."""
    device = config["device"]
    with torch.no_grad():
        logits = _resnet_model(tensor.to(device))
        prob   = float(torch.softmax(logits, dim=1)[0, _resnet_frac_idx])
    label = "Fractured" if prob >= _resnet_threshold else "Non-Fractured"
    return label, prob


def run_densenet(tensor, config):
    """DenseNet-169 forward pass. Returns (label, probability)."""
    device = config["device"]
    with torch.no_grad():
        logits = _densenet_model(tensor.to(device))
        prob   = float(torch.softmax(logits, dim=1)[0, _densenet_frac_idx])
    label = "Fractured" if prob >= _densenet_threshold else "Non-Fractured"
    return label, prob


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _encode_base64(img_bgr):
    import base64
    import cv2 as _cv2
    _, buf = _cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def _image_to_base64(image_path):
    img = _imread(image_path)
    if img is None:
        return None
    return _encode_base64(img)


def _draw_bbox_base64(image_path, bbox, confidence):
    img = _imread(image_path)
    if img is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    red_bgr = (0, 91, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), red_bgr, 2)
    label = f"FRACTURE {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), red_bgr, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return _encode_base64(img)


# ---------------------------------------------------------------------------
# Top-level predict
# ---------------------------------------------------------------------------

def predict(image_path, config):
    """Orchestrate selective cascade and return result dict.

    Decision tree:
      YOLO fires box  → YOLO-LED   (YOLO primary)
      YOLO no box     → CLASSIFIER-LED (ResNet-18 primary, if loaded)

    ResNet-18 and DenseNet-169 both run in parallel when loaded.
    ResNet-18 drives the cascade decision; DenseNet-169 is a secondary output.
    GradCAM generated on ResNet-18 when loaded.
    """
    result = {
        "mode":                  None,
        "label":                 "Non-Fractured",
        "fracture_probability":  0.0,
        "yolo_confidence":       None,
        "bbox":                  None,
        "resnet_probability":    None,
        "densenet_probability":  None,
        "gradcam_image":         None,
        "xray_with_box":         None,
        "disclaimer": (
            "This prediction is provided to support radiologist review. "
            "Clinical judgment is required for diagnosis."
        ),
        "error": None,
    }

    # --- Shared preprocessing tensors (built once, reused) ---
    resnet_tensor   = _preprocess(image_path, config["resnet_input_size"], config) if _resnet_loaded else None
    densenet_tensor = _preprocess(image_path, config["densenet_input_size"], config) if _densenet_loaded else None

    # Step 1: YOLO
    detections = run_yolo(_yolo_model, image_path, config)

    if detections:
        # ----------------------------------------------------------------
        # YOLO-LED: box detected
        # ----------------------------------------------------------------
        best = max(detections, key=lambda d: d["confidence"])
        result["mode"]                 = "YOLO-LED"
        result["label"]                = "Fractured"
        result["yolo_confidence"]      = best["confidence"]
        result["fracture_probability"] = best["confidence"]
        result["bbox"]                 = best["bbox"]
        result["xray_with_box"]        = _draw_bbox_base64(
            image_path, best["bbox"], best["confidence"]
        )

        if _resnet_loaded:
            _, resnet_prob = run_resnet(resnet_tensor, config)
            result["resnet_probability"] = resnet_prob
            result["gradcam_image"] = gradcam_utils.to_base64(
                _resnet_model, resnet_tensor, image_path,
                _resnet_frac_idx, config["device"],
                layer_name=config.get("gradcam_layer", "layer4"),
            )
        else:
            result["gradcam_image"] = result["xray_with_box"]

        if _densenet_loaded:
            _, densenet_prob = run_densenet(densenet_tensor, config)
            result["densenet_probability"] = densenet_prob

    else:
        # ----------------------------------------------------------------
        # CLASSIFIER-LED: YOLO found nothing
        # ----------------------------------------------------------------
        result["mode"] = "CLASSIFIER-LED"

        if _resnet_loaded:
            label, resnet_prob = run_resnet(resnet_tensor, config)
            result["label"]                = label
            result["resnet_probability"]   = resnet_prob
            result["fracture_probability"] = resnet_prob
            result["gradcam_image"] = gradcam_utils.to_base64(
                _resnet_model, resnet_tensor, image_path,
                _resnet_frac_idx, config["device"],
                layer_name=config.get("gradcam_layer", "layer4"),
            )
        else:
            result["label"]       = "Non-Fractured"
            result["gradcam_image"] = _image_to_base64(image_path)

        if _densenet_loaded:
            _, densenet_prob = run_densenet(densenet_tensor, config)
            result["densenet_probability"] = densenet_prob

    return result
