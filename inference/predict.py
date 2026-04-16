"""
inference/predict.py — FracAssist inference module.

Selective ensemble logic:
  YOLO-LED       : YOLO fires a box (conf >= 0.25) → fracture confirmed by detector.
  CLASSIFIER-LED : YOLO no box → ResNet-18 decides (if loaded); else defaults Non-Fractured.

GradCAM always generated on the ResNet-18 branch when ResNet is loaded.

ResNet-18 weights are optional at startup — the module degrades gracefully if
E4e_best.pth is not yet present (YOLO-only mode until weights are placed).
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
from torchvision import models, transforms
from ultralytics import YOLO

import utils.gradcam as gradcam_utils

# ---------------------------------------------------------------------------
# Module-level model handles — loaded once at startup, reused per request
# ---------------------------------------------------------------------------
_yolo_model    = None
_resnet_model  = None
_resnet_loaded = False
_frac_idx      = 0      # Fractured class index (0 for FracAtlas ImageFolder)
_val_threshold = 0.5    # Updated from checkpoint at load time


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(config):
    """Load YOLO and (optionally) ResNet-18 from paths in config.
    Called once at app startup. Raises FileNotFoundError if YOLO weights missing."""
    global _yolo_model, _resnet_model, _resnet_loaded, _frac_idx, _val_threshold

    # --- YOLO (required) ---
    yolo_path = config["yolo_weights"]
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(
            f"YOLO weights not found: {yolo_path}\n"
            "Place Y1B_detect_best.pt in the weights/ directory."
        )
    _yolo_model = YOLO(yolo_path)
    print(f"[INFO] YOLO loaded: {yolo_path}")

    # --- ResNet-18 (optional — degrades gracefully if absent) ---
    resnet_path = config["resnet_weights"]
    if os.path.exists(resnet_path):
        device = config["device"]

        # Load checkpoint — our training format wraps state_dict in a dict
        ckpt = torch.load(resnet_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state       = ckpt["model_state_dict"]
            _frac_idx      = ckpt.get("frac_idx",       0)
            _val_threshold = ckpt.get("val_threshold",  config["resnet_threshold"])
        else:
            # Legacy raw state dict
            state          = ckpt
            _frac_idx      = 0
            _val_threshold = config["resnet_threshold"]

        # Reconstruct architecture from state dict key shape
        # With dropout head:    fc.0 = Dropout, fc.1 = Linear → key "fc.1.weight"
        # Without dropout head: fc   = Linear               → key "fc.weight"
        has_dropout = "fc.1.weight" in state

        m = models.resnet18(weights=None)
        if has_dropout:
            m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2))
        else:
            m.fc = nn.Linear(512, 2)

        m.load_state_dict(state)
        m.eval()
        _resnet_model = m.to(device)
        _resnet_loaded = True
        print(
            f"[INFO] ResNet-18 loaded: {resnet_path} "
            f"(device={device}, frac_idx={_frac_idx}, threshold={_val_threshold:.3f}, "
            f"dropout_head={has_dropout})"
        )
    else:
        print(
            f"[WARN] ResNet-18 weights not found at {resnet_path}. "
            "Running in YOLO-only mode until weights are placed."
        )
        _resnet_loaded = False

    return _yolo_model, _resnet_model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_resnet(image_path, config):
    """Val/test transform — matches the val_tf used during training exactly.

    Resize to (input_size × input_size), Grayscale → 3ch, ToTensor, Normalize.
    Returns a (1, 3, input_size, input_size) float tensor.
    """
    size = config["resnet_input_size"]
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),   # X-ray 1ch → 3ch for ImageNet model
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config["imagenet_mean"],
            std=config["imagenet_std"],
        ),
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


def run_resnet(model, tensor, config):
    """Softmax forward pass, threshold on fracture class probability.

    Returns (label: str, probability: float) where probability is the
    model's confidence that the image contains a fracture.
    """
    device    = config["device"]
    threshold = _val_threshold
    with torch.no_grad():
        logits = model(tensor.to(device))                      # (1, 2)
        prob   = float(torch.softmax(logits, dim=1)[0, _frac_idx])
    label = "Fractured" if prob >= threshold else "Non-Fractured"
    return label, prob


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _encode_base64(img_bgr):
    """Encode a BGR numpy image as a data:image/png;base64,... string."""
    import base64
    import cv2 as _cv2
    _, buf = _cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def _image_to_base64(image_path):
    """Read an image file and return it as a base64 PNG string."""
    img = _imread(image_path)
    if img is None:
        return None
    return _encode_base64(img)


def _draw_bbox_base64(image_path, bbox, confidence):
    """Draw YOLO bounding box on the original image, return base64 PNG."""
    img = _imread(image_path)
    if img is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    red_bgr = (0, 91, 255)   # BGR red matching the UI's --fa-red
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
    """Orchestrate selective ensemble and return result dict.

    Decision tree:
      YOLO fires box  → YOLO-LED   (YOLO primary, ResNet secondary if loaded)
      YOLO no box     → CLASSIFIER-LED (ResNet if loaded, else Non-Fractured stub)

    GradCAM is always generated when ResNet is loaded.
    """
    result = {
        "mode":                  None,
        "label":                 "Non-Fractured",
        "fracture_probability":  0.0,
        "yolo_confidence":       None,
        "bbox":                  None,
        "resnet_probability":    0.0,
        "gradcam_image":         None,
        "xray_with_box":         None,
        "body_part":             "Unknown",   # placeholder — body region model pending
        "body_part_confidence":  0.0,
        "disclaimer": (
            "This prediction is provided to support radiologist review. "
            "Clinical judgment is required for diagnosis."
        ),
        "error": None,
    }

    # Step 1: YOLO
    detections = run_yolo(_yolo_model, image_path, config)

    if detections:
        # ----------------------------------------------------------------
        # Step 2a — YOLO-LED: box detected, fracture evidence present
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
            tensor = preprocess_resnet(image_path, config)
            _, prob = run_resnet(_resnet_model, tensor, config)
            result["resnet_probability"] = prob
            result["gradcam_image"] = gradcam_utils.to_base64(
                _resnet_model, tensor, image_path,
                _frac_idx, config["device"],
                layer_name=config.get("gradcam_layer", "layer4"),
            )
        else:
            # No ResNet yet — use box overlay as the GradCAM slot placeholder
            result["gradcam_image"] = result["xray_with_box"]

    else:
        # ----------------------------------------------------------------
        # Step 2b — CLASSIFIER-LED: YOLO found nothing
        # ----------------------------------------------------------------
        result["mode"] = "CLASSIFIER-LED"

        if _resnet_loaded:
            tensor = preprocess_resnet(image_path, config)
            label, prob = run_resnet(_resnet_model, tensor, config)
            result["label"]                = label
            result["resnet_probability"]   = prob
            result["fracture_probability"] = prob
            result["gradcam_image"] = gradcam_utils.to_base64(
                _resnet_model, tensor, image_path,
                _frac_idx, config["device"],
                layer_name=config.get("gradcam_layer", "layer4"),
            )
        else:
            # YOLO-only mode, no box found → default to Non-Fractured, show image
            result["label"]       = "Non-Fractured"
            result["gradcam_image"] = _image_to_base64(image_path)

    return result
