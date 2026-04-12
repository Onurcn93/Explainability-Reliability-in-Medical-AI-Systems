"""
inference/predict.py — FracAssist inference module.

Selective ensemble logic:
  YOLO-LED    : YOLO fires a box (conf >= 0.25) → fracture confirmed by detector.
  CLASSIFIER-LED : YOLO no box → ResNet-18 decides (if loaded); else defaults Non-Fractured.

GradCAM always generated on the ResNet-18 branch when ResNet is loaded.

ResNet-18 weights are optional at startup — the module degrades gracefully if
resnet18_e4e.pth is not yet present (YOLO-only mode until weights are placed).
"""

import base64
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Module-level model handles — loaded once at startup, reused per request
# ---------------------------------------------------------------------------
_yolo_model = None
_resnet_model = None
_resnet_loaded = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(config):
    """Load YOLO and (optionally) ResNet-18 from paths in config.
    Called once at app startup. Raises FileNotFoundError if YOLO weights missing."""
    global _yolo_model, _resnet_model, _resnet_loaded

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
        m = models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 1))
        m.load_state_dict(torch.load(resnet_path, map_location=device))
        m.eval()
        _resnet_model = m.to(device)
        _resnet_loaded = True
        print(f"[INFO] ResNet-18 loaded: {resnet_path} (device={device})")
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
    """Val/test transform pipeline — no augmentation, no random crop.
    Returns a (1, 3, 224, 224) float tensor."""
    transform = transforms.Compose([
        transforms.Resize(config["resnet_resize"]),
        transforms.CenterCrop(config["resnet_input_size"]),
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
    """Forward pass with sigmoid + fixed threshold 0.425.
    Returns (label: str, probability: float)."""
    device = config["device"]
    with torch.no_grad():
        logit = model(tensor.to(device))
        prob = float(torch.sigmoid(logit))
    label = "Fractured" if prob >= config["resnet_threshold"] else "Non-Fractured"
    return label, prob


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

def generate_gradcam(model, tensor, image_path, device):
    """GradCAM on model.layer4[-1].

    Steps:
      1. Register forward hook  → capture activations
      2. Register backward hook → capture gradients
      3. Forward pass → backward pass (no torch.no_grad here — needs grad graph)
      4. Global-average-pool gradients → weights
      5. Weighted sum of activations → ReLU → normalise
      6. Resize, apply JET colormap, blend alpha=0.5 with original
      7. Remove hooks in finally block (prevents memory leak)

    Returns base64-encoded PNG string (data:image/png;base64,...).
    """
    activations = {}
    gradients = {}

    target_layer = model.layer4[-1]

    def _fwd_hook(module, inp, out):
        activations["feat"] = out.detach().clone()

    def _bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach().clone()

    h_fwd = target_layer.register_forward_hook(_fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(_bwd_hook)

    try:
        model.zero_grad()
        t = tensor.to(device)
        output = model(t)           # (1, 1), has grad_fn via model params
        output.squeeze().backward() # scalar backward — computes all gradients

        act  = activations["feat"].squeeze(0)   # (C, H, W)
        grad = gradients["feat"].squeeze(0)     # (C, H, W)
        weights = grad.mean(dim=[1, 2])         # (C,)  — global avg pool

        cam = torch.relu((weights[:, None, None] * act).sum(0))  # (H, W)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam_np = cam.cpu().numpy()

        # Read original, resize to 224×224 for overlay
        orig = cv2.imread(image_path)
        if orig is None:
            orig = np.zeros((224, 224, 3), dtype=np.uint8)
        orig_resized = cv2.resize(orig, (224, 224))

        cam_u8  = np.uint8(255 * cv2.resize(cam_np, (224, 224)))
        heatmap = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(orig_resized, 0.5, heatmap, 0.5, 0)

        return _encode_base64(blended)

    finally:
        h_fwd.remove()
        h_bwd.remove()
        model.zero_grad()  # Clean up accumulated parameter gradients


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _encode_base64(img_bgr):
    """Encode a BGR numpy image as a data:image/png;base64,... string."""
    _, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def _image_to_base64(image_path):
    """Read an image file and return it as a base64 PNG string."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return _encode_base64(img)


def _draw_bbox_base64(image_path, bbox, confidence):
    """Draw YOLO bounding box on the original image, return base64 PNG."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    red_bgr = (0, 91, 255)  # BGR red matching the UI's --fa-red
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
            result["gradcam_image"] = generate_gradcam(
                _resnet_model, tensor, image_path, config["device"]
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
            result["gradcam_image"] = generate_gradcam(
                _resnet_model, tensor, image_path, config["device"]
            )
        else:
            # YOLO-only mode, no box found → default to Non-Fractured, show image
            result["label"]       = "Non-Fractured"
            result["gradcam_image"] = _image_to_base64(image_path)

    return result
