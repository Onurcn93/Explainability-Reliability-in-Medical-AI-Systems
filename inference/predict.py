"""
inference/predict.py — FracAssist inference module.

Primary mode: GEL (Gated Ensemble Logic)
  All four models run in parallel (YOLO + ResNet-18 + DenseNet-169 + EfficientNet-B3).
  BVG authenticates the YOLO bbox via classifier consensus.
  OAM penalises an outlier classifier when |p_i - mu| > delta (deviation from mean).
  PDWF fuses loaded classifiers with F1-performance weights.
  GEL adapts to 2 or 3 loaded classifiers automatically.
  GradCAM generated from DenseNet-169 denseblock4.

Additional modes: "yolo" (detector only), "resnet" (all classifiers, no YOLO).

Classifier weights are optional — the module degrades gracefully if checkpoints are absent.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tv_models, transforms
from ultralytics import YOLO

import utils.gradcam as gradcam_utils


def _imread(path: str):
    """cv2.imread that works on Windows paths containing spaces or Unicode."""
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


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

_efficientnet_model     = None
_efficientnet_loaded    = False
_efficientnet_frac_idx  = 0
_efficientnet_threshold = 0.5   # placeholder; overridden from checkpoint at load time


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(config):
    """Load YOLO, ResNet-18, DenseNet-169, and EfficientNet-B3 (classifiers optional/graceful).
    Called once at app startup. Raises FileNotFoundError if YOLO weights missing."""
    global _yolo_model
    global _resnet_model, _resnet_loaded, _resnet_frac_idx, _resnet_threshold
    global _densenet_model, _densenet_loaded, _densenet_frac_idx, _densenet_threshold
    global _efficientnet_model, _efficientnet_loaded, _efficientnet_frac_idx, _efficientnet_threshold

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
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 2)) if has_dropout else nn.Linear(512, 2)  # type: ignore[assignment]
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
        m.classifier = (  # type: ignore[assignment]
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

    # --- EfficientNet-B3 (optional — F1) ---
    efficientnet_path = config.get("efficientnet_weights", "")
    if efficientnet_path and os.path.exists(efficientnet_path):
        ckpt = torch.load(efficientnet_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state                   = ckpt["model_state_dict"]
            _efficientnet_frac_idx  = ckpt.get("frac_idx", 0)
            _efficientnet_threshold = ckpt.get("val_threshold", config.get("efficientnet_threshold", 0.5))
        else:
            state                   = ckpt
            _efficientnet_frac_idx  = 0
            _efficientnet_threshold = config.get("efficientnet_threshold", 0.5)

        # Detect dropout: Sequential(Dropout, Linear) → "classifier.1.weight"
        has_dropout = "classifier.1.weight" in state
        m = tv_models.efficientnet_b3(weights=None)
        in_feat = m.classifier[1].in_features  # 1536
        m.classifier = (
            nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 2))
            if has_dropout else nn.Linear(in_feat, 2)
        )
        m.load_state_dict(state)
        m.eval()
        _efficientnet_model  = m.to(device)
        _efficientnet_loaded = True
        print(
            f"[INFO] EfficientNet-B3 loaded: {efficientnet_path} "
            f"(threshold={_efficientnet_threshold:.3f}, dropout={has_dropout})"
        )
    else:
        print(f"[INFO] EfficientNet-B3 weights not found — will show as pending in UI.")
        _efficientnet_loaded = False

    return _yolo_model, _resnet_model, _densenet_model, _efficientnet_model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image_path, input_size, config):
    """Val/test transform — Resize, Grayscale 3ch, ToTensor, ImageNet norm.

    NOTE: No CLAHE here. Correct for E4a_m050 (ResNet), D1 (DenseNet), and F1
    (EfficientNet-B3), all trained without CLAHE. If weights are swapped to
    CAALMIX-trained models (E5/E6/E7), add CLAHETransform() as the first step —
    omitting it would cause silent input-distribution mismatch and accuracy degradation.
    """
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["imagenet_mean"], std=config["imagenet_std"]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # type: ignore[union-attr]


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
    assert _resnet_model is not None, "ResNet not loaded"
    device = config["device"]
    with torch.no_grad():
        logits = _resnet_model(tensor.to(device))
        prob   = float(torch.softmax(logits, dim=1)[0, _resnet_frac_idx])
    label = "Fractured" if prob >= _resnet_threshold else "Non-Fractured"
    return label, prob


def run_densenet(tensor, config):
    """DenseNet-169 forward pass. Returns (label, probability)."""
    assert _densenet_model is not None, "DenseNet not loaded"
    device = config["device"]
    with torch.no_grad():
        logits = _densenet_model(tensor.to(device))
        prob   = float(torch.softmax(logits, dim=1)[0, _densenet_frac_idx])
    label = "Fractured" if prob >= _densenet_threshold else "Non-Fractured"
    return label, prob


def run_efficientnet(tensor, config):
    """EfficientNet-B3 forward pass. Returns (label, probability)."""
    assert _efficientnet_model is not None, "EfficientNet not loaded"
    device = config["device"]
    with torch.no_grad():
        logits = _efficientnet_model(tensor.to(device))
        prob   = float(torch.softmax(logits, dim=1)[0, _efficientnet_frac_idx])
    label = "Fractured" if prob >= _efficientnet_threshold else "Non-Fractured"
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
# GEL — Gated Ensemble Logic
# ---------------------------------------------------------------------------

def _run_gel(probs_f1, config):
    """RC init -> OAM -> PDWF -> P_final -> BVG gate. Returns (p_final, gate_passed).

    OAM is Direction-Aware Asymmetric: a HIGH outlier (lone fracture signal) is penalised
    leniently (k_high=0.30) to preserve the fracture signal; a LOW outlier (lone no-fracture
    dissenter against a fracture consensus) is penalised aggressively (k_low=0.10) to protect
    against missed fractures. Both directions are clinically aligned — fracture signals are
    harder to suppress than no-fracture signals.

    BVG gate uses P_final — the fully OAM-adjusted ensemble probability — so the gate and the
    fracture probability shown to the clinician are derived from the same calibrated estimate.

    probs_f1: list of (probability, f1_weight) tuples — one per loaded classifier.
              Accepts 2 or 3 classifiers; logic is identical regardless of count.

    YOLO is intentionally excluded from PDWF: detection confidence is not
    a classification probability and cannot be meaningfully mixed with
    softmax outputs in a weighted sum.
    """
    tau    = config["gel_tau"]
    dlim   = config["gel_disagree_lim"]
    k_low  = config["gel_penalty_k_low"]   # 0.10 — aggressive: LOW outlier (no-frac dissenter)
    k_high = config["gel_penalty_k_high"]  # 0.30 — lenient:    HIGH outlier (lone fracture signal)

    # Step 2 — RC initialisation (F1-normalised classifier weights)
    total_f1 = sum(f1 for _, f1 in probs_f1)
    rcs = [f1 / total_f1 for _, f1 in probs_f1]

    # Step 3 — Direction-Aware Asymmetric OAM
    mu  = sum(p for p, _ in probs_f1) / len(probs_f1)
    rcs = [rc * (k_low if p < mu else k_high) if abs(p - mu) > dlim else rc
           for (p, _), rc in zip(probs_f1, rcs)]

    # Step 4 — Performance-Driven Weighted Fusion
    total_rc = sum(rcs)
    p_final  = sum(p * rc for (p, _), rc in zip(probs_f1, rcs)) / total_rc

    # Step 5 — BVG gate: authenticate YOLO bbox using P_final (post-OAM)
    gate_passed = p_final >= tau

    return p_final, gate_passed


# ---------------------------------------------------------------------------
# Top-level predict
# ---------------------------------------------------------------------------

def predict(image_path, config, inference_mode="gel"):
    """Orchestrate inference and return result dict.

    inference_mode:
      "gel"    — GEL: all models + BVG/OAM/PDWF (default).
      "yolo"   — YOLO only. Classifiers skipped entirely.
      "resnet" — All classifiers (ResNet-18 + DenseNet-169 + EfficientNet-B3). YOLO skipped.
    """
    result = {
        "mode":                       None,
        "label":                      "Non-Fractured",
        "fracture_probability":       0.0,
        "yolo_confidence":            None,
        "bbox":                       None,
        "resnet_probability":         None,
        "densenet_probability":       None,
        "efficientnet_probability":   None,
        "gel_gate_passed":            None,
        "gel_consensus":              None,
        "gradcam_image":              None,
        "xray_with_box":              None,
        "disclaimer": (
            "This prediction is provided to support radiologist review. "
            "Clinical judgment is required for diagnosis."
        ),
        "error": None,
    }

    # ----------------------------------------------------------------
    # YOLO-ONLY: run detector, skip classifiers
    # ----------------------------------------------------------------
    if inference_mode == "yolo":
        detections = run_yolo(_yolo_model, image_path, config)
        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            result["mode"]                 = "YOLO-ONLY"
            result["label"]                = "Fractured"
            result["yolo_confidence"]      = best["confidence"]
            result["fracture_probability"] = best["confidence"]
            result["bbox"]                 = best["bbox"]
            result["xray_with_box"]        = _draw_bbox_base64(
                image_path, best["bbox"], best["confidence"]
            )
            result["gradcam_image"] = result["xray_with_box"]
        else:
            result["mode"]          = "YOLO-ONLY"
            result["label"]         = "Non-Fractured"
            result["xray_with_box"] = _image_to_base64(image_path)
            result["gradcam_image"] = result["xray_with_box"]
        return result

    # ----------------------------------------------------------------
    # CLASSIFIER-ONLY: skip YOLO, run all classifiers
    # ----------------------------------------------------------------
    if inference_mode == "resnet":
        result["mode"]          = "CLASSIFIER-ONLY"
        result["xray_with_box"] = _image_to_base64(image_path)

        resnet_tensor       = _preprocess(image_path, config["resnet_input_size"],       config) if _resnet_loaded       else None
        densenet_tensor     = _preprocess(image_path, config["densenet_input_size"],     config) if _densenet_loaded     else None
        efficientnet_tensor = _preprocess(image_path, config["efficientnet_input_size"], config) if _efficientnet_loaded else None

        if _resnet_loaded:
            label, resnet_prob = run_resnet(resnet_tensor, config)
            result["label"]                = label
            result["resnet_probability"]   = resnet_prob
            result["fracture_probability"] = resnet_prob
        else:
            result["label"] = "Non-Fractured"

        if _densenet_loaded:
            _, densenet_prob = run_densenet(densenet_tensor, config)
            result["densenet_probability"] = densenet_prob
            result["gradcam_image"] = gradcam_utils.to_base64(
                _densenet_model, densenet_tensor, image_path,  # type: ignore[arg-type]
                _densenet_frac_idx, config["device"],
                layer_name=config.get("gradcam_layer", "features.denseblock4"),
            )
        else:
            result["gradcam_image"] = result["xray_with_box"]

        if _efficientnet_loaded:
            _, efficientnet_prob = run_efficientnet(efficientnet_tensor, config)
            result["efficientnet_probability"] = efficientnet_prob

        return result

    # ----------------------------------------------------------------
    # GEL: Gated Ensemble Logic (novel ensemble contribution)
    # All models always run. BVG gates the YOLO bbox.
    # P_final comes from classifier-only PDWF (2 or 3 classifiers).
    # ----------------------------------------------------------------
    if inference_mode == "gel":
        resnet_tensor       = _preprocess(image_path, config["resnet_input_size"],       config) if _resnet_loaded       else None
        densenet_tensor     = _preprocess(image_path, config["densenet_input_size"],     config) if _densenet_loaded     else None
        efficientnet_tensor = _preprocess(image_path, config["efficientnet_input_size"], config) if _efficientnet_loaded else None
        detections          = run_yolo(_yolo_model, image_path, config)

        # Raw classifier probabilities
        p_r = run_resnet(resnet_tensor,           config)[1] if _resnet_loaded       else None
        p_d = run_densenet(densenet_tensor,       config)[1] if _densenet_loaded     else None
        p_e = run_efficientnet(efficientnet_tensor, config)[1] if _efficientnet_loaded else None

        result["resnet_probability"]       = p_r
        result["densenet_probability"]     = p_d
        result["efficientnet_probability"] = p_e

        # YOLO detection (bbox + confidence, independent of gate)
        best = None
        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            result["yolo_confidence"] = best["confidence"]
            result["bbox"]            = best["bbox"]

        # Build (prob, f1_weight) list for all loaded classifiers
        probs_f1 = []
        if p_r is not None:
            probs_f1.append((p_r, config["gel_f1_resnet"]))
        if p_d is not None:
            probs_f1.append((p_d, config["gel_f1_densenet"]))
        if p_e is not None:
            probs_f1.append((p_e, config["gel_f1_efficientnet"]))

        if len(probs_f1) >= 2:
            # Full GEL: RC init -> OAM -> PDWF -> P_final -> BVG gate
            p_final, gate_passed = _run_gel(probs_f1, config)
            result["mode"]            = "GEL"
            result["gel_gate_passed"] = gate_passed
            result["gel_consensus"]   = round(p_final, 4)
        elif len(probs_f1) == 1:
            # Degraded: single classifier available, no OAM
            p_solo      = probs_f1[0][0]
            gate_passed = p_solo >= config["gel_tau"]
            p_final     = p_solo if gate_passed else 0.0
            result["mode"]            = "GEL-DEGRADED"
            result["gel_gate_passed"] = gate_passed
            result["gel_consensus"]   = round(p_solo, 4)
        else:
            # No classifiers — fall through to YOLO result
            result["mode"]  = "GEL-DEGRADED"
            p_final         = best["confidence"] if best else 0.0
            gate_passed     = best is not None
            result["gel_gate_passed"] = gate_passed
            result["gel_consensus"]   = None

        result["fracture_probability"] = p_final
        result["label"] = "Fractured" if p_final >= 0.5 else "Non-Fractured"

        # Authenticated bbox: YOLO fired AND BVG gate passed
        if best and gate_passed:
            result["xray_with_box"] = _draw_bbox_base64(image_path, best["bbox"], best["confidence"])
        else:
            result["xray_with_box"] = _image_to_base64(image_path)

        # GradCAM from DenseNet-169
        if _densenet_loaded:
            result["gradcam_image"] = gradcam_utils.to_base64(
                _densenet_model, densenet_tensor, image_path,  # type: ignore[arg-type]
                _densenet_frac_idx, config["device"],
                layer_name=config.get("gradcam_layer", "features.denseblock4"),
            )
        else:
            result["gradcam_image"] = result["xray_with_box"]

        return result

    return result
