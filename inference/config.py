import os
import torch

# Repo root is one level above this file (inference/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    # Weight paths — place files in repo_root/weights/
    "yolo_weights":  os.path.join(_ROOT, "weights", "Y1B_detect_best.pt"),
    "resnet_weights": os.path.join(_ROOT, "weights", "resnet18_e4e.pth"),

    # YOLO inference — fixed from Y1B training
    "yolo_conf_threshold": 0.25,
    "yolo_iou_threshold":  0.5,
    "yolo_imgsz":          600,

    # ResNet-18 inference — fixed from E4e cosine warmup training
    "resnet_threshold":  0.425,
    "resnet_input_size": 224,
    "resnet_resize":     256,

    # GradCAM
    "gradcam_layer": "layer4",  # hooks model.layer4[-1]

    # Device: auto-upgrade to CUDA if available
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ImageNet normalisation (val_test_transforms)
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std":  [0.229, 0.224, 0.225],

    # Served static file
    "index_html": os.path.join(_ROOT, "index.html"),
}