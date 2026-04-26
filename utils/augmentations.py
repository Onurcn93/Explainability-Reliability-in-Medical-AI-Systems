"""utils/augmentations.py — shared augmentation blocks for CAALMIX pipeline.

AlbumentationsDelta  (E6+): geometric + noise delta on top of torchvision pipeline.
XRayAugMix           (E7+): AugMix principle with X-ray-specific ops.

Both accept and return PIL Image — drop into transforms.Compose unchanged.
No cv2 objects stored at init time — safe for Windows multiprocessing workers.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import albumentations as A


class AlbumentationsDelta:
    """Delta augmentation block for CAALMIX E6+. Train-only.

    Drop into transforms.Compose after Grayscale(3ch) and before ToTensor.
    Ops: Affine (shift/scale/rotate), ElasticTransform, GaussNoise.
    """

    def __init__(
        self,
        p_shift:   float = 0.50,
        p_elastic: float = 0.30,
        p_noise:   float = 0.20,
    ):
        self._pipeline = A.Compose([
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                p=p_shift,
            ),
            A.ElasticTransform(p=p_elastic),
            A.GaussNoise(std_range=(0.01, 0.03), p=p_noise),
        ])

    def __call__(self, img: Image.Image) -> Image.Image:
        arr    = np.array(img, dtype=np.uint8)
        result = self._pipeline(image=arr)
        return Image.fromarray(result["image"])


class XRayAugMix:
    """AugMix for grayscale X-ray images. Train-only.

    Applies the AugMix mixing principle using ops chosen for X-ray acquisition
    variability rather than natural-image colour diversity:

      _clahe_vary  — CLAHE with random clip_limit (0.5–4.0): scanner contrast variation
      _gamma       — power-law intensity curve (gamma 0.5–2.0): exposure variation
      _contrast    — linear contrast factor (0.7–1.5): kVp/mAs variation
      _blur        — Gaussian blur (sigma 0.5–1.5): motion blur / low-res scan

    Mixing:
      1. Sample k augmented branches; weight by Dirichlet(alpha).
      2. Beta(alpha, alpha) weight m controls original vs augmented blend:
             output = m * original + (1 - m) * weighted_mix

    Drop into transforms.Compose after AlbumentationsDelta and before ToTensor.
    cv2.CLAHE is created lazily inside the op — picklable on Windows workers.
    """

    def __init__(
        self,
        mixture_width: int   = 3,
        alpha:         float = 1.0,
    ):
        self.mixture_width = mixture_width
        self.alpha         = alpha

    # ── X-ray augmentation ops ──────────────────────────────────────── #

    def _clahe_vary(self, img: Image.Image) -> Image.Image:
        clip  = np.random.uniform(0.5, 4.0)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        gray  = np.array(img.convert("L"), dtype=np.uint8)
        return Image.fromarray(clahe.apply(gray)).convert("RGB")

    def _gamma(self, img: Image.Image) -> Image.Image:
        gamma = np.random.uniform(0.5, 2.0)
        arr   = np.array(img, dtype=np.float32) / 255.0
        arr   = np.power(arr, gamma)
        return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

    def _contrast(self, img: Image.Image) -> Image.Image:
        factor = np.random.uniform(0.7, 1.5)
        return ImageEnhance.Contrast(img).enhance(factor)

    def _blur(self, img: Image.Image) -> Image.Image:
        sigma = np.random.uniform(0.5, 1.5)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # ── Augmentation chain (1–3 random ops per branch) ─────────────── #

    def _augment_branch(self, img: Image.Image) -> Image.Image:
        ops   = [self._clahe_vary, self._gamma, self._contrast, self._blur]
        depth = np.random.randint(1, 4)
        for idx in np.random.choice(len(ops), size=depth, replace=False):
            img = ops[idx](img)
        return img

    # ── AugMix forward pass ─────────────────────────────────────────── #

    def __call__(self, img: Image.Image) -> Image.Image:
        weights = np.random.dirichlet([self.alpha] * self.mixture_width).astype(np.float32)
        m       = float(np.random.beta(self.alpha, self.alpha))

        orig  = np.array(img, dtype=np.float32)
        mixed = np.zeros_like(orig, dtype=np.float32)
        for w in weights:
            mixed += w * np.array(self._augment_branch(img.copy()), dtype=np.float32)

        result = m * orig + (1.0 - m) * mixed
        return Image.fromarray(result.clip(0, 255).astype(np.uint8))
