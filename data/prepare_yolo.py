"""
data/prepare_yolo.py

Builds the YOLO dataset folder structure from FracAtlas source files.

Detection mode (default):
    Reads official Fracture Split CSVs and copies images + YOLO bbox labels.
    Output: data/dataset_yolo/

Segmentation mode (--seg):
    Same splits, but writes polygon labels from COCO JSON instead of bbox labels.
    Output: data/dataset_yolo_seg/

Negative sampling (--n_neg):
    Optionally adds non-fractured images (empty labels) to the training split.
    --n_neg -1  includes all 3,366 non-fractured images (default SOTA practice)
    --n_neg 0   fractured-only (original behaviour)
    --n_neg N   random sample of N non-fractured images (seed=42)

Test-set inclusion (--include_test):
    Appends test.csv images to the training split — replicating the author notebook
    behaviour (Abedeen et al. notebook train: 635 detect / 608 seg).
    NOTE: this leaks the official test split into training. Use only for Y0B
    reproduction. In seg mode, only test images with COCO polygon annotations
    are included (not all 61 have polygon labels).

Usage:
    python data/prepare_yolo.py                                      # detection, no negatives
    python data/prepare_yolo.py --n_neg -1                           # detection + all negatives
    python data/prepare_yolo.py --seg --n_neg -1                     # segmentation + all negatives
    python data/prepare_yolo.py --n_neg -1 --clean                   # wipe and rebuild
    python data/prepare_yolo.py --include_test --out_dir data/dataset_yolo_Y0B
    python data/prepare_yolo.py --seg --include_test --out_dir data/dataset_yolo_seg_Y0B
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


SPLITS = ["train", "valid"]
SPLIT_CSV = {"train": "train.csv", "valid": "valid.csv", "test": "test.csv"}


def read_split_csv(csv_path: Path) -> list:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row["image_id"] for row in reader]


def prepare_split(
    split: str,
    image_ids: list,
    images_src: Path,
    labels_src: Path,
    out_dir: Path,
) -> None:
    img_dst = out_dir / split / "images"
    lbl_dst = out_dir / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    missing_images = []
    missing_labels = []

    for image_id in image_ids:
        src_img = images_src / image_id
        if src_img.exists():
            shutil.copy2(src_img, img_dst / image_id)
        else:
            missing_images.append(image_id)

        label_id = Path(image_id).stem + ".txt"
        src_lbl = labels_src / label_id
        if src_lbl.exists():
            shutil.copy2(src_lbl, lbl_dst / label_id)
        else:
            missing_labels.append(label_id)

    print(f"[data] {split:>5}: {len(image_ids) - len(missing_images)}/{len(image_ids)} images copied")

    if missing_images:
        print(f"[warn] {split}: {len(missing_images)} images not found: {missing_images[:5]}")
    if missing_labels:
        print(f"[warn] {split}: {len(missing_labels)} labels not found: {missing_labels[:5]}")


def write_data_yaml(out_dir: Path) -> None:
    content = (
        "train: train/images\n"
        "val:   valid/images\n"
        "\n"
        "nc: 1\n"
        'names: ["fractured"]\n'
    )
    path = out_dir / "data.yaml"
    path.write_text(content)
    print(f"[data] Written {path}")


def load_coco_seg_labels(coco_json_path: Path) -> dict:
    """Parse COCO JSON and return filename -> list of YOLO seg label strings.

    YOLO seg format per line: class_id x1 y1 x2 y2 ... xn yn  (normalized 0-1)
    One line per annotation polygon.
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    id_to_img = {img["id"]: img for img in coco["images"]}

    filename_to_labels: dict = {}
    for ann in coco["annotations"]:
        img  = id_to_img[ann["image_id"]]
        W, H = img["width"], img["height"]
        fname = img["file_name"]

        for polygon in ann["segmentation"]:
            if len(polygon) < 6:  # need at least 3 points
                continue
            coords = []
            for i in range(0, len(polygon), 2):
                coords.append(f"{polygon[i] / W:.6f}")
                coords.append(f"{polygon[i + 1] / H:.6f}")
            filename_to_labels.setdefault(fname, []).append("0 " + " ".join(coords))

    return filename_to_labels


def prepare_seg_split(
    split: str,
    image_ids: list,
    images_src: Path,
    filename_to_labels: dict,
    out_dir: Path,
) -> None:
    img_dst = out_dir / split / "images"
    lbl_dst = out_dir / split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    missing_images  = []
    missing_labels  = []

    for image_id in image_ids:
        src_img = images_src / image_id
        if src_img.exists():
            shutil.copy2(src_img, img_dst / image_id)
        else:
            missing_images.append(image_id)

        label_id = Path(image_id).stem + ".txt"
        lines = filename_to_labels.get(image_id, [])
        (lbl_dst / label_id).write_text("\n".join(lines) + "\n" if lines else "")
        if not lines:
            missing_labels.append(image_id)

    print(f"[data] {split:>5}: {len(image_ids) - len(missing_images)}/{len(image_ids)} images copied")
    if missing_images:
        print(f"[warn] {split}: {len(missing_images)} images not found: {missing_images[:5]}")
    if missing_labels:
        print(f"[warn] {split}: {len(missing_labels)} images had no COCO polygon annotation")


def add_negatives(
    n_neg: int,
    non_frac_images_src: Path,
    labels_src: Path,          # YOLO bbox labels dir (for detection); None for seg
    out_dir: Path,
    seed: int = 42,
) -> None:
    """Copy non-fractured images (+ empty labels) into train split.

    Args:
        n_neg:              Number of negatives to add. -1 = all available.
        non_frac_images_src: FracAtlas/images/Non_fractured/
        labels_src:         FracAtlas/Annotations/YOLO/ for detection (has empty .txt files).
                            Pass None for segmentation (empty files written directly).
        out_dir:            Dataset output root (e.g. data/dataset_yolo/).
        seed:               Random seed for reproducible sampling when n_neg > 0.
    """
    all_images = sorted(non_frac_images_src.glob("*.jpg"))
    if n_neg == -1:
        selected = all_images
    else:
        rng = random.Random(seed)
        selected = rng.sample(all_images, min(n_neg, len(all_images)))

    img_dst = out_dir / "train" / "images"
    lbl_dst = out_dir / "train" / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for img_path in selected:
        shutil.copy2(img_path, img_dst / img_path.name)
        label_name = img_path.stem + ".txt"
        if labels_src is not None:
            src_lbl = labels_src / label_name
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_dst / label_name)
                continue
        # Segmentation mode or missing label — write empty file
        (lbl_dst / label_name).write_text("")

    n_added = len(selected)
    print(f"[data] train: +{n_added} non-fractured images added (empty labels)")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from FracAtlas")
    parser.add_argument(
        "--fracatlas_root",
        type=str,
        default=None,
        help="Path to FracAtlas folder. Defaults to <repo_root>/FracAtlas",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to dataset_yolo or dataset_yolo_seg",
    )
    parser.add_argument(
        "--seg",
        action="store_true",
        help="Build segmentation dataset (polygon labels from COCO JSON)",
    )
    parser.add_argument(
        "--n_neg",
        type=int,
        default=0,
        help="Non-fractured images to add to training. -1=all, 0=none (default), N=random sample",
    )
    parser.add_argument(
        "--include_test",
        action="store_true",
        help=(
            "Append test.csv images to the training split. Replicates the author "
            "notebook behaviour (Y0B reproduction). In seg mode only test images "
            "with COCO polygon annotations are included."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove and rebuild the output directory from scratch",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    fracatlas_root = Path(args.fracatlas_root) if args.fracatlas_root else repo_root / "FracAtlas"

    default_out = "dataset_yolo_seg" if args.seg else "dataset_yolo"
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "data" / default_out

    if not fracatlas_root.exists():
        raise FileNotFoundError(
            f"FracAtlas not found at: {fracatlas_root}\n"
            "Place the dataset at the repo root — see README for setup instructions."
        )

    images_src     = fracatlas_root / "images" / "Fractured"
    non_frac_src   = fracatlas_root / "images" / "Non_fractured"
    splits_src     = fracatlas_root / "Utilities" / "Fracture Split"

    if args.seg:
        coco_json = fracatlas_root / "Annotations" / "COCO JSON" / "COCO_fracture_masks.json"
        for p in [images_src, splits_src, coco_json]:
            if not p.exists():
                raise FileNotFoundError(f"Expected FracAtlas path not found: {p}")
    else:
        labels_src = fracatlas_root / "Annotations" / "YOLO"
        for p in [images_src, labels_src, splits_src]:
            if not p.exists():
                raise FileNotFoundError(f"Expected FracAtlas path not found: {p}")

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"[data] Cleaned {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    mode    = "segmentation" if args.seg else "detection"
    neg_str = "all" if args.n_neg == -1 else str(args.n_neg)
    print(f"[data] Mode         : {mode}")
    print(f"[data] Negatives    : {neg_str} non-fractured images in train")
    print(f"[data] Include test : {args.include_test}")
    print(f"[data] FracAtlas    : {fracatlas_root}")
    print(f"[data] Output       : {out_dir}")
    print()

    if args.seg:
        print(f"[data] Loading COCO polygon annotations...")
        filename_to_labels = load_coco_seg_labels(coco_json)
        print(f"[data] Found polygon annotations for {len(filename_to_labels)} images")
        print()
        for split in SPLITS:
            csv_path = splits_src / SPLIT_CSV[split]
            image_ids = read_split_csv(csv_path)
            if split == "train" and args.include_test:
                test_ids = read_split_csv(splits_src / SPLIT_CSV["test"])
                # Only add test images that have COCO polygon annotations
                test_ids_with_ann = [tid for tid in test_ids if tid in filename_to_labels]
                n_added = len(test_ids_with_ann)
                image_ids = image_ids + test_ids_with_ann
                print(f"[data] --include_test: added {n_added}/{len(test_ids)} test images with COCO annotations to train")
            prepare_seg_split(split, image_ids, images_src, filename_to_labels, out_dir)
        if args.n_neg != 0:
            add_negatives(args.n_neg, non_frac_src, None, out_dir)
    else:
        for split in SPLITS:
            csv_path = splits_src / SPLIT_CSV[split]
            image_ids = read_split_csv(csv_path)
            if split == "train" and args.include_test:
                test_ids = read_split_csv(splits_src / SPLIT_CSV["test"])
                image_ids = image_ids + test_ids
                print(f"[data] --include_test: added {len(test_ids)} test images to train")
            prepare_split(split, image_ids, images_src, labels_src, out_dir)
        if args.n_neg != 0:
            add_negatives(args.n_neg, non_frac_src, labels_src, out_dir)

    write_data_yaml(out_dir)

    print()
    print("[data] Done. Dataset ready at:", out_dir)


if __name__ == "__main__":
    main()