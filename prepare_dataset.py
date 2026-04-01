"""
prepare_dataset.py

Converts M3OT dataset to COCO format for RF-DETR fine-tuning.

M3OT structure (from figshare):
    M3OT/
    ├── RGB/
    │   ├── sequence_001/
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   └── ...
    ├── IR/
    │   ├── sequence_001/
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   └── ...
    └── annotations/
        ├── sequence_001.txt    # MOT format: frame,id,x,y,w,h,conf,class,vis
        └── ...

Output COCO format:
    data/splits/
    ├── train.json
    ├── val.json
    ├── test.json
    └── images/
        ├── train/
        ├── val/
        └── test/

Usage:
    python scripts/prepare_dataset.py --input data/raw/M3OT --output data/splits
    python scripts/prepare_dataset.py --input data/raw/M3OT --output data/splits --modality IR
"""

import argparse
import json
import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

import yaml
import cv2
from tqdm import tqdm


# M3OT class label → our class index
# Inspect your actual M3OT annotation files and adjust if labels differ
M3OT_CLASS_MAP = {
    "pedestrian": 0,   # person
    "person":     0,
    "car":        1,   # vehicle
    "truck":      1,
    "bus":        1,
    "van":        1,
    "freight_car": 1,
}

CATEGORIES = [
    {"id": 0, "name": "person",  "supercategory": "living"},
    {"id": 1, "name": "vehicle", "supercategory": "object"},
    {"id": 2, "name": "animal",  "supercategory": "living"},
]

MIN_BOX_AREA = 16       # 4x4 pixels minimum
MIN_VISIBILITY = 0.3    # drop heavily occluded instances


def parse_mot_annotation(ann_file: Path) -> list[dict]:
    """Parse MOT-format annotation file.
    
    MOT format: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    Class field may be an integer or string depending on M3OT version.
    
    Returns list of dicts with keys: frame, id, bbox (xywh), conf, class_name, visibility
    """
    annotations = []
    if not ann_file.exists():
        return annotations

    with open(ann_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                frame = int(parts[0])
                obj_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6]) if len(parts) > 6 else 1.0
                cls = parts[7].strip().lower() if len(parts) > 7 else "pedestrian"
                vis = float(parts[8]) if len(parts) > 8 else 1.0

                # Try numeric class if string fails
                if cls.isdigit():
                    int_cls = int(cls)
                    # M3OT may use: 1=pedestrian, 2=car, etc. — adjust if needed
                    cls = {1: "pedestrian", 2: "car", 3: "truck", 4: "bus", 5: "van"}.get(int_cls, "pedestrian")

                annotations.append({
                    "frame": frame,
                    "id": obj_id,
                    "bbox": [x, y, w, h],
                    "conf": conf,
                    "class_name": cls,
                    "visibility": vis,
                })
            except (ValueError, IndexError):
                continue

    return annotations


def build_coco_dataset(sequences: list[Path], image_root: Path, modality: str = "RGB") -> dict:
    """Build a COCO-format dict from a list of M3OT sequence paths."""
    coco = {
        "info": {"description": "M3OT aerial EO dataset for RF-DETR fine-tuning"},
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }

    image_id = 0
    ann_id = 0

    for seq_path in tqdm(sequences, desc="Processing sequences"):
        seq_name = seq_path.name
        img_dir = image_root / modality / seq_name
        ann_file = image_root / "annotations" / f"{seq_name}.txt"

        if not img_dir.exists():
            print(f"  [WARN] Image dir not found: {img_dir}")
            continue

        annotations = parse_mot_annotation(ann_file)

        # Group annotations by frame
        frame_anns = defaultdict(list)
        for ann in annotations:
            frame_anns[ann["frame"]].append(ann)

        # Get sorted image list
        image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

        for img_path in image_files:
            # Extract frame number from filename
            try:
                frame_num = int(img_path.stem)
            except ValueError:
                continue

            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            image_id += 1
            coco["images"].append({
                "id": image_id,
                "file_name": str(img_path),   # Absolute path — simplest for training
                "width": w,
                "height": h,
                "sequence": seq_name,
                "frame": frame_num,
            })

            # Add annotations for this frame
            for ann in frame_anns.get(frame_num, []):
                class_name = ann["class_name"]
                cat_id = M3OT_CLASS_MAP.get(class_name)
                if cat_id is None:
                    continue  # Unknown class — skip

                x, y, bw, bh = ann["bbox"]
                area = bw * bh

                if area < MIN_BOX_AREA:
                    continue
                if ann["visibility"] < MIN_VISIBILITY:
                    continue

                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x, y, bw, bh],
                    "area": area,
                    "iscrowd": 0,
                    "visibility": ann["visibility"],
                })

    return coco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    type=str, required=True, help="Path to M3OT root dir")
    parser.add_argument("--output",   type=str, required=True, help="Output splits dir")
    parser.add_argument("--modality", type=str, default="RGB", choices=["RGB", "IR"])
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover sequences — split by sequence to prevent frame leakage
    modality_dir = input_root / args.modality
    if not modality_dir.exists():
        raise FileNotFoundError(f"Modality dir not found: {modality_dir}")

    sequences = sorted([p for p in modality_dir.iterdir() if p.is_dir()])
    print(f"Found {len(sequences)} sequences in {modality_dir}")

    # Shuffle and split
    random.shuffle(sequences)
    n = len(sequences)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train_seqs = sequences[:n_train]
    val_seqs   = sequences[n_train:n_train + n_val]
    test_seqs  = sequences[n_train + n_val:]

    print(f"Split: {len(train_seqs)} train / {len(val_seqs)} val / {len(test_seqs)} test sequences")

    for split_name, seqs in [("train", train_seqs), ("val", val_seqs), ("test", test_seqs)]:
        print(f"\nBuilding {split_name} split...")
        coco = build_coco_dataset(seqs, input_root, modality=args.modality)

        out_file = output_dir / f"{split_name}.json"
        with open(out_file, "w") as f:
            json.dump(coco, f)

        n_images = len(coco["images"])
        n_anns   = len(coco["annotations"])
        print(f"  → {out_file}: {n_images} images, {n_anns} annotations")

    print("\nDone. Dataset ready for training.")
    print(f"  train: {output_dir}/train.json")
    print(f"  val:   {output_dir}/val.json")
    print(f"  test:  {output_dir}/test.json")


if __name__ == "__main__":
    main()
