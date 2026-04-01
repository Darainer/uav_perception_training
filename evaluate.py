"""
evaluate.py

Evaluates a fine-tuned RF-DETR checkpoint on the test split.
Reports per-class and overall mAP (AP50, AP75, AP50-95).

Usage:
    python scripts/evaluate.py --checkpoint models/checkpoints/best.pth
    python scripts/evaluate.py --checkpoint models/checkpoints/best.pth --split val
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


CLASSES = ["person", "vehicle", "animal"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--config",      type=str, default="configs/train.yaml")
    parser.add_argument("--split",       type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf_thresh", type=float, default=0.3)
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ann_file = cfg["data"][f"{args.split}_json"]
    if not Path(ann_file).exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    print(f"Loading checkpoint: {args.checkpoint}")
    from rfdetr import RFDETRBase

    model = RFDETRBase(num_classes=3, resolution=cfg["data"]["image_size"])
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval().to(args.device)

    print(f"Evaluating on {args.split} split: {ann_file}")
    coco_gt = COCO(ann_file)
    image_ids = coco_gt.getImgIds()

    results = []

    import cv2
    import numpy as np

    for img_id in tqdm(image_ids, desc="Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = img_info["file_name"]

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # RF-DETR inference
        with torch.no_grad():
            detections = model.predict(img_rgb, threshold=args.conf_thresh)

        # Convert to COCO results format
        for det in detections:
            # det: supervision Detection object
            for i in range(len(det.xyxy)):
                x1, y1, x2, y2 = det.xyxy[i]
                score = float(det.confidence[i])
                cat_id = int(det.class_id[i])

                results.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": score,
                })

    if not results:
        print("No detections — check confidence threshold or model loading.")
        return

    # COCO evaluation
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # Per-class breakdown
    print("\nPer-class AP50:")
    for class_idx, class_name in enumerate(CLASSES):
        evaluator.params.catIds = [class_idx]
        evaluator.evaluate()
        evaluator.accumulate()
        stats = evaluator.stats
        print(f"  {class_name:10s}: AP50={stats[1]:.3f}")


if __name__ == "__main__":
    main()
