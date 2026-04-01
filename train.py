"""
train.py

Fine-tunes RF-DETR-Small on aerial EO imagery using the rfdetr package.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --resume models/checkpoints/last.pth
"""

import argparse
import os
import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    m = cfg["model"]
    t = cfg["training"]
    d = cfg["data"]

    # Validate dataset files exist
    for split in ["train_json", "val_json"]:
        p = Path(d[split])
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {p}\n"
                f"Run: python scripts/prepare_dataset.py first"
            )

    Path(t["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Import here so missing deps fail loudly after config validation
    from rfdetr import RFDETRBase
    from rfdetr.util.coco_utils import get_coco_api_from_dataset

    print(f"UAV Perception Training Fine-tuning")
    print(f"  Model:      {m['variant']}")
    print(f"  Classes:    {m['num_classes']}")
    print(f"  Epochs:     {t['epochs']}")
    print(f"  Batch size: {t['batch_size']} (effective: {t['batch_size'] * t['grad_accum_steps']})")
    print(f"  Image size: {d['image_size']}")
    print(f"  Train:      {d['train_json']}")
    print(f"  Val:        {d['val_json']}")
    print()

    # Initialise model
    model = RFDETRBase(
        pretrain_weights="RF-DETR-B.pth",   # Downloads COCO pretrained weights
        num_classes=m["num_classes"],
        resolution=d["image_size"],
    )

    # RF-DETR uses its own training loop
    model.train(
        dataset_dir=None,                    # We pass json paths directly
        train_annotations=d["train_json"],
        val_annotations=d["val_json"],
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        grad_accum_steps=t["grad_accum_steps"],
        lr=t["learning_rate"],
        lr_encoder=t["lr_backbone"],
        weight_decay=t["weight_decay"],
        checkpoint_dir=t["checkpoint_dir"],
        use_wandb=cfg["logging"]["use_wandb"],
        project=cfg["logging"]["project"],
        resume=args.resume,
    )

    print(f"\nTraining complete. Best checkpoint: {t['checkpoint_dir']}/best.pth")
    print(f"Next step: python scripts/export.py --checkpoint {t['checkpoint_dir']}/best.pth")


if __name__ == "__main__":
    main()
