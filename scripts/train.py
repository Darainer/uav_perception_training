"""
train.py

Fine-tunes RF-DETR-Small on aerial EO imagery using the rfdetr package.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --resume models/checkpoints/checkpoint_best_total.pth
"""

import argparse
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

    # Validate dataset directory exists
    dataset_dir = Path(d["dataset_dir"])
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset dir not found: {dataset_dir}\n"
            f"Run: python scripts/prepare_dataset.py first"
        )

    output_dir = Path(t["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    from rfdetr import RFDETRSmall

    print(f"UAV Perception Training — Fine-tuning")
    print(f"  Model:      {m['variant']}")
    print(f"  Classes:    {m['num_classes']}")
    print(f"  Epochs:     {t['epochs']}")
    print(f"  Batch size: {t['batch_size']} (effective: {t['batch_size'] * t['grad_accum_steps']})")
    print(f"  Resolution: {d['resolution']}")
    print(f"  Dataset:    {d['dataset_dir']}")
    print()

    model = RFDETRSmall(num_classes=m["num_classes"])

    model.train(
        dataset_dir=str(dataset_dir),
        epochs=t["epochs"],
        batch_size=t["batch_size"],
        grad_accum_steps=t["grad_accum_steps"],
        lr=t["learning_rate"],
        lr_encoder=t["lr_encoder"],
        weight_decay=t["weight_decay"],
        resolution=d["resolution"],
        output_dir=str(output_dir),
        checkpoint_interval=t["checkpoint_interval"],
        early_stopping=t["early_stopping"],
        early_stopping_patience=t["early_stopping_patience"],
        device=args.device,
        tensorboard=cfg["logging"]["tensorboard"],
        use_wandb=cfg["logging"]["use_wandb"],
        project=cfg["logging"]["project"],
        resume=args.resume,
    )

    print(f"\nTraining complete. Best checkpoint: {output_dir}/checkpoint_best_total.pth")
    print(f"Next step: python scripts/export.py --checkpoint {output_dir}/checkpoint_best_total.pth")


if __name__ == "__main__":
    main()
