# uav_perception_training

Fine-tuning RF-DETR-Small on aerial EO imagery for UAV-based detection.

## Model
- **Architecture:** RF-DETR-Small (Roboflow, Apache 2.0)
- **Backbone:** DINOv2 ViT
- **Input:** 512×512, NMS-free
- **Baseline mAP:** 53.0 (COCO val2017)
- **Target classes:** person, vehicle, animal (fawn)

## Dataset
- **Source:** M3OT — Multi-Drone Multi-Modality dataset
- **Modality:** EO (RGB) — Phase 1. IR fusion in Phase 2.
- **Scale:** ~10,790 paired frames, 220k bounding boxes
- **Altitude:** 100–120m AGL
- **Conditions:** daytime, dusk, night, urban, suburban

## Pipeline
```
M3OT raw → preprocess → COCO format → fine-tune RF-DETR → ONNX → TensorRT FP16 → ROS2 node
```

## Project Structure
```
uav_perception_training/
├── data/
│   ├── raw/            # M3OT raw download (gitignored)
│   ├── processed/      # Converted to COCO format
│   └── splits/         # train/valid/test with _annotations.coco.json
├── configs/
│   ├── train.yaml      # Training hyperparameters
│   └── dataset.yaml    # Dataset paths and class config
├── scripts/
│   ├── download_m3ot.py        # TODO: Dataset download helper
│   ├── prepare_dataset.py      # M3OT → COCO conversion
│   ├── train.py                # Fine-tuning entrypoint
│   ├── evaluate.py             # mAP evaluation
│   └── export.py               # ONNX → TensorRT FP16 export
├── notebooks/
│   └── explore_dataset.ipynb   # TODO: EDA and annotation sanity checks
├── models/
│   └── checkpoints/            # Saved .pth files (gitignored)
├── exports/                    # ONNX and .engine files (gitignored)
└── logs/                       # Tensorboard / wandb logs
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download M3OT dataset (TODO: download_m3ot.py not yet implemented)
# Manual download: https://figshare.com/s/01fa8d1163f4e9a5a13a → data/raw/M3OT/

# 3. Prepare dataset (reads class map, splits, filters from configs/dataset.yaml)
python scripts/prepare_dataset.py --config configs/dataset.yaml

# 4. Train
python scripts/train.py --config configs/train.yaml

# 5. Evaluate
python scripts/evaluate.py --checkpoint models/checkpoints/checkpoint_best_total.pth

# 6. Export for deployment
python scripts/export.py --checkpoint models/checkpoints/checkpoint_best_total.pth --output exports/
```

## Deployment
Copy the exported engine to your drone platform:
```bash
scp exports/uav_perception_training.engine user@orin:/home/dev/models/RF-DETR-SMALL.engine
```
The existing ROS2 rfdetr_node picks it up automatically via `config/rfdetr.yaml`.

## Hardware
- **Training:** RTX 3060 12GB, FP16 mixed precision
- **Inference target:** NVIDIA Orin Nano 8GB, TensorRT FP16
