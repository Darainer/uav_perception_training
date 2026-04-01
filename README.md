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
│   └── splits/         # train/val/test JSON + images
├── configs/
│   ├── train.yaml      # Training hyperparameters
│   └── dataset.yaml    # Dataset paths and class config
├── scripts/
│   ├── download_m3ot.py        # Dataset download helper
│   ├── prepare_dataset.py      # M3OT → COCO conversion
│   ├── train.py                # Fine-tuning entrypoint
│   ├── evaluate.py             # mAP evaluation
│   └── export.py               # ONNX → TensorRT FP16 export
├── notebooks/
│   └── explore_dataset.ipynb   # EDA and sanity checks
├── models/
│   └── checkpoints/            # Saved .pth files (gitignored)
├── exports/                    # ONNX and .engine files (gitignored)
└── logs/                       # Tensorboard / wandb logs
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and prepare M3OT dataset
python scripts/download_m3ot.py --output data/raw
python scripts/prepare_dataset.py --input data/raw --output data/splits

# 3. Train
python scripts/train.py --config configs/train.yaml

# 4. Evaluate
python scripts/evaluate.py --checkpoint models/checkpoints/best.pth

# 5. Export for deployment
python scripts/export.py --checkpoint models/checkpoints/best.pth --output exports/
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
