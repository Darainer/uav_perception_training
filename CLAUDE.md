# CLAUDE.md — uav_perception_training

Context handoff from Claude.ai session. Drop this file in the repo root and Claude Code will pick it up automatically.

---

## What this repo is

Fine-tuning RF-DETR-Small (Roboflow, DINOv2 ViT backbone) on aerial EO imagery for UAV-based object detection. Output is a TensorRT FP16 engine that drops directly into an existing ROS2 rfdetr_node.

**Companion repo:** github.com/Darainer/drone_autonomy_platform
The trained engine deploys to that repo's perception node via:
`config/rfdetr.yaml → engine_path: /home/dev/models/RF-DETR-SMALL.engine`

---

## Architecture decisions made

**Phase 1: EO only (current)**
RF-DETR-Small takes standard 3-channel RGB input. No modifications to the backbone needed. Gets the pipeline working end-to-end quickly.

**Phase 2: Dual-encoder EO+IR fusion (planned)**
Separate encoders per modality — do NOT use DJI's pre-fused stream. Reason: DJI bakes factory boresight calibration into the fused output, which makes the model platform-specific and non-portable. With separate encoders, only the fusion layer needs retraining when the sensor changes.

**Dataset: M3OT (figshare)**
- 10,790 paired RGB+IR frames, 220k bounding boxes
- UAV altitude 100–120m AGL
- Annotations: pedestrian, car, truck, bus, van, freight_car
- Split by video sequence (not frame) to prevent leakage

**Classes: 3**
- 0: person (M3OT: pedestrian)
- 1: vehicle (M3OT: car, truck, bus, van, freight_car)
- 2: animal/fawn — NO M3OT labels for this class yet

**Animal class gap:** M3OT has no fawn/wildlife annotations. Options:
- Reduce to 2 classes for initial training (recommended)
- Add BIRDSAI dataset for animal class later
- Fine-tune further on own M4TD footage once collected

---

## Hardware context

| Role | Hardware |
|------|----------|
| Training | RTX 3060 12GB |
| Inference target | NVIDIA Orin Nano 8GB |
| Drone platform | DJI M4TD (planned purchase) |
| Current drone | DJI M3T |

**Batch size:** 8 at 512×512 FP16 on 3060, effective batch 16 with grad_accum_steps=2

---

## Project structure

```
uav_perception_training/
├── CLAUDE.md               ← you are here
├── README.md
├── requirements.txt
├── configs/
│   ├── train.yaml          # Hyperparameters, paths, logging
│   └── dataset.yaml        # Class mapping, split ratios, filters
├── scripts/
│   ├── prepare_dataset.py  # M3OT → COCO format conversion
│   ├── train.py            # Fine-tuning entrypoint (rfdetr package)
│   ├── evaluate.py         # mAP eval with per-class breakdown
│   └── export.py           # ONNX → TensorRT FP16 export
├── data/
│   ├── raw/                # M3OT raw download (gitignored)
│   ├── processed/          # Intermediate (gitignored)
│   └── splits/             # train.json, val.json, test.json (gitignored)
├── models/checkpoints/     # .pth files (gitignored)
├── exports/                # .onnx and .engine files (gitignored)
└── logs/                   # Tensorboard logs (gitignored)
```

---

## Execution order

```bash
# 1. Install
pip install -r requirements.txt

# 2. Inspect M3OT annotation format BEFORE running prepare_dataset.py
# Open one .txt file in data/raw/M3OT/annotations/ and verify:
# - Column order: frame, id, x, y, w, h, conf, class, visibility
# - Class labels: string or integer? (adjust M3OT_CLASS_MAP in prepare_dataset.py)

# 3. Convert dataset
python scripts/prepare_dataset.py --input data/raw/M3OT --output data/splits

# 4. Train
python scripts/train.py --config configs/train.yaml

# 5. Evaluate
python scripts/evaluate.py --checkpoint models/checkpoints/best.pth

# 6. Export ONNX (on 3060)
python scripts/export.py --checkpoint models/checkpoints/best.pth --onnx_only

# 7. Build TensorRT engine ON THE ORIN (not on 3060 — engines are platform-specific)
# Copy uav_perception_training.onnx to Orin, then:
# trtexec --onnx=uav_perception_training.onnx --saveEngine=RF-DETR-SMALL.engine --fp16 --workspace=4096
```

---

## Known issues / TODOs

- [ ] Verify M3OT annotation column format before running prepare_dataset.py
- [ ] Decide: keep 3 classes or reduce to 2 (drop animal) for initial training
- [ ] Check whether M3OT figshare download provides RGB and IR as separate dirs or interleaved
- [ ] prepare_dataset.py uses absolute image paths in COCO JSON — verify rfdetr package handles this
- [ ] Add download_m3ot.py script (figshare URL: https://figshare.com/s/01fa8d1163f4e9a5a13a)
- [ ] Notebook: explore_dataset.ipynb for EDA and annotation sanity checks
- [ ] Phase 2: dual-encoder EO+IR architecture (separate streams, cross-attention fusion)
- [ ] Phase 3: fawn class — needs separate dataset (BIRDSAI or own M4TD captures)

---

## Background context (from design session)

**Why separate EO+IR encoders (Phase 2 design):**
DJI's fused output stream bakes factory boresight calibration into the image, encoding the EO/IR spatial relationship implicitly in any model trained on it. This makes the model non-portable across platforms. Separate encoders isolate the platform-specific knowledge to the fusion layer only, which is smaller and cheaper to retrain.

**Deployment chain:**
```
RTX 3060 training → ONNX → Orin Nano TensorRT build → RF-DETR-SMALL.engine
→ drone_autonomy_platform/config/rfdetr.yaml picks it up automatically
```
