# Design Decisions

## Why RF-DETR-Small

RF-DETR-Small (32M params, Roboflow) was chosen for three reasons:

1. **Orin Nano constraint.** The inference target has 8GB — Small is the largest RF-DETR variant that fits comfortably in TensorRT FP16 on this hardware.
2. **NMS-free.** RF-DETR uses learnt object queries instead of non-maximum suppression, which matters in dense aerial scenes where NMS thresholds are hard to tune (overlapping cars in parking lots, crowds at altitude).
3. **DINOv2 backbone.** Self-supervised ViT pretraining generalises well across domains. Aerial imagery is a significant distribution shift from COCO — a strong backbone reduces the amount of fine-tuning data needed.

The model is already deployed in the companion repo (`drone_autonomy_platform/rfdetr_node`), so the training output (ONNX -> TensorRT engine) drops in without pipeline changes.

## Dataset: M3OT

**Source:** Multi-Drone Multi-Object Tracking dataset from figshare.
~10,790 paired RGB+IR frames with 220k bounding boxes, captured at 100-120m AGL from multiple UAVs.

### Class mapping

M3OT annotates six fine-grained classes. We merge them into two operational classes (with a third placeholder):

| Our class | ID | M3OT source labels |
|-----------|----|--------------------|
| person    | 0  | pedestrian         |
| vehicle   | 1  | car, truck, bus, van, freight_car |
| animal    | 2  | *(none in M3OT)*   |

**Why merge vehicles?** At 100m+ AGL the operational question is "is there a vehicle" not "is it a bus or a van." Fine-grained vehicle classification at this altitude is unreliable and not useful for the current mission profiles (search, tracking, landing site assessment).

**Why keep the animal class?** Fawn detection during agricultural mowing is a Wallering Robotics use case. M3OT has no animal annotations, so this class is a placeholder. Options for populating it later:
- BIRDSAI dataset (wildlife from aerial thermal)
- Own M4TD captures once the platform is operational

For initial training, set `num_classes: 2` in `configs/train.yaml` and remove the animal entry from `configs/dataset.yaml` if you want a clean two-class model.

### Split strategy

Splits are done **by video sequence**, not by individual frame.

If you split by frame, consecutive frames from the same sequence end up in both train and val/test. These frames are near-identical (a few pixels of motion between frames at 30fps), so the model memorises appearance rather than learning to generalise. Validation metrics look great but the model fails on new footage.

Splitting by sequence ensures that no frames from a given flyover appear in more than one split. The ratios (configurable in `configs/dataset.yaml`):

| Split | Ratio | Purpose |
|-------|-------|---------|
| train | 0.70  | Model training |
| valid | 0.15  | Hyperparameter tuning, early stopping |
| test  | 0.15  | Final held-out evaluation only |

### Annotation filters

Two filters are applied during conversion (`configs/dataset.yaml`):

- **`min_box_area: 16`** (4x4 px) — At 100m+ AGL, objects below this size are annotation noise or ambiguous single-pixel dots. Training on them hurts precision.
- **`min_visibility: 0.3`** — M3OT includes a per-instance visibility/occlusion score. Heavily occluded instances (< 30% visible) are dropped because the bounding box no longer tightly fits the visible portion, which teaches the model bad localisation.

## Modality plan

**Phase 1 (current):** EO (RGB) only. Standard 3-channel input, no architecture changes needed.

**Phase 2 (planned):** Dual-encoder EO+IR fusion with separate encoder streams and a cross-attention fusion layer. Critically, we do NOT use DJI's pre-fused output stream — it bakes factory boresight calibration into the image, making any model trained on it non-portable across platforms. Separate encoders isolate the platform-specific knowledge to the fusion layer, which is small and cheap to retrain when the sensor changes.
