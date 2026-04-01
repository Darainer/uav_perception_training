"""
export.py

Exports fine-tuned RF-DETR checkpoint to ONNX, then TensorRT FP16 engine.
The output .engine drops directly into the ROS2 rfdetr_node.

Usage:
    # ONNX + TensorRT (full pipeline)
    python scripts/export.py --checkpoint models/checkpoints/checkpoint_best_total.pth --output exports/

    # ONNX only (if running on training machine, TensorRT on Orin separately)
    python scripts/export.py --checkpoint models/checkpoints/checkpoint_best_total.pth --output exports/ --onnx_only

Deployment:
    scp exports/uav_perception_training.engine user@orin:/home/dev/models/RF-DETR-SMALL.engine
"""

import argparse
from pathlib import Path

import torch
import yaml


def export_onnx(model, output_path: Path, image_size: int = 512):
    """Export model to ONNX."""
    print(f"Exporting to ONNX: {output_path}")
    dummy_input = torch.randn(1, 3, image_size, image_size).cuda()

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["images"],
        output_names=["logits", "boxes"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "boxes":  {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    print(f"  ONNX export complete: {output_path}")


def simplify_onnx(onnx_path: Path) -> Path:
    """Run onnxsim to simplify graph."""
    try:
        import onnx
        from onnxsim import simplify

        model = onnx.load(str(onnx_path))
        model_simplified, check = simplify(model)
        if check:
            simplified_path = onnx_path.with_suffix(".simplified.onnx")
            onnx.save(model_simplified, str(simplified_path))
            print(f"  ONNX simplified: {simplified_path}")
            return simplified_path
        else:
            print("  [WARN] ONNX simplification failed — using original")
            return onnx_path
    except ImportError:
        print("  [WARN] onnxsim not installed — skipping simplification")
        return onnx_path


def export_tensorrt(onnx_path: Path, engine_path: Path, fp16: bool = True, workspace_gb: int = 4):
    """Convert ONNX to TensorRT engine.

    Note: TensorRT engines are platform-specific.
    Build on the Orin for deployment on the Orin.
    Build on the 3060 for local testing only.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("\n[INFO] TensorRT not available on this machine.")
        print("To build the engine on your Orin Nano:")
        print(f"  1. Copy {onnx_path} to the Orin")
        print(f"  2. Run: trtexec --onnx={onnx_path.name} --saveEngine=uav_perception_training.engine --fp16 --workspace={workspace_gb * 1024}")
        print(f"  3. Place engine at: /home/dev/models/RF-DETR-SMALL.engine")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled")

    print(f"  Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"  [ERROR] {parser.get_error(error)}")
            raise RuntimeError("ONNX parsing failed")

    print(f"  Building TensorRT engine (this takes several minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    print(f"  TensorRT engine: {engine_path}")
    print(f"  Copy to Orin: scp {engine_path} user@orin:/home/dev/models/RF-DETR-SMALL.engine")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--config",      type=str, default="configs/train.yaml")
    parser.add_argument("--output",      type=str, default="exports/")
    parser.add_argument("--onnx_only",   action="store_true")
    parser.add_argument("--no-fp16",     action="store_true", help="Disable FP16 for TensorRT")
    parser.add_argument("--workspace_gb", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["model"]["num_classes"]
    resolution = cfg["data"]["resolution"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    from rfdetr import RFDETRSmall

    model = RFDETRSmall(
        pretrain_weights=str(checkpoint_path),
        num_classes=num_classes,
    )
    model.eval().cuda()

    # ONNX export
    onnx_path = output_dir / "uav_perception_training.onnx"
    export_onnx(model, onnx_path, resolution)
    onnx_path = simplify_onnx(onnx_path)

    fp16 = not getattr(args, "no_fp16", False)
    if not args.onnx_only:
        engine_path = output_dir / "uav_perception_training.engine"
        export_tensorrt(onnx_path, engine_path, fp16=fp16, workspace_gb=args.workspace_gb)

    print("\nExport complete.")
    print(f"  ONNX:   {onnx_path}")
    if not args.onnx_only:
        print(f"  Engine: {output_dir / 'uav_perception_training.engine'}")


if __name__ == "__main__":
    main()
