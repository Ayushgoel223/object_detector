"""
BlindAid - Model Export for Mobile / Edge Deployment
=======================================================
Exports the trained YOLOv8 model to:
  - ONNX      : Cross-platform (Windows, Linux, Android via ONNX Runtime)
  - TFLite    : Android / Raspberry Pi deployment
  - OpenVINO  : Intel CPU optimization (optional)

Usage:
    python models/export.py
    python models/export.py --weights yolov8n.pt --format onnx
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export_model(weights: str, format: str, img_size: int, optimize: bool):
    print("=" * 60)
    print("  BlindAid — Model Export")
    print("=" * 60)
    print(f"  Weights : {weights}")
    print(f"  Format  : {format}")
    print(f"  ImgSize : {img_size}")
    print("=" * 60)

    model = YOLO(weights)

    extra_kwargs = {}
    if optimize:
        extra_kwargs["optimize"] = True      # Reduce model size

    exported_path = model.export(
        format=format,
        imgsz=img_size,
        **extra_kwargs,
    )

    print(f"\n[Export] ✓ Model exported to: {exported_path}")
    print("[Export] You can now use this model in:")
    if format == "onnx":
        print("  • Windows/Linux: onnxruntime")
        print("  • Android: ONNX Runtime Mobile")
        print("  • Web: onnxruntime-web")
    elif format == "tflite":
        print("  • Android: TFLite Interpreter")
        print("  • Raspberry Pi: TFLite runtime")

    return exported_path


def main():
    parser = argparse.ArgumentParser(description="BlindAid Model Exporter")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                        help="Path to weights file")
    parser.add_argument("--format", type=str, default="onnx",
                        choices=["onnx", "tflite", "openvino", "coreml", "torchscript"],
                        help="Export format")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--optimize", action="store_true",
                        help="Enable TorchScript optimization (mobile)")
    args = parser.parse_args()

    export_model(
        weights=args.weights,
        format=args.format,
        img_size=args.imgsz,
        optimize=args.optimize,
    )


if __name__ == "__main__":
    main()
