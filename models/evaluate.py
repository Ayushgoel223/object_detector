"""
BlindAid - Model Evaluation Script
=====================================
Evaluates the YOLOv8 model on COCO validation set.
Reports mAP50, mAP50-95, precision, recall per class.

Usage:
    python models/evaluate.py
    python models/evaluate.py --weights yolov8s.pt --data coco128.yaml
"""

import argparse
import yaml
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def evaluate(weights: str, data_yaml: str, img_size: int, device: str):
    print("=" * 60)
    print("  BlindAid — Model Evaluation")
    print("=" * 60)
    print(f"  Weights : {weights}")
    print(f"  Dataset : {data_yaml}")
    print(f"  Device  : {device}")
    print("=" * 60)

    model = YOLO(weights)

    print("\n[Eval] Running validation...")
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=device,
        verbose=True,
        plots=True,          # saves PR curves, confusion matrix
        save_json=True,      # saves results.json for further analysis
    )

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  mAP50        : {metrics.box.map50:.4f}")
    print(f"  mAP50-95     : {metrics.box.map:.4f}")
    print(f"  Precision    : {metrics.box.mp:.4f}")
    print(f"  Recall       : {metrics.box.mr:.4f}")
    print("=" * 60)
    print("\n[Eval] Plots saved to runs/val/")
    print("[Eval] mAP50 >= 0.50 is production-acceptable for navigation use.")


def main():
    parser = argparse.ArgumentParser(description="BlindAid Model Evaluator")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                        help="Path to weights file")
    parser.add_argument("--data", type=str, default="coco128.yaml",
                        help="Dataset YAML (coco128.yaml downloads automatically)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for evaluation")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or 0 (GPU)")
    args = parser.parse_args()

    evaluate(
        weights=args.weights,
        data_yaml=args.data,
        img_size=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
