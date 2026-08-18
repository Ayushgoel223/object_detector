"""
BlindAid — Model Evaluator (for CI/CD)
========================================
Compares new model vs old model performance.
Outputs /tmp/model_improved.txt = 'true' or 'false'
Used by GitHub Actions to decide whether to upload the new model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def evaluate_yolo(model_path: Path, data_yaml: Path, img_size: int = 640) -> dict:
    """Run YOLO validation and return metrics dict."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        results = model.val(
            data=str(data_yaml),
            imgsz=img_size,
            device="cpu",
            verbose=False,
        )
        return {
            "map50":     float(results.box.map50),
            "map50_95":  float(results.box.map),
            "precision": float(results.box.mp),
            "recall":    float(results.box.mr),
        }
    except Exception as e:
        print(f"[Eval] YOLO eval failed: {e}")
        return {"map50": 0.0, "map50_95": 0.0}


def compare(output_file: str = "/tmp/model_improved.txt") -> bool:
    """
    Compare new trained model vs previous best.
    Returns True and writes 'true' to output_file if improved.
    """
    new_model   = ROOT / "models" / "blindaid_supervised.pt"
    old_metrics_path = ROOT / "models" / ".last_metrics.json"
    data_yaml   = ROOT / "training" / "data" / "data.yaml"

    if not new_model.exists():
        print("[Eval] No new model found. Skipping comparison.")
        Path(output_file).write_text("false")
        return False

    if not data_yaml.exists():
        print("[Eval] No data.yaml found. Assuming improved (first run).")
        Path(output_file).write_text("true")
        return True

    print(f"[Eval] Evaluating new model: {new_model}")
    new_metrics = evaluate_yolo(new_model, data_yaml)
    print(f"[Eval] New metrics: {new_metrics}")

    if old_metrics_path.exists():
        old_metrics = json.loads(old_metrics_path.read_text())
        print(f"[Eval] Old metrics: {old_metrics}")
        improved = new_metrics["map50"] > old_metrics.get("map50", 0.0)
    else:
        improved = True   # First training run — always upload

    # Save new metrics as baseline
    old_metrics_path.write_text(json.dumps(new_metrics, indent=2))

    result = "true" if improved else "false"
    Path(output_file).write_text(result)
    print(f"[Eval] Model improved: {improved}")
    return improved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--output", default="/tmp/model_improved.txt")
    args = parser.parse_args()

    if args.compare:
        compare(args.output)
