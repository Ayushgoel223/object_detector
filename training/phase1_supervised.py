"""
BlindAid — Phase 1 Supervised Training
========================================
Fine-tunes YOLOv8n on a custom dataset built from:
  - Internet-fetched videos (YouTube frames)
  - Roboflow labeled datasets (doors, stairs, exit signs)
  - Open Images v7 (people, chairs, signs)
  - COCO subset (navigation classes)

New custom classes added beyond COCO:
  door, stairs, elevator, exit_sign, ramp, sign

Pipeline:
  1. Run internet_fetcher.py to download raw data
  2. Run this script to auto-label + train
  3. Best model saved to models/blindaid_supervised.pt
  4. Training metrics logged to MySQL training_runs table

Also trains the CNN-OCR model (CRNN) on OCR datasets:
  - IIIT5K, SVT word images → supervised character recognition

Usage:
  python training/phase1_supervised.py --task yolo --epochs 50
  python training/phase1_supervised.py --task ocr  --epochs 30
  python training/phase1_supervised.py --task all  --epochs 50
  python training/phase1_supervised.py --dry-run
"""

import argparse
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[Phase1] %(message)s")

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "training" / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))


# ── YOLO Dataset Preparation ──────────────────────────────────────────────────

class DatasetPreparer:
    """
    Organises downloaded images into a YOLO-format dataset:
      training/data/
        images/train/  ← JPG images
        images/val/
        labels/train/  ← YOLO .txt annotations
        labels/val/
    """

    CUSTOM_CLASSES = ["door", "stairs", "elevator", "exit_sign", "ramp", "sign"]
    COCO_NAV_IDS   = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 16, 24, 26, 28,
                      39, 56, 57, 58, 59, 60, 62, 63, 67, 73, 75]

    def __init__(self, raw_dir: Path = None, output_dir: Path = None):
        self.raw_dir    = raw_dir    or (DATA_DIR / "raw")
        self.output_dir = output_dir or DATA_DIR

    def prepare(self, val_split: float = 0.15) -> Path:
        """
        Scan raw_dir for images, auto-label with base YOLO, split train/val.
        Returns path to data.yaml.
        """
        images = self._collect_images()
        logger.info(f"Found {len(images)} images to process.")

        if len(images) == 0:
            logger.warning("No images found! Run internet_fetcher.py first.")
            return self._write_data_yaml([])

        # Shuffle + split
        np.random.shuffle(images)
        n_val   = max(1, int(len(images) * val_split))
        val_set = images[:n_val]
        trn_set = images[n_val:]

        self._setup_dirs()
        self._copy_and_label(trn_set, "train")
        self._copy_and_label(val_set, "val")

        data_yaml = self._write_data_yaml(self.CUSTOM_CLASSES)
        logger.info(f"Dataset ready: {len(trn_set)} train, {n_val} val")
        return data_yaml

    def _collect_images(self) -> List[Path]:
        exts = (".jpg", ".jpeg", ".png", ".webp")
        imgs = [p for ext in exts for p in self.raw_dir.rglob(f"*{ext}")]
        # Filter corrupted
        valid = []
        for p in imgs:
            if cv2.imread(str(p)) is not None:
                valid.append(p)
        return valid

    def _setup_dirs(self):
        for split in ("train", "val"):
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _copy_and_label(self, images: List[Path], split: str):
        """Copy images and create auto-generated pseudo-labels using base YOLO."""
        try:
            from ultralytics import YOLO
            auto_labeler = YOLO("yolov8n.pt")
            use_auto = True
        except Exception:
            auto_labeler = None
            use_auto = False

        img_out = self.output_dir / "images" / split
        lbl_out = self.output_dir / "labels" / split

        for img_path in images:
            dest = img_out / img_path.name
            shutil.copy2(img_path, dest)

            if use_auto:
                # Auto-label with base YOLOv8
                results = auto_labeler.predict(
                    str(img_path), conf=0.35, verbose=False
                )
                label_lines = []
                if results and results[0].boxes is not None:
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        if cls_id not in self.COCO_NAV_IDS:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                else:
                    label_lines = []  # No detections = empty label file
            else:
                label_lines = []

            lbl_file = lbl_out / (img_path.stem + ".txt")
            lbl_file.write_text("\n".join(label_lines))

    def _write_data_yaml(self, custom_classes: List[str]) -> Path:
        """Write Ultralytics data.yaml."""
        # Build class list: COCO names first, then custom
        from inference.detector import COCO_NAMES
        all_classes = {}
        for cid in sorted(self.COCO_NAV_IDS):
            all_classes[cid] = COCO_NAMES.get(cid, f"class_{cid}")
        # Append custom classes after COCO
        next_id = max(all_classes.keys()) + 1 if all_classes else 80
        for cc in custom_classes:
            all_classes[next_id] = cc
            next_id += 1

        data = {
            "path":  str(self.output_dir),
            "train": "images/train",
            "val":   "images/val",
            "nc":    len(all_classes),
            "names": {v: k for k, v in all_classes.items()},  # name→id format
        }

        # Ultralytics expects list format
        data["names"] = [all_classes[i] for i in sorted(all_classes.keys())]

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.info(f"data.yaml written: {len(all_classes)} classes → {yaml_path}")
        return yaml_path


# ── YOLO Fine-tuner ───────────────────────────────────────────────────────────

class YOLOFineTuner:
    """Fine-tunes YOLOv8n on the prepared custom dataset."""

    def __init__(self, data_yaml: Path, model_name: str = "yolov8n.pt",
                 output_name: str = "blindaid_supervised.pt",
                 db_manager=None):
        self.data_yaml   = data_yaml
        self.model_name  = model_name
        self.output_path = MODEL_DIR / output_name
        self.run_id      = str(uuid.uuid4())[:8]
        self.db          = db_manager

    def train(self, epochs: int = 50, batch: int = 8, img_size: int = 640,
               device: str = "cpu") -> str:
        """
        Run YOLOv8 fine-tuning.
        Returns path to best checkpoint.
        """
        from ultralytics import YOLO
        logger.info(f"[YOLO] Starting fine-tune: {epochs} epochs, batch={batch}, "
                    f"device={device}")

        model = YOLO(self.model_name)
        results = model.train(
            data     = str(self.data_yaml),
            epochs   = epochs,
            batch    = batch,
            imgsz    = img_size,
            device   = device,
            name     = f"blindaid_{self.run_id}",
            project  = str(ROOT / "training" / "runs"),
            exist_ok = True,
            augment  = True,
            patience = 10,    # Early stopping
            save     = True,
            plots    = True,
            verbose  = False,
        )

        # Copy best checkpoint to models/
        run_dir  = ROOT / "training" / "runs" / f"blindaid_{self.run_id}"
        best_pt  = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            shutil.copy2(best_pt, self.output_path)
            logger.info(f"[YOLO] Best model → {self.output_path}")

        # Log metrics to DB
        if self.db:
            try:
                metrics = results.results_dict if hasattr(results, "results_dict") else {}
                self.db.insert_training_run(
                    run_id=self.run_id, phase="supervised",
                    epoch=epochs,
                    metrics={
                        "map50":     metrics.get("metrics/mAP50(B)", 0),
                        "map50_95":  metrics.get("metrics/mAP50-95(B)", 0),
                        "precision": metrics.get("metrics/precision(B)", 0),
                        "recall":    metrics.get("metrics/recall(B)", 0),
                    },
                    notes=f"YOLOv8 fine-tune, {epochs} epochs"
                )
            except Exception as e:
                logger.debug(f"DB logging error: {e}")

        return str(self.output_path)


# ── OCR Supervised Trainer ────────────────────────────────────────────────────

class CRNNSupervisedTrainer:
    """
    Trains the CRNN on word-image OCR datasets (IIIT5K, SVT).
    Uses CTC loss for variable-length text recognition.
    """

    def __init__(self, data_dir: Path = None, db_manager=None):
        self.data_dir = data_dir or (DATA_DIR / "raw" / "ocr_datasets")
        self.db       = db_manager
        self.run_id   = str(uuid.uuid4())[:8]

    def _build_dataset(self) -> Optional[object]:
        """Build a torch Dataset from OCR image+label pairs."""
        try:
            import torch
            from torch.utils.data import Dataset
            from torchvision import transforms
            from PIL import Image

            class OCRWordDataset(Dataset):
                def __init__(self, data_dir: Path):
                    self.pairs = []  # (img_path, label_str)
                    # Load IIIT5K or SVT annotations
                    for label_file in data_dir.rglob("*.txt"):
                        for line in label_file.read_text().splitlines():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                img_p = label_file.parent / parts[0]
                                label = parts[1]
                                if img_p.exists() and label.isascii():
                                    self.pairs.append((img_p, label))
                    logger.info(f"[OCR Dataset] {len(self.pairs)} word samples")

                    self.transform = transforms.Compose([
                        transforms.Grayscale(),
                        transforms.Resize((32, 128)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ])

                def __len__(self):
                    return len(self.pairs)

                def __getitem__(self, idx):
                    img_p, label = self.pairs[idx]
                    img = Image.open(img_p).convert("RGB")
                    return self.transform(img), label

            return OCRWordDataset(self.data_dir)
        except ImportError:
            return None

    def train(self, epochs: int = 30, batch_size: int = 32) -> Optional[str]:
        """Train CRNN with CTC loss. Returns model save path or None."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            from inference.cnn_ocr.model import CRNN, NUM_CHARS, BLANK_IDX, char_to_idx
        except ImportError:
            logger.error("PyTorch or CRNN model not available.")
            return None

        dataset = self._build_dataset()
        if dataset is None or len(dataset) == 0:
            logger.warning("[OCR Train] No OCR data found. "
                           "Run internet_fetcher.py --source ocr first.")
            return None

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model   = CRNN(num_chars=NUM_CHARS).to(device)
        optim   = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctc_fn  = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

        def collate(batch):
            imgs, labels = zip(*batch)
            imgs   = torch.stack(imgs)
            return imgs, labels

        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, drop_last=True, num_workers=2)
        save_path = MODEL_DIR / "cnn_ocr_supervised.pt"
        best_loss = float("inf")

        logger.info(f"[OCR Train] Training CRNN: {epochs} epochs, {len(dataset)} samples")

        for epoch in range(1, epochs + 1):
            model.train()
            ep_loss = 0.0

            for imgs, label_strs in loader:
                imgs = imgs.to(device)          # (B, 1, 32, 128)
                log_probs = model(imgs)          # (T, B, C)
                T, B, C   = log_probs.shape

                # Encode labels
                targets, target_lengths = [], []
                for s in label_strs:
                    enc = [char_to_idx(c) for c in s if c in __import__("inference.cnn_ocr.model", fromlist=["CHARSET"]).CHARSET]
                    targets.extend(enc)
                    target_lengths.append(len(enc))

                targets        = torch.LongTensor(targets)
                target_lengths = torch.LongTensor(target_lengths)
                input_lengths  = torch.full((B,), T, dtype=torch.long)

                loss = ctc_fn(log_probs, targets, input_lengths, target_lengths)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
                ep_loss += loss.item()

            avg_loss = ep_loss / max(1, len(loader))
            logger.info(f"[OCR Train] Epoch {epoch}/{epochs} | CTC Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_path)

            if self.db:
                try:
                    self.db.insert_training_run(
                        run_id=self.run_id, phase="supervised",
                        epoch=epoch, metrics={"loss": avg_loss},
                        notes="CRNN CTC training"
                    )
                except Exception:
                    pass

        logger.info(f"[OCR Train] Done. Best CTC loss: {best_loss:.4f} → {save_path}")
        return str(save_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Supervised Training")
    parser.add_argument("--task",    choices=["yolo", "ocr", "prepare", "all"], default="all")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("=== Phase 1 Supervised — Dry Run ===")
        prep = DatasetPreparer()
        imgs = prep._collect_images()
        print(f"Found {len(imgs)} images in raw dir")
        print(f"Would train YOLO for {args.epochs} epochs on device={args.device}")
        print(f"Would train CRNN OCR for {args.epochs} epochs")
        sys.exit(0)

    if args.task in ("prepare", "yolo", "all"):
        logger.info("=== Preparing Dataset ===")
        prep = DatasetPreparer()
        data_yaml = prep.prepare()

    if args.task in ("yolo", "all"):
        logger.info("=== Phase 1: YOLO Fine-tuning ===")
        trainer = YOLOFineTuner(data_yaml)
        path = trainer.train(epochs=args.epochs, batch=args.batch, device=args.device)
        logger.info(f"YOLO model saved: {path}")

    if args.task in ("ocr", "all"):
        logger.info("=== Phase 1: CRNN OCR Training ===")
        ocr_trainer = CRNNSupervisedTrainer()
        path = ocr_trainer.train(epochs=max(args.epochs, 30))
        if path:
            logger.info(f"CRNN model saved: {path}")
