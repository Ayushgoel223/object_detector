"""
BlindAid — CNN-OCR Package
============================
Convolutional Neural Network based OCR:
  - TextRegionCNN  : detects text bounding boxes in a frame
  - CRNN           : reads characters from cropped text regions (CTC)
  - CNNOCRReader   : high-level API used by the inference pipeline
"""

from .ocr_reader import CNNOCRReader, OCRResult

__all__ = ["CNNOCRReader", "OCRResult"]
