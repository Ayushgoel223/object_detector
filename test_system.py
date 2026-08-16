"""
BlindAid - Quick System Test
==============================
Runs a sanity check on all components WITHOUT needing a camera.
Tests: YOLOv8 loading, spatial analyzer, voice assistant.

Usage:
    python test_system.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

def test_detector():
    print("\n[Test 1/3] Object Detector")
    print("-" * 40)
    try:
        from inference.detector import ObjectDetector, COCO_NAMES
        detector = ObjectDetector(config_path=str(ROOT / "config.yaml"))
        print(f"  ✓ Model loaded: {detector.weights}")
        print(f"  ✓ Watching {len(detector.active_ids)} classes")
        print(f"  ✓ Device: {detector.device}")
        return True, detector
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False, None


def test_spatial_analyzer(detector):
    print("\n[Test 2/3] Spatial Analyzer")
    print("-" * 40)
    try:
        from inference.spatial_analyzer import SpatialAnalyzer
        from inference.detector import Detection

        analyzer = SpatialAnalyzer(config_path=str(ROOT / "config.yaml"))

        # Simulate a car dead-ahead, large (critical)
        fake_detections = [
            Detection(
                label="car",
                class_id=2,
                confidence=0.92,
                bbox=(200, 100, 750, 480),
                center_x=0.50,        # center
                center_y=0.60,
                area_fraction=0.35,   # 35% of frame = CRITICAL
            ),
            Detection(
                label="person",
                class_id=0,
                confidence=0.78,
                bbox=(10, 200, 120, 480),
                center_x=0.10,        # left
                center_y=0.70,
                area_fraction=0.08,   # NEAR
            ),
        ]

        instructions = analyzer.analyze(fake_detections)
        for inst in instructions:
            print(f"  → [{inst.urgency.name}] {inst.message}")

        print(f"  ✓ Spatial analyzer working. {len(instructions)} instructions generated.")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_voice_assistant():
    print("\n[Test 3/3] Voice Assistant")
    print("-" * 40)
    try:
        from inference.voice_assistant import VoiceAssistant
        va = VoiceAssistant(config_path=str(ROOT / "config.yaml"))
        va.start()
        print("  → Speaking test message...")
        va.speak_now("BlindAid system test complete. All components are working.")
        time.sleep(1)
        va.stop()
        print("  ✓ Voice assistant working.")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def main():
    print("=" * 50)
    print("  BlindAid — System Test")
    print("=" * 50)

    ok1, detector = test_detector()
    ok2 = test_spatial_analyzer(detector) if ok1 else False
    ok3 = test_voice_assistant()

    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  Object Detector  : {'✓ PASS' if ok1 else '✗ FAIL'}")
    print(f"  Spatial Analyzer : {'✓ PASS' if ok2 else '✗ FAIL'}")
    print(f"  Voice Assistant  : {'✓ PASS' if ok3 else '✗ FAIL'}")

    all_pass = ok1 and ok2 and ok3
    print("=" * 50)
    if all_pass:
        print("  ✓ ALL TESTS PASSED. Run: python app/main.py")
    else:
        print("  ✗ Some tests failed. Check error messages above.")
    print("=" * 50)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
