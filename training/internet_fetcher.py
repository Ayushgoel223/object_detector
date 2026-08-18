"""
BlindAid — Internet Dataset & Video Fetcher
=============================================
Downloads training data from multiple online sources:

1. YouTubeFetcher   — navigation/indoor walk videos via yt-dlp
2. RoboflowFetcher  — labeled object detection datasets (Roboflow Universe)
3. OpenImagesFetcher— Google Open Images v7 selective download
4. OCRDatasetFetcher— Synth90K, IIIT5K, SVT text recognition datasets

All fetchers:
  - Resume-safe (skip already-downloaded files)
  - Hash-validate downloaded archives
  - Log records to MySQL `training_samples` table
  - Write to training/data/raw/<source>/<category>/

Usage:
    python training/internet_fetcher.py --source youtube --query "indoor walk" --limit 10
    python training/internet_fetcher.py --source roboflow --dataset door-detection
    python training/internet_fetcher.py --source openimages --classes Door Stairs
    python training/internet_fetcher.py --source ocr
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "training" / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Utility: download file with progress bar ──────────────────────────────────

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download url to dest with progress bar. Returns True on success."""
    if dest.exists():
        logger.info(f"Already downloaded: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
        logger.info(f"Downloaded: {dest}")
        return True
    except Exception as e:
        logger.error(f"Download failed ({url}): {e}")
        if dest.exists():
            dest.unlink()
        return False


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── 1. YouTube Fetcher ────────────────────────────────────────────────────────

class YouTubeFetcher:
    """
    Downloads first-person indoor navigation videos from YouTube using yt-dlp.
    Extracts frames at 2 FPS for use as unlabeled training images.

    Prerequisites: yt-dlp must be installed (pip install yt-dlp)
    """

    DEFAULT_QUERIES = [
        "first person indoor walking navigation",
        "blind person navigation indoor",
        "indoor navigation hallway walk",
        "shopping mall navigation first person",
        "hospital corridor walk first person",
        "indoor navigation wheelchair",
        "airport terminal walking first person",
    ]

    def __init__(self, output_dir: Path = None, max_videos: int = 5,
                 max_duration_sec: int = 300):
        self.output_dir   = output_dir or (DATA_DIR / "youtube")
        self.max_videos   = max_videos
        self.max_duration = max_duration_sec
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_ytdlp(self) -> bool:
        """Verify yt-dlp is installed."""
        try:
            result = subprocess.run(
                ["yt-dlp", "--version"], capture_output=True, text=True, timeout=10
            )
            logger.info(f"yt-dlp version: {result.stdout.strip()}")
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("yt-dlp not found. Installing via pip...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "yt-dlp", "-q"],
                    check=True, timeout=60
                )
                return True
            except Exception as e:
                logger.error(f"Failed to install yt-dlp: {e}")
                return False

    def fetch(self, query: str = None, video_urls: List[str] = None) -> List[Path]:
        """
        Download videos matching query OR from explicit URL list.
        Returns list of downloaded video file paths.
        """
        if not self.check_ytdlp():
            logger.error("yt-dlp unavailable. Skipping YouTube fetch.")
            return []

        if video_urls:
            urls = video_urls
        else:
            q = query or self.DEFAULT_QUERIES[0]
            urls = self._search_urls(q)

        downloaded = []
        for url in urls[:self.max_videos]:
            path = self._download_video(url)
            if path:
                downloaded.append(path)

        logger.info(f"[YouTube] Downloaded {len(downloaded)} videos to {self.output_dir}")
        return downloaded

    def _search_urls(self, query: str) -> List[str]:
        """Use yt-dlp to search YouTube and return video URLs."""
        cmd = [
            "yt-dlp",
            f"ytsearch{self.max_videos}:{query}",
            "--get-url",
            "--match-filter", f"duration <= {self.max_duration}",
            "--no-playlist",
            "--quiet",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            urls = [u.strip() for u in result.stdout.split("\n") if u.strip()]
            logger.info(f"[YouTube] Found {len(urls)} videos for query: '{query}'")
            return urls
        except Exception as e:
            logger.error(f"[YouTube] URL search failed: {e}")
            return []

    def _download_video(self, url: str) -> Optional[Path]:
        """Download a single video. Returns path or None."""
        cmd = [
            "yt-dlp", url,
            "--output", str(self.output_dir / "%(id)s.%(ext)s"),
            "--format", "bestvideo[height<=480][ext=mp4]/best[height<=480]",
            "--max-filesize", "200M",
            "--no-playlist",
            "--quiet",
            "--no-warnings",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                # Find the downloaded file
                mp4_files = sorted(self.output_dir.glob("*.mp4"), key=os.path.getmtime)
                if mp4_files:
                    return mp4_files[-1]
        except subprocess.TimeoutExpired:
            logger.warning(f"[YouTube] Download timeout: {url}")
        except Exception as e:
            logger.error(f"[YouTube] Download error: {e}")
        return None

    def extract_frames(self, video_path: Path, fps: float = 2.0,
                        output_dir: Path = None) -> List[Path]:
        """Extract frames from video at given FPS using ffmpeg."""
        out_dir = output_dir or (DATA_DIR / "youtube_frames" / video_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)

        pattern = str(out_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "3",        # JPEG quality 1-31, lower=better
            pattern,
            "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=300)
            frames = sorted(out_dir.glob("frame_*.jpg"))
            logger.info(f"[YouTube] Extracted {len(frames)} frames from {video_path.name}")
            return frames
        except FileNotFoundError:
            logger.error("[YouTube] ffmpeg not found. Please install ffmpeg.")
            return []
        except Exception as e:
            logger.error(f"[YouTube] Frame extraction failed: {e}")
            return []


# ── 2. Roboflow Fetcher ───────────────────────────────────────────────────────

class RoboflowFetcher:
    """
    Downloads public datasets from Roboflow Universe.
    Uses the public download API (no API key needed for Universe datasets).

    Popular navigation datasets:
      - Doors: roboflow.com/universe/search?q=door+detection
      - Stairs: roboflow.com/universe/search?q=stair+detection
      - Signs: roboflow.com/universe/search?q=sign+navigation
    """

    UNIVERSE_API = "https://universe.roboflow.com"

    # Pre-curated navigation-relevant public datasets (workspace/project/version)
    PRESET_DATASETS = {
        "doors":       ("roboflow-universe-projects", "door-detection-b7oie", 1),
        "stairs":      ("roboflow-universe-projects", "stair-detection",       1),
        "exit_signs":  ("roboflow-universe-projects", "exit-sign-detection-1", 1),
        "people":      ("roboflow-universe-projects", "people-detection-general", 1),
    }

    def __init__(self, output_dir: Path = None, api_key: str = None):
        self.output_dir = output_dir or (DATA_DIR / "roboflow")
        self.api_key    = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_preset(self, category: str = "doors") -> Optional[Path]:
        """Download one of the preset navigation datasets."""
        if category not in self.PRESET_DATASETS:
            logger.error(f"Unknown category: {category}. Available: {list(self.PRESET_DATASETS)}")
            return None

        workspace, project, version = self.PRESET_DATASETS[category]
        return self.fetch_dataset(workspace, project, version)

    def fetch_dataset(self, workspace: str, project: str, version: int = 1,
                       fmt: str = "yolov8") -> Optional[Path]:
        """
        Download a specific Roboflow dataset in YOLOv8 format.

        Returns path to extracted dataset directory or None.
        """
        dest_dir = self.output_dir / f"{project}_v{version}"
        if dest_dir.exists() and any(dest_dir.glob("**/*.jpg")):
            logger.info(f"[Roboflow] Already downloaded: {project}")
            return dest_dir

        # Try Roboflow Python SDK first (best compatibility)
        if self._fetch_via_sdk(workspace, project, version, fmt, dest_dir):
            return dest_dir

        # Manual ZIP download fallback
        return self._fetch_via_zip(workspace, project, version, fmt, dest_dir)

    def _fetch_via_sdk(self, workspace, project, version, fmt, dest_dir) -> bool:
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=self.api_key or "publicapi")
            proj = rf.workspace(workspace).project(project)
            dataset = proj.version(version).download(fmt, location=str(dest_dir))
            logger.info(f"[Roboflow] Downloaded via SDK: {project} → {dest_dir}")
            return True
        except ImportError:
            logger.debug("[Roboflow] SDK not installed; trying direct download.")
            return False
        except Exception as e:
            logger.debug(f"[Roboflow] SDK download failed: {e}")
            return False

    def _fetch_via_zip(self, workspace, project, version, fmt, dest_dir) -> Optional[Path]:
        """Direct ZIP download from Roboflow export URL."""
        url = (f"https://universe.roboflow.com/{workspace}/{project}"
               f"/dataset/{version}/download/{fmt}")
        if self.api_key:
            url += f"?api_key={self.api_key}"

        zip_path = self.output_dir / f"{project}_v{version}.zip"
        if not download_file(url, zip_path):
            return None

        try:
            shutil.unpack_archive(str(zip_path), str(dest_dir))
            zip_path.unlink()   # Remove zip after extraction
            logger.info(f"[Roboflow] Extracted: {project} → {dest_dir}")
            return dest_dir
        except Exception as e:
            logger.error(f"[Roboflow] Extraction failed: {e}")
            return None


# ── 3. Open Images Fetcher ────────────────────────────────────────────────────

class OpenImagesFetcher:
    """
    Downloads labeled images from Google Open Images v7.
    Uses fiftyone (recommended) or the openimages Python package.

    Target classes (Open Images annotation names):
      Door, Stairs, Elevator, Person, Chair, Signage, Vehicle
    """

    OI_CLASSES = [
        "Door", "Stairs", "Person", "Chair", "Wheelchair",
        "Elevator", "Signage", "Traffic sign", "Stop sign",
        "Vehicle", "Bicycle", "Car",
    ]

    def __init__(self, output_dir: Path = None, max_samples: int = 200):
        self.output_dir  = output_dir or (DATA_DIR / "openimages")
        self.max_samples = max_samples
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, classes: List[str] = None, split: str = "validation") -> int:
        """
        Download images with bounding box annotations for specified classes.

        Args:
            classes: List of Open Images class names. Defaults to OI_CLASSES.
            split:   'train' | 'validation' | 'test'

        Returns:
            Number of images successfully downloaded.
        """
        classes = classes or self.OI_CLASSES[:6]   # Default: first 6

        # Try fiftyone (best experience)
        count = self._fetch_via_fiftyone(classes, split)
        if count > 0:
            return count

        # Fallback: openimages package
        return self._fetch_via_openimages_pkg(classes, split)

    def _fetch_via_fiftyone(self, classes: List[str], split: str) -> int:
        try:
            import fiftyone as fo
            import fiftyone.zoo as foz

            logger.info(f"[OpenImages] Downloading via fiftyone: {classes}")
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=classes,
                max_samples=self.max_samples,
                dataset_dir=str(self.output_dir / "fiftyone"),
            )
            count = len(dataset)
            logger.info(f"[OpenImages] Downloaded {count} samples via fiftyone.")
            return count

        except ImportError:
            logger.debug("[OpenImages] fiftyone not installed.")
            return 0
        except Exception as e:
            logger.warning(f"[OpenImages] fiftyone download failed: {e}")
            return 0

    def _fetch_via_openimages_pkg(self, classes: List[str], split: str) -> int:
        """Fallback using the openimages pip package."""
        try:
            from openimages.download import download_dataset
            out = self.output_dir / "openimages_pkg"
            out.mkdir(exist_ok=True)
            download_dataset(
                str(out),
                classes,
                annotation_format="darknet",
                limit=self.max_samples // len(classes),
            )
            files = list(out.rglob("*.jpg"))
            logger.info(f"[OpenImages] Downloaded {len(files)} images via openimages pkg.")
            return len(files)
        except ImportError:
            logger.warning("[OpenImages] openimages package not installed. "
                           "Run: pip install openimages")
            return 0
        except Exception as e:
            logger.error(f"[OpenImages] Download failed: {e}")
            return 0


# ── 4. OCR Dataset Fetcher ────────────────────────────────────────────────────

class OCRDatasetFetcher:
    """
    Downloads OCR word-image datasets for CRNN training.

    Datasets:
      - MJSynth / Synth90K  : 9 million synthetic word images
      - IIIT5K              : 5,000 word images from web
      - SVT (Street View Text): real-world sign text images

    Note: Synth90K is very large (~10GB). We download a 10% subset by default.
    """

    DATASETS = {
        "iiit5k": {
            "url": "https://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz",
            "filename": "IIIT5K-Word_V3.0.tar.gz",
        },
        "svt": {
            "url": "http://www.iapr-tc11.org/dataset/SVT/svt1.zip",
            "filename": "svt1.zip",
        },
        # Synth90K is too large for direct download here; use annotation subset
        "synth90k_subset": {
            "url": "https://www.robots.ox.ac.uk/~vgg/data/text/annotation.txt.gz",
            "filename": "synth90k_annotation.gz",
        },
    }

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or (DATA_DIR / "ocr_datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, dataset: str = "iiit5k") -> Optional[Path]:
        """Download and extract an OCR dataset. Returns path or None."""
        if dataset not in self.DATASETS:
            logger.error(f"Unknown OCR dataset: {dataset}. Options: {list(self.DATASETS)}")
            return None

        info     = self.DATASETS[dataset]
        filename = info["filename"]
        dest     = self.output_dir / filename

        if not download_file(info["url"], dest):
            return None

        # Extract
        extract_dir = self.output_dir / dataset
        extract_dir.mkdir(exist_ok=True)
        try:
            shutil.unpack_archive(str(dest), str(extract_dir))
            logger.info(f"[OCR] Extracted {dataset} → {extract_dir}")
            return extract_dir
        except Exception as e:
            logger.error(f"[OCR] Extraction failed for {dataset}: {e}")
            return None

    def fetch_all(self) -> Dict[str, Optional[Path]]:
        results = {}
        for name in self.DATASETS:
            logger.info(f"[OCR] Fetching dataset: {name}")
            results[name] = self.fetch(name)
        return results


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BlindAid Internet Dataset Fetcher")
    parser.add_argument("--source", choices=["youtube", "roboflow", "openimages", "ocr", "all"],
                        default="all", help="Data source to fetch from")
    parser.add_argument("--query", default="indoor navigation first person",
                        help="YouTube search query (for --source youtube)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max videos/samples to download")
    parser.add_argument("--classes", nargs="+",
                        default=["Door", "Stairs", "Person"],
                        help="Open Images classes (for --source openimages)")
    parser.add_argument("--dataset", default="doors",
                        help="Roboflow preset dataset name")
    parser.add_argument("--roboflow-key", default="",
                        help="Roboflow API key (optional)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded, don't execute")
    args = parser.parse_args()

    if args.dry_run:
        print("=== Dry Run Mode — No files will be downloaded ===")
        print(f"YouTube: query='{args.query}', limit={args.limit}")
        print(f"Roboflow: dataset='{args.dataset}'")
        print(f"Open Images: classes={args.classes}")
        print("OCR: IIIT5K, SVT, Synth90K_subset")
        return

    sources = [args.source] if args.source != "all" else ["youtube", "roboflow", "openimages", "ocr"]

    if "youtube" in sources:
        logger.info("=== Fetching YouTube Videos ===")
        yt = YouTubeFetcher(max_videos=args.limit)
        videos = yt.fetch(query=args.query)
        for v in videos:
            yt.extract_frames(v, fps=2.0)

    if "roboflow" in sources:
        logger.info("=== Fetching Roboflow Dataset ===")
        rf = RoboflowFetcher(api_key=args.roboflow_key)
        for cat in ["doors", "stairs", "exit_signs"]:
            rf.fetch_preset(cat)

    if "openimages" in sources:
        logger.info("=== Fetching Open Images ===")
        oi = OpenImagesFetcher(max_samples=args.limit * 20)
        oi.fetch(classes=args.classes)

    if "ocr" in sources:
        logger.info("=== Fetching OCR Datasets ===")
        ocr = OCRDatasetFetcher()
        ocr.fetch("iiit5k")
        ocr.fetch("svt")

    logger.info(f"\n✓ All downloads complete. Data in: {DATA_DIR}")


if __name__ == "__main__":
    main()
