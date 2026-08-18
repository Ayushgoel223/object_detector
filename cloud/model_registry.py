"""
BlindAid — Model Registry (HuggingFace Hub)
=============================================
Stores and versions all trained models on HuggingFace Hub (FREE).

Models stored:
  blindaid/blindaid_supervised.pt   — Fine-tuned YOLOv8
  blindaid/cnn_ocr_supervised.pt    — CRNN text recognizer
  blindaid/cnn_ocr_simclr_backbone.pt — SimCLR pretrained backbone
  blindaid/scene_encoder.pt         — SceneEncoder autoencoder
  blindaid/rl_agent.pt              — DQN navigation agent
  blindaid/rl_agent_best.pt         — Best RL checkpoint

On laptop startup:
  → Checks if cloud model is newer than local model
  → Auto-downloads if improved version available
  → System runs with the best model automatically

Usage:
  python cloud/model_registry.py --action download --phase supervised
  python cloud/model_registry.py --action upload   --phase rl
  python cloud/model_registry.py --action check    --phase all
  python cloud/model_registry.py --action status
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[ModelRegistry] %(message)s")

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

REGISTRY_CACHE = MODEL_DIR / ".registry_cache.json"

# ── Try HuggingFace Hub ───────────────────────────────────────────────────────
try:
    from huggingface_hub import HfApi, hf_hub_download, upload_file, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not installed. Run: pip install huggingface_hub")


# ── Model Manifest ────────────────────────────────────────────────────────────
# Maps phase → list of model files managed for that phase

MODEL_MANIFEST: Dict[str, List[str]] = {
    "supervised": [
        "blindaid_supervised.pt",
        "cnn_ocr_supervised.pt",
    ],
    "backbone": [
        "cnn_ocr_simclr_backbone.pt",
        "scene_encoder.pt",
    ],
    "rl": [
        "rl_agent.pt",
        "rl_agent_best.pt",
    ],
    "all": [],   # Filled dynamically = all of the above
}
MODEL_MANIFEST["all"] = [m for lst in MODEL_MANIFEST.values() for m in lst]


class ModelRegistry:
    """
    Manages model versioning between local disk and HuggingFace Hub.
    Completely free tier — public or private repo.

    HF Repo format: {hf_username}/{repo_name}
    Default: "blindaid-project/models"  (change to your HF username)
    """

    def __init__(self, repo_id: str = None, token: str = None):
        self.token   = token or os.environ.get("HF_TOKEN", "")
        self.repo_id = repo_id or os.environ.get("HF_REPO_ID", "blindaid-project/models")
        self.api     = HfApi(token=self.token) if HF_AVAILABLE and self.token else None
        self._cache  = self._load_cache()

    # ── Download ──────────────────────────────────────────────────────────────

    def download(self, phase: str = "all") -> Dict[str, bool]:
        """
        Download models for the given phase from HuggingFace.
        Only downloads if cloud version is newer than local.

        Returns dict: {filename: True if downloaded}
        """
        if not self._check_hf():
            return {}

        files = MODEL_MANIFEST.get(phase, [phase + ".pt"])
        results = {}

        for filename in files:
            try:
                local_path  = MODEL_DIR / filename
                cloud_hash  = self._get_cloud_hash(filename)
                local_hash  = self._md5(local_path) if local_path.exists() else None

                if cloud_hash and cloud_hash == local_hash:
                    logger.info(f"  {filename}: up to date (hash match)")
                    results[filename] = False
                    continue

                logger.info(f"  Downloading {filename} from HuggingFace...")
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    token=self.token,
                    local_dir=str(MODEL_DIR),
                    local_dir_use_symlinks=False,
                )
                # Move to model dir if needed
                dl_path = Path(downloaded)
                if dl_path != local_path and dl_path.exists():
                    shutil.move(str(dl_path), str(local_path))

                logger.info(f"  ✓ Downloaded: {filename}")
                self._cache[filename] = {
                    "downloaded_at": datetime.utcnow().isoformat(),
                    "hash": self._md5(local_path),
                }
                self._save_cache()
                results[filename] = True

            except Exception as e:
                logger.warning(f"  {filename}: download failed ({e})")
                results[filename] = False

        return results

    # ── Upload ────────────────────────────────────────────────────────────────

    def upload(self, phase: str = "all") -> Dict[str, bool]:
        """
        Upload locally trained models to HuggingFace Hub.
        Creates the repo if it doesn't exist.
        """
        if not self._check_hf():
            return {}

        self._ensure_repo()
        files = MODEL_MANIFEST.get(phase, [phase + ".pt"])
        results = {}

        for filename in files:
            local_path = MODEL_DIR / filename
            if not local_path.exists():
                logger.warning(f"  {filename}: not found locally, skipping upload.")
                results[filename] = False
                continue

            try:
                logger.info(f"  Uploading {filename} → {self.repo_id}...")
                self.api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=filename,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=f"Auto-train: {filename} @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
                )
                # Update local hash cache
                self._cache[filename] = {
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "hash": self._md5(local_path),
                }
                self._save_cache()
                logger.info(f"  ✓ Uploaded: {filename}")
                results[filename] = True

            except Exception as e:
                logger.error(f"  {filename}: upload failed ({e})")
                results[filename] = False

        return results

    # ── Check / Status ────────────────────────────────────────────────────────

    def check_for_updates(self, phase: str = "all") -> Dict[str, bool]:
        """
        Check if cloud has newer models than local.
        Returns {filename: True if update available}.
        """
        if not self._check_hf():
            return {}

        files = MODEL_MANIFEST.get(phase, [phase + ".pt"])
        updates = {}

        for filename in files:
            local_path  = MODEL_DIR / filename
            cloud_hash  = self._get_cloud_hash(filename)
            local_hash  = self._md5(local_path) if local_path.exists() else None
            has_update  = cloud_hash is not None and cloud_hash != local_hash
            updates[filename] = has_update
            status = "⬆ UPDATE AVAILABLE" if has_update else "✓ up to date"
            logger.info(f"  {filename}: {status}")

        return updates

    def status(self) -> None:
        """Print full registry status."""
        print("\n=== BlindAid Model Registry Status ===")
        print(f"HuggingFace Repo : {self.repo_id}")
        print(f"Local Model Dir  : {MODEL_DIR}")
        print(f"HF Available     : {HF_AVAILABLE}")
        print(f"HF Token         : {'✓ set' if self.token else '✗ missing'}")
        print()
        print("Local Models:")
        for phase, files in MODEL_MANIFEST.items():
            if phase == "all":
                continue
            print(f"  [{phase}]")
            for f in files:
                path = MODEL_DIR / f
                if path.exists():
                    size = path.stat().st_size / 1024 / 1024
                    mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    print(f"    ✓ {f:<40} {size:>6.1f} MB   {mtime}")
                else:
                    print(f"    ✗ {f:<40} (not trained yet)")

        if self._cache:
            print("\nLast Operations:")
            for fname, meta in list(self._cache.items())[:5]:
                print(f"  {fname}: {meta}")

    # ── Auto-Update on Startup ────────────────────────────────────────────────

    def auto_update_on_startup(self, phase: str = "all") -> bool:
        """
        Called when the laptop turns on and BlindAid starts.
        Downloads any improved models silently in background.
        Returns True if any model was updated.
        """
        if not self._check_hf():
            return False

        logger.info("[Registry] Checking for cloud model updates...")
        updates = self.check_for_updates(phase)
        needs_download = [f for f, has_update in updates.items() if has_update]

        if not needs_download:
            logger.info("[Registry] All models up to date.")
            return False

        logger.info(f"[Registry] Downloading {len(needs_download)} improved model(s)...")
        results = self.download(phase)
        downloaded = sum(1 for v in results.values() if v)
        logger.info(f"[Registry] Updated {downloaded} model(s). System now uses latest trained weights.")
        return downloaded > 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_hf(self) -> bool:
        if not HF_AVAILABLE:
            logger.warning("huggingface_hub not installed.")
            return False
        if not self.token:
            logger.warning("HF_TOKEN not set. Add it to .env or GitHub Secrets.")
            return False
        return True

    def _ensure_repo(self):
        """Create HF repo if it doesn't exist."""
        try:
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=True,        # Private repo — your models stay yours
                exist_ok=True,
            )
        except Exception as e:
            logger.debug(f"Repo create: {e}")

    def _get_cloud_hash(self, filename: str) -> Optional[str]:
        """Get the SHA256/MD5 of a file on HuggingFace (via model card metadata)."""
        try:
            info = self.api.model_info(self.repo_id, token=self.token)
            # Check siblings (uploaded files)
            for sibling in (info.siblings or []):
                if sibling.rfilename == filename:
                    return sibling.blob_id   # HF blob ID acts as version hash
        except Exception:
            pass
        return None

    def _md5(self, path: Path) -> Optional[str]:
        if not path.exists():
            return None
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_cache(self) -> dict:
        if REGISTRY_CACHE.exists():
            try:
                return json.loads(REGISTRY_CACHE.read_text())
            except Exception:
                pass
        return {}

    def _save_cache(self):
        REGISTRY_CACHE.write_text(json.dumps(self._cache, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlindAid Model Registry")
    parser.add_argument("--action", choices=["download", "upload", "check", "status", "auto"],
                        default="status")
    parser.add_argument("--phase", default="all",
                        choices=["supervised", "backbone", "rl", "all"])
    parser.add_argument("--repo",  default="", help="HuggingFace repo ID (user/repo)")
    args = parser.parse_args()

    # Load .env
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    registry = ModelRegistry(repo_id=args.repo or None)

    if args.action == "status":
        registry.status()
    elif args.action == "download":
        registry.download(args.phase)
    elif args.action == "upload":
        registry.upload(args.phase)
    elif args.action == "check":
        registry.check_for_updates(args.phase)
    elif args.action == "auto":
        updated = registry.auto_update_on_startup(args.phase)
        sys.exit(0 if updated else 1)
