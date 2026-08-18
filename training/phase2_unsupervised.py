"""
BlindAid — Phase 2 Unsupervised Learning
==========================================
Self-supervised and clustering methods to discover visual patterns
without requiring labeled data.

Modules:
  FeatureExtractor  — Extract CNN embeddings from unlabeled images
  ClusterAnalyzer   — K-Means + DBSCAN on feature space
  SimCLRTrainer     — Contrastive pretraining of CNN backbone
  SceneEmbedding    — Autoencoder for full-frame RL state encoding
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

ROOT      = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR  = ROOT / "training" / "data"
MODEL_DIR.mkdir(exist_ok=True)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logger.error("PyTorch not available. Phase 2 requires PyTorch.")

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("scikit-learn not available. Clustering disabled.")


# ── Dataset Helper ────────────────────────────────────────────────────────────

class UnlabeledImageDataset(Dataset):
    """Loads images from a directory (no labels needed)."""

    def __init__(self, img_dir: Path, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        self.paths     = [p for ext in extensions for p in img_dir.rglob(f"*{ext}")]
        self.transform = transform
        logger.info(f"[Dataset] {len(self.paths)} unlabeled images from {img_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        import cv2
        path  = self.paths[idx]
        img   = cv2.imread(str(path))
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        return img, str(path)


class AugmentedPairDataset(Dataset):
    """
    For SimCLR: returns two differently-augmented views of each image.
    """

    def __init__(self, img_dir: Path, img_size: int = 128):
        self.paths  = [p for ext in (".jpg", ".jpeg", ".png")
                       for p in img_dir.rglob(f"*{ext}")]
        aug = [
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        self.augment = T.Compose(aug)
        logger.info(f"[SimCLR Dataset] {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.augment(img), self.augment(img)   # Two different augmentations


# ── Feature Extractor ─────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Runs unlabeled images through a frozen CNN backbone to produce feature vectors.
    These features are used for clustering (K-Means / DBSCAN).
    """

    def __init__(self, model_path: Optional[Path] = None, device=None):
        if not TORCH_OK:
            raise RuntimeError("PyTorch required.")

        self.device = device or (torch.device("cuda") if torch.cuda.is_available()
                                  else torch.device("cpu"))
        self.model = self._load_backbone(model_path)
        self.model.eval()

    def _load_backbone(self, model_path: Optional[Path]):
        """Load custom CRNN encoder or fall back to ImageNet ResNet-18."""
        # Try custom pretrained CRNN backbone
        crnn_path = model_path or (MODEL_DIR / "cnn_ocr_supervised.pt")
        if crnn_path.exists():
            try:
                sys.path.insert(0, str(ROOT))
                from inference.cnn_ocr.model import CRNNFeatureExtractor
                cnn = CRNNFeatureExtractor()
                # Load only the CNN part weights
                state = torch.load(crnn_path, map_location=self.device)
                # Filter keys matching cnn.*
                cnn_state = {k.replace("cnn.", ""): v
                             for k, v in state.items() if k.startswith("cnn.")}
                if cnn_state:
                    cnn.load_state_dict(cnn_state, strict=False)
                    logger.info("[FeatExtract] Loaded custom CNN backbone from CRNN.")
                return cnn.to(self.device)
            except Exception as e:
                logger.debug(f"Custom backbone load failed ({e}), using ResNet-18.")

        # Fallback: ImageNet ResNet-18 (truncated at layer3)
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            import torchvision.models as m
            base = m.resnet18(pretrained=True)

        # Keep only up to layer3 → 256 channels
        backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        for p in backbone.parameters():
            p.requires_grad = False
        logger.info("[FeatExtract] Using frozen ResNet-18 backbone.")
        return backbone.to(self.device)

    def extract(self, img_dir: Path, batch_size: int = 32,
                 img_size: int = 128) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features for all images in img_dir.

        Returns:
            features : (N, D) float32 array
            paths    : list of N image paths
        """
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        dataset  = UnlabeledImageDataset(img_dir, transform=transform)
        loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        all_features: List[np.ndarray] = []
        all_paths:    List[str]        = []

        with torch.no_grad():
            for batch_imgs, batch_paths in loader:
                batch_imgs = batch_imgs.to(self.device)
                feats = self.model(batch_imgs)
                # Flatten if not already
                if feats.dim() > 2:
                    feats = feats.view(feats.size(0), -1)
                all_features.append(feats.cpu().numpy())
                all_paths.extend(batch_paths)

        features = np.vstack(all_features).astype(np.float32)
        logger.info(f"[FeatExtract] Extracted {features.shape} feature matrix.")
        return features, all_paths

    def save(self, features: np.ndarray, paths: List[str], save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(save_dir / "features.npy"), features)
        (save_dir / "paths.txt").write_text("\n".join(paths))
        logger.info(f"[FeatExtract] Saved features → {save_dir}")

    @staticmethod
    def load(save_dir: Path) -> Tuple[np.ndarray, List[str]]:
        features = np.load(str(save_dir / "features.npy"))
        paths    = (save_dir / "paths.txt").read_text().split("\n")
        return features, paths


# ── Cluster Analyzer ─────────────────────────────────────────────────────────

class ClusterAnalyzer:
    """
    Clusters CNN features to discover visual sub-classes and anomalies.

    K-Means  : finds K dominant visual patterns
    DBSCAN   : finds anomalies (rare/unseen scenes) without fixed K
    PCA      : reduces dimensionality for visualization
    """

    def __init__(self, n_clusters: int = 20, dbscan_eps: float = 0.5,
                 pca_components: int = 50):
        self.n_clusters      = n_clusters
        self.dbscan_eps      = dbscan_eps
        self.pca_components  = pca_components
        self.kmeans          = None
        self.scaler          = None
        self.pca             = None

    def fit_kmeans(self, features: np.ndarray) -> np.ndarray:
        """
        Fit K-Means on features. Returns cluster labels (N,).
        """
        if not SKLEARN_OK:
            raise RuntimeError("scikit-learn required for clustering.")

        logger.info(f"[Cluster] Fitting K-Means (k={self.n_clusters}) on {features.shape}...")

        # Normalize + optional PCA reduction
        self.scaler = StandardScaler()
        features_norm = self.scaler.fit_transform(features)

        if features.shape[1] > self.pca_components:
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            features_reduced = self.pca.fit_transform(features_norm)
            logger.info(f"[Cluster] PCA: {features.shape[1]} → {self.pca_components} dims "
                        f"({self.pca.explained_variance_ratio_.sum():.1%} variance)")
        else:
            features_reduced = features_norm

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42,
                              n_init=10, max_iter=300)
        labels = self.kmeans.fit_predict(features_reduced)

        unique, counts = np.unique(labels, return_counts=True)
        logger.info(f"[Cluster] K-Means complete. Cluster sizes: "
                    f"{dict(zip(unique.tolist(), counts.tolist()))}")
        return labels

    def find_anomalies(self, features: np.ndarray) -> np.ndarray:
        """
        DBSCAN for anomaly detection. Returns boolean mask (True = anomaly).
        """
        if not SKLEARN_OK:
            return np.zeros(len(features), dtype=bool)

        logger.info(f"[Cluster] Running DBSCAN (eps={self.dbscan_eps})...")
        if self.scaler is None:
            self.scaler = StandardScaler()
            features_norm = self.scaler.fit_transform(features)
        else:
            features_norm = self.scaler.transform(features)

        if features_norm.shape[1] > 50:
            pca = PCA(n_components=50, random_state=42)
            features_norm = pca.fit_transform(features_norm)

        db = DBSCAN(eps=self.dbscan_eps, min_samples=5, n_jobs=-1)
        db_labels = db.fit_predict(features_norm)
        anomalies = db_labels == -1
        logger.info(f"[Cluster] DBSCAN found {anomalies.sum()} anomalies "
                    f"({anomalies.mean():.1%} of dataset).")
        return anomalies

    def report(self, labels: np.ndarray, paths: List[str],
                anomalies: Optional[np.ndarray] = None) -> Dict:
        """Generate cluster statistics report."""
        unique, counts = np.unique(labels, return_counts=True)
        report_data = {
            "n_clusters": len(unique),
            "total_samples": len(labels),
            "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
        }
        if anomalies is not None:
            report_data["n_anomalies"] = int(anomalies.sum())
            report_data["anomaly_rate"] = float(anomalies.mean())

        # Sample paths per cluster
        cluster_samples = {}
        for cid in unique[:5]:   # First 5 clusters
            mask = labels == cid
            cluster_samples[int(cid)] = [paths[i] for i in np.where(mask)[0][:3]]
        report_data["sample_paths"] = cluster_samples

        return report_data


# ── SimCLR Contrastive Trainer ────────────────────────────────────────────────

class SimCLRTrainer:
    """
    Self-supervised contrastive pretraining of the CNN-OCR backbone.
    No labels needed — uses image augmentation pairs.

    After training, the backbone produces better features for:
      - Phase 1 fine-tuning (faster convergence)
      - Phase 2 clustering (more semantically meaningful clusters)
      - Phase 3 RL (richer state representation)
    """

    def __init__(self, backbone, projection_dim: int = 128,
                 device=None, temperature: float = 0.07):
        if not TORCH_OK:
            raise RuntimeError("PyTorch required.")

        self.device      = device or (torch.device("cuda") if torch.cuda.is_available()
                                        else torch.device("cpu"))
        self.temperature = temperature

        # Import here to avoid circular deps
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from inference.cnn_ocr.model import SimCLRProjectionHead

        self.backbone   = backbone.to(self.device)
        self.proj_head  = SimCLRProjectionHead(output_dim=projection_dim).to(self.device)
        self.optimizer  = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.proj_head.parameters()),
            lr=3e-4,
        )

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent (normalized temperature-scaled cross-entropy) loss."""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)      # (2B, D)
        # Cosine similarity matrix
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        # Positive pairs: (i, i+B) and (i+B, i)
        targets = torch.arange(B, device=z.device)
        targets = torch.cat([targets + B, targets])
        return F.cross_entropy(sim, targets)

    def train(self, img_dir: Path, epochs: int = 10, batch_size: int = 32,
               save_path: Optional[Path] = None):
        """Run SimCLR pretraining."""
        dataset = AugmentedPairDataset(img_dir)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)

        save_path = save_path or (MODEL_DIR / "cnn_ocr_simclr_backbone.pt")
        best_loss = float("inf")

        logger.info(f"[SimCLR] Training {epochs} epochs on {len(dataset)} images...")
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            self.backbone.train()
            self.proj_head.train()

            for view1, view2 in loader:
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                # For CRNN backbone: expects (B,1,32,W); adapt to (B,1,128,128)
                if view1.dim() == 4 and view1.size(1) == 3:
                    view1 = view1.mean(dim=1, keepdim=True)  # RGB → grayscale
                    view2 = view2.mean(dim=1, keepdim=True)

                try:
                    h1 = self.backbone(view1)
                    h2 = self.backbone(view2)
                    # Global average pool if needed
                    if h1.dim() > 2:
                        h1 = h1.mean(dim=[-1, -2]) if h1.dim() == 4 else h1.view(h1.size(0), -1)
                        h2 = h2.mean(dim=[-1, -2]) if h2.dim() == 4 else h2.view(h2.size(0), -1)
                except Exception:
                    # ResNet fallback: h1 is already (B, D)
                    pass

                z1 = self.proj_head(h1)
                z2 = self.proj_head(h2)

                loss = self.contrastive_loss(z1, z2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            logger.info(f"[SimCLR] Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.backbone.state_dict(), save_path)
                logger.info(f"[SimCLR] Saved best backbone → {save_path}")

        logger.info(f"[SimCLR] Training complete. Best loss: {best_loss:.4f}")
        return best_loss


# ── Scene Embedding Trainer ───────────────────────────────────────────────────

class SceneEmbeddingTrainer:
    """
    Trains the SceneEncoder autoencoder on full frames.
    The encoder's 256-dim output feeds into the RL agent as state.
    """

    def __init__(self, device=None):
        if not TORCH_OK:
            raise RuntimeError("PyTorch required.")

        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from inference.cnn_ocr.model import SceneEncoder

        self.device  = device or (torch.device("cuda") if torch.cuda.is_available()
                                    else torch.device("cpu"))
        self.model   = SceneEncoder(latent_dim=256).to(self.device)
        self.optim   = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.save_path = MODEL_DIR / "scene_encoder.pt"

    def train(self, img_dir: Path, epochs: int = 15, batch_size: int = 16):
        transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        dataset = UnlabeledImageDataset(img_dir, transform=transform)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True)

        logger.info(f"[SceneEnc] Training {epochs} epochs on {len(dataset)} images...")
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            ep_loss = 0.0
            self.model.train()
            for imgs, _ in loader:
                imgs = imgs.to(self.device)
                z, recon = self.model(imgs)
                loss = F.mse_loss(recon, imgs)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                ep_loss += loss.item()

            avg = ep_loss / len(loader)
            logger.info(f"[SceneEnc] Epoch {epoch}/{epochs} | MSE: {avg:.6f}")
            if avg < best_loss:
                best_loss = avg
                torch.save(self.model.state_dict(), self.save_path)

        logger.info(f"[SceneEnc] Done. Best MSE: {best_loss:.6f}. Saved → {self.save_path}")
        return self.model


# ── CLI ───────────────────────────────────────────────────────────────────────

import sys

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 Unsupervised Training")
    parser.add_argument("--task", choices=["features", "cluster", "simclr", "scene_enc"],
                        default="cluster")
    parser.add_argument("--img-dir", default=str(ROOT / "training" / "data" / "raw"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--n-clusters", type=int, default=20)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    save_dir = ROOT / "training" / "data" / "features"

    if args.task == "features":
        ext = FeatureExtractor()
        features, paths = ext.extract(img_dir)
        ext.save(features, paths, save_dir)

    elif args.task == "cluster":
        if not (save_dir / "features.npy").exists():
            print("Run --task features first.")
            sys.exit(1)
        features, paths = FeatureExtractor.load(save_dir)
        ca = ClusterAnalyzer(n_clusters=args.n_clusters)
        labels    = ca.fit_kmeans(features)
        anomalies = ca.find_anomalies(features)
        report    = ca.report(labels, paths, anomalies)
        import json
        print(json.dumps(report, indent=2, default=str))

    elif args.task == "simclr":
        ext = FeatureExtractor()
        trainer = SimCLRTrainer(backbone=ext.model)
        trainer.train(img_dir, epochs=args.epochs, batch_size=args.batch)

    elif args.task == "scene_enc":
        trainer = SceneEmbeddingTrainer()
        trainer.train(img_dir, epochs=args.epochs, batch_size=args.batch)
