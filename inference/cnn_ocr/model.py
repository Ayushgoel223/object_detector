"""
BlindAid — CNN-OCR Model Definitions
=======================================
Two-stage architecture:

Stage 1 — TextRegionCNN
    ResNet-18 backbone (ImageNet pre-trained) with a lightweight
    detection head that outputs text-region bounding boxes.
    Inspired by CRAFT (Character Region Awareness For Text detection).

Stage 2 — CRNN (Convolutional Recurrent Neural Network)
    For each text-region crop:
      CNN feature extractor (VGG-style conv stack)
      → Bidirectional GRU sequence encoder
      → CTC (Connectionist Temporal Classification) head
      → decoded character sequence

Training phases:
  Phase 1 (Supervised)   : CTC loss on labeled word images (Synth90K / IIIT5K / SVT)
  Phase 2 (Unsupervised) : SimCLR contrastive pretraining on unlabeled crops
  Phase 3 (RL)           : CRNN output feeds into RL state vector

References:
  - CRNN: Shi et al. 2016 (https://arxiv.org/abs/1507.05717)
  - CRAFT: Baek et al. 2019 (https://arxiv.org/abs/1904.01941)
  - SimCLR: Chen et al. 2020 (https://arxiv.org/abs/2002.05709)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ── Character Vocabulary ──────────────────────────────────────────────────────
# 94 printable ASCII characters + blank (CTC blank token at index 0)

CHARSET = (
    "-"                                # index 0 = CTC blank
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    r" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)
NUM_CHARS = len(CHARSET)               # ~95
BLANK_IDX = 0


def char_to_idx(c: str) -> int:
    return CHARSET.index(c) if c in CHARSET else BLANK_IDX


def idx_to_char(i: int) -> str:
    return CHARSET[i] if 0 <= i < len(CHARSET) else ""


# ── Stage 1: TextRegionCNN ────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm + ReLU building block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TextRegionHead(nn.Module):
    """
    Lightweight detection head on top of ResNet-18 features.
    Outputs a heatmap (H/4 × W/4) indicating text-region probability.
    """
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.upconv1  = ConvBnRelu(in_channels, 256)
        self.upconv2  = ConvBnRelu(256, 128)
        self.upconv3  = ConvBnRelu(128, 64)
        self.heatmap  = nn.Conv2d(64, 2, kernel_size=1)   # [text_prob, link_prob]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.upconv2(x)
        x = self.upconv3(x)
        return torch.sigmoid(self.heatmap(x))   # (B, 2, H/4, W/4)


class TextRegionCNN(nn.Module):
    """
    ResNet-18 backbone + TextRegionHead.
    Input  : (B, 3, H, W)  BGR frame, normalized
    Output : (B, 2, H/4, W/4) text/link heatmap
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        except Exception:
            # Older torchvision API
            import torchvision.models as models
            backbone = models.resnet18(pretrained=pretrained)

        # Remove avgpool + fc; keep feature layers
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # out: 64 ch
        self.layer2 = backbone.layer2   # out: 128 ch
        self.layer3 = backbone.layer3   # out: 256 ch
        self.layer4 = backbone.layer4   # out: 512 ch
        self.head   = TextRegionHead(in_channels=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)

    def freeze_backbone(self):
        """Freeze ResNet weights; only train the detection head."""
        for name, param in self.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


# ── Stage 2: CRNN ─────────────────────────────────────────────────────────────

class CRNNFeatureExtractor(nn.Module):
    """
    VGG-style CNN for per-crop feature extraction.
    Input  : (B, 1, 32, W)   grayscale word crop, height normalized to 32px
    Output : (B, 512, 1, W') feature map collapsed to height=1
    """

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1
            ConvBnRelu(1, 64),
            nn.MaxPool2d(2, 2),                        # → (64, 16, W/2)
            # Block 2
            ConvBnRelu(64, 128),
            nn.MaxPool2d(2, 2),                        # → (128, 8, W/4)
            # Block 3
            ConvBnRelu(128, 256),
            ConvBnRelu(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),              # → (256, 4, W/4)
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),              # → (512, 2, W/4)
            # Block 5 — squeeze height to 1
            nn.Conv2d(512, 512, (2, 1), bias=False),   # → (512, 1, W/4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)   # (B, 512, 1, W')


class BidirectionalGRU(nn.Module):
    """Stacked bidirectional GRU for sequence modeling."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)            # (B, T, 2*hidden)
        return self.fc(out)             # (B, T, output_size)


class CRNN(nn.Module):
    """
    Full CRNN: CNN feature extractor → BiGRU → CTC head.

    Input  : (B, 1, 32, W) grayscale crop
    Output : (T, B, num_chars) log-softmax logits for CTC loss

    Decode with ctc_greedy_decode() or beam_search_decode().
    """

    def __init__(self, num_chars: int = NUM_CHARS, hidden_size: int = 256):
        super().__init__()
        self.cnn     = CRNNFeatureExtractor()
        self.bigru   = BidirectionalGRU(512, hidden_size, num_chars, num_layers=2)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN features: (B, 512, 1, W')
        features = self.cnn(x)

        # Reshape for RNN: (B, W', 512)
        B, C, H, W = features.shape
        features = features.squeeze(2)          # (B, 512, W')
        features = features.permute(0, 2, 1)    # (B, W', 512)

        # GRU: (B, W', num_chars)
        logits = self.bigru(features)

        # CTC expects (T, B, C)
        logits = logits.permute(1, 0, 2)        # (T, B, num_chars)
        return self.log_softmax(logits)


# ── CTC Decoder ───────────────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = BLANK_IDX) -> List[str]:
    """
    Greedy CTC decoder.

    Args:
        log_probs: (T, B, C) output from CRNN
        blank:     blank token index

    Returns:
        List of decoded strings, one per batch element.
    """
    # Argmax over classes
    indices = log_probs.argmax(dim=2)  # (T, B)
    indices = indices.permute(1, 0)    # (B, T)

    results = []
    for seq in indices:
        seq = seq.tolist()
        # Collapse repeats and remove blanks
        prev = None
        chars = []
        for idx in seq:
            if idx != prev:
                if idx != blank:
                    chars.append(idx_to_char(idx))
                prev = idx
        results.append("".join(chars))
    return results


def ctc_beam_decode(log_probs: torch.Tensor, beam_width: int = 10,
                    blank: int = BLANK_IDX) -> List[Tuple[str, float]]:
    """
    Simple beam search CTC decoder (single batch element).
    Returns list of (text, score) for top beam_width candidates.
    """
    T, B, C = log_probs.shape
    assert B == 1, "Beam decode supports single item only."

    probs = log_probs[:, 0, :].exp().cpu().numpy()  # (T, C)

    # Beam: list of (prefix_text, prob_not_blank, prob_blank)
    beams = [("", 0.0, 1.0)]

    for t in range(T):
        new_beams = {}
        for prefix, p_nb, p_b in beams:
            for c in range(C):
                p = float(probs[t, c])
                if c == blank:
                    # Extend with blank
                    key = prefix
                    prev_nb, prev_b = new_beams.get(key, (0.0, 0.0))
                    new_beams[key] = (prev_nb, prev_b + (p_nb + p_b) * p)
                else:
                    char = idx_to_char(c)
                    # Extend with character
                    if prefix and prefix[-1] == char:
                        # Same char — only blank can extend
                        key = prefix + char
                        prev_nb, prev_b = new_beams.get(key, (0.0, 0.0))
                        new_beams[key] = (prev_nb + p_b * p, prev_b)
                    else:
                        key = prefix + char
                        prev_nb, prev_b = new_beams.get(key, (0.0, 0.0))
                        new_beams[key] = (prev_nb + (p_nb + p_b) * p, prev_b)

        # Prune to top beam_width
        scored = [(k, v[0] + v[1]) for k, v in new_beams.items()]
        scored.sort(key=lambda x: -x[1])
        beams = [(k, new_beams[k][0], new_beams[k][1]) for k, _ in scored[:beam_width]]

    return [(b[0], b[1] + b[2]) for b in beams]


# ── SimCLR Projection Head (Phase 2 Unsupervised) ────────────────────────────

class SimCLRProjectionHead(nn.Module):
    """
    Non-linear projection head for SimCLR contrastive pretraining.
    Attached on top of CRNN CNN features during Phase 2.
    Discarded after pretraining; only the CNN backbone is kept.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# ── Scene Autoencoder (Phase 2 — full frame embedding) ───────────────────────

class SceneEncoder(nn.Module):
    """
    Lightweight CNN autoencoder for full-frame scene embedding.
    Encoder output (256-dim) is used as RL state vector.
    Input : (B, 3, H, W) — resized to (3, 128, 128) before passing
    Output: (B, 256) latent vector
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBnRelu(3, 32, stride=2),     # (32, 64, 64)
            ConvBnRelu(32, 64, stride=2),    # (64, 32, 32)
            ConvBnRelu(64, 128, stride=2),   # (128, 16, 16)
            ConvBnRelu(128, 256, stride=2),  # (256, 8, 8)
            nn.AdaptiveAvgPool2d(1),         # (256, 1, 1)
            nn.Flatten(),                    # (256,)
            nn.Linear(256, latent_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # (128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # (64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # (32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # (3, 128, 128)
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z    = self.encode(x)
        recon = self.decode(z)
        return z, recon


# ── DQN Network (Phase 3 RL) ─────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for navigation alert optimization.

    State  : 256-dim scene encoding + detection features + OCR features
    Actions: 6 discrete — see training/phase3_rl/environment.py
    """
    def __init__(self, state_dim: int = 256 + 64 + 32,  # scene + det + ocr
                 num_actions: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        shared = self.shared(state)
        value  = self.value_stream(shared)
        adv    = self.advantage_stream(shared)
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        q = value + adv - adv.mean(dim=1, keepdim=True)
        return q


# ── Model Factory ─────────────────────────────────────────────────────────────

def build_text_region_cnn(pretrained: bool = True) -> TextRegionCNN:
    return TextRegionCNN(pretrained=pretrained)


def build_crnn(num_chars: int = NUM_CHARS, hidden_size: int = 256) -> CRNN:
    return CRNN(num_chars=num_chars, hidden_size=hidden_size)


def build_scene_encoder(latent_dim: int = 256) -> SceneEncoder:
    return SceneEncoder(latent_dim=latent_dim)


def build_dqn(state_dim: int = 352, num_actions: int = 6) -> DuelingDQN:
    return DuelingDQN(state_dim=state_dim, num_actions=num_actions)


def get_device() -> torch.device:
    """Auto-select CUDA, MPS (Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    # Quick sanity check
    device = get_device()
    print(f"Device: {device}")

    # Text Region CNN
    tr = build_text_region_cnn(pretrained=False).to(device)
    dummy_frame = torch.randn(1, 3, 640, 640).to(device)
    hmap = tr(dummy_frame)
    print(f"TextRegionCNN output: {hmap.shape}")   # (1, 2, 160, 160)

    # CRNN
    crnn = build_crnn().to(device)
    dummy_crop = torch.randn(1, 1, 32, 128).to(device)
    logits = crnn(dummy_crop)
    print(f"CRNN output: {logits.shape}")           # (T, 1, 95)
    decoded = ctc_greedy_decode(logits)
    print(f"Greedy decode: '{decoded[0]}'")

    # Scene encoder
    enc = build_scene_encoder().to(device)
    dummy_scene = torch.randn(1, 3, 128, 128).to(device)
    z, recon = enc(dummy_scene)
    print(f"SceneEncoder latent: {z.shape}, recon: {recon.shape}")

    # DQN
    dqn = build_dqn().to(device)
    state = torch.randn(1, 352).to(device)
    q_values = dqn(state)
    print(f"DQN Q-values: {q_values.shape}")       # (1, 6)
    print("All model checks passed ✓")
