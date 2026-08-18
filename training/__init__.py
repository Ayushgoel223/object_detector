"""
BlindAid — Training Package
=============================
Three-phase ML training pipeline:

Phase 1 — Supervised Learning
  - YOLOv8 fine-tuning on internet-fetched navigation datasets
  - CRNN training on OCR word datasets (IIIT5K, SVT, Synth90K)

Phase 2 — Unsupervised Learning
  - SimCLR contrastive pretraining (no labels needed)
  - K-Means + DBSCAN clustering on CNN features
  - SceneEncoder autoencoder for RL state representation

Phase 3 — Reinforcement Learning
  - Dueling DQN with experience replay
  - Reward-shaped navigation alert optimization

Usage:
  python training/internet_fetcher.py    # Fetch data
  python training/phase1_supervised.py   # Train detectors
  python training/phase2_unsupervised.py # Cluster + SimCLR
  python training/phase3_rl.py           # RL agent
"""

__version__ = "2.0.0"
