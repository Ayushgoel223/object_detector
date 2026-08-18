"""
BlindAid — Phase 3: Reinforcement Learning Agent
===================================================
Deep Q-Network (Dueling DQN) that learns the optimal policy for
announcing obstacles and OCR-detected text to a visually impaired user.

STATE  : scene_encoder(frame) + detection_features + ocr_features = 352-dim
ACTIONS: 6 discrete actions
  0 — ANNOUNCE_OBSTACLE  (speak the top obstacle)
  1 — ANNOUNCE_OCR_TEXT  (speak the OCR-detected text)
  2 — ANNOUNCE_CLEAR     ("Path is clear, proceed")
  3 — HOLD_SILENT        (say nothing this cycle)
  4 — INCREASE_URGENCY   (boost the urgency threshold → more sensitive)
  5 — DECREASE_URGENCY   (lower the urgency threshold → less sensitive)

REWARD:
  +10  : ANNOUNCE_OBSTACLE when CRITICAL obstacle present
  +5   : ANNOUNCE_OCR_TEXT when navigation-relevant sign detected
  +2   : ANNOUNCE_CLEAR when no obstacle and path is truly clear
  +1   : HOLD_SILENT when nothing important (correct silence)
  -2   : HOLD_SILENT when CRITICAL obstacle is present (danger!)
  -1   : Redundant announcement (same object in <2s)
  -0.5 : Any announcement when path is actually clear (false alarm)

TRAINING: Runs on pre-recorded video clips from Phase 1 dataset.
          Simulates a "ground-truth" environment using detection labels.
"""

import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[RL] %(message)s")

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    logger.error("PyTorch required for Phase 3 RL.")


# ── Action & State Constants ──────────────────────────────────────────────────

ACTION_NAMES = [
    "ANNOUNCE_OBSTACLE",
    "ANNOUNCE_OCR_TEXT",
    "ANNOUNCE_CLEAR",
    "HOLD_SILENT",
    "INCREASE_URGENCY",
    "DECREASE_URGENCY",
]
NUM_ACTIONS  = len(ACTION_NAMES)
STATE_DIM    = 352   # 256 (scene) + 64 (detection feats) + 32 (ocr feats)


# ── Transition ────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Experience replay buffer with optional MySQL persistence.
    Stores up to `capacity` transitions in memory.
    """

    def __init__(self, capacity: int = 50_000, db_manager=None):
        self.buffer     = deque(maxlen=capacity)
        self.db         = db_manager
        self._episode   = 0
        self._step      = 0

    def push(self, transition: Transition):
        self.buffer.append(transition)
        self._step += 1

        # Persist to DB periodically (every 100 steps)
        if self.db and self._step % 100 == 0:
            try:
                self.db.insert_rl_episode(
                    episode=self._episode,
                    step=self._step,
                    state=transition.state.tolist()[:16],  # Only first 16 dims (space-saving)
                    action=transition.action,
                    reward=float(transition.reward),
                    next_state=transition.next_state.tolist()[:16],
                    done=transition.done,
                )
            except Exception:
                pass

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def new_episode(self):
        self._episode += 1
        self._step = 0


# ── Simulated Environment ─────────────────────────────────────────────────────

class BlindAidEnv:
    """
    Gym-like environment for RL training.

    Observations are synthesized from:
      - Stored detection records (from Phase 1 training data)
      - Simulated scene encoder outputs
      - Simulated OCR flags

    In production, these come from the live inference pipeline.
    """

    def __init__(self, scenarios: List[dict] = None):
        """
        Args:
            scenarios: List of scenario dicts, each with keys:
                {has_critical, has_near, has_ocr_sign, is_clear, scene_context}
        """
        self.scenarios    = scenarios or self._default_scenarios()
        self._idx         = 0
        self._step_count  = 0
        self._max_steps   = 20
        self._last_action = -1
        self._last_action_time = -10
        self._urgency_adj = 0.0   # modified by INCREASE/DECREASE_URGENCY actions

    def _default_scenarios(self) -> List[dict]:
        """Built-in training scenarios covering all reward cases."""
        return [
            {"has_critical": True,  "has_near": False, "has_ocr_sign": False, "is_clear": False},
            {"has_critical": False, "has_near": True,  "has_ocr_sign": False, "is_clear": False},
            {"has_critical": False, "has_near": False, "has_ocr_sign": True,  "is_clear": False},
            {"has_critical": False, "has_near": False, "has_ocr_sign": False, "is_clear": True},
            {"has_critical": True,  "has_near": False, "has_ocr_sign": True,  "is_clear": False},
            {"has_critical": False, "has_near": True,  "has_ocr_sign": True,  "is_clear": False},
        ] * 50  # Repeat for a full episode pool

    def reset(self) -> np.ndarray:
        random.shuffle(self.scenarios)
        self._idx         = 0
        self._step_count  = 0
        self._last_action = -1
        self._last_action_time = -10
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action, return (next_state, reward, done).
        """
        scenario = self.scenarios[self._idx % len(self.scenarios)]
        reward   = self._compute_reward(action, scenario)

        self._last_action      = action
        self._last_action_time = self._step_count
        self._step_count      += 1

        # Advance urgency if adjusted
        if action == 4:
            self._urgency_adj = min(0.3, self._urgency_adj + 0.05)
        elif action == 5:
            self._urgency_adj = max(-0.2, self._urgency_adj - 0.05)

        self._idx    += 1
        done          = self._step_count >= self._max_steps
        next_state    = self._get_state()
        return next_state, reward, done

    def _get_state(self) -> np.ndarray:
        """
        Construct a 352-dim state vector:
          [0:256]   — scene encoder output (simulated as random + scenario signal)
          [256:320] — detection features (64-dim)
          [320:352] — OCR features (32-dim)
        """
        scenario = self.scenarios[self._idx % len(self.scenarios)]

        rng = np.random.default_rng(self._step_count)

        # Scene encoding (simulated)
        scene = rng.standard_normal(256).astype(np.float32) * 0.1
        if scenario["has_critical"]:
            scene[0:8]   += 2.0   # Critical signal in first 8 dims
        if scenario["has_near"]:
            scene[8:16]  += 1.0
        if scenario["is_clear"]:
            scene[16:24] += 1.5

        # Detection features (64-dim)
        det_feats = np.zeros(64, dtype=np.float32)
        if scenario["has_critical"]:
            det_feats[0]  = 1.0   # critical flag
            det_feats[1]  = 0.9   # confidence proxy
        elif scenario["has_near"]:
            det_feats[0]  = 0.5
            det_feats[1]  = 0.7
        det_feats[2] = self._urgency_adj   # current threshold adjustment

        # OCR features (32-dim)
        ocr_feats = np.zeros(32, dtype=np.float32)
        if scenario["has_ocr_sign"]:
            ocr_feats[0]  = 1.0
            ocr_feats[1]  = 0.85

        return np.concatenate([scene, det_feats, ocr_feats])

    def _compute_reward(self, action: int, scenario: dict) -> float:
        reward = 0.0

        if action == 0:  # ANNOUNCE_OBSTACLE
            if scenario["has_critical"]:
                reward = 10.0
            elif scenario["has_near"]:
                reward = 4.0
            elif scenario["is_clear"]:
                reward = -0.5   # False alarm

        elif action == 1:  # ANNOUNCE_OCR_TEXT
            if scenario["has_ocr_sign"]:
                reward = 5.0
            else:
                reward = -0.5   # Nothing to announce

        elif action == 2:  # ANNOUNCE_CLEAR
            if scenario["is_clear"]:
                reward = 2.0
            elif scenario["has_critical"]:
                reward = -3.0   # Dangerous: said "clear" when CRITICAL present
            else:
                reward = -0.5

        elif action == 3:  # HOLD_SILENT
            if scenario["has_critical"]:
                reward = -2.0   # Dangerous silence
            elif scenario["has_near"]:
                reward = -0.5   # Missed a warning
            elif scenario["is_clear"]:
                reward = 1.0    # Correct silence
            else:
                reward = 0.3

        elif action in (4, 5):  # INCREASE/DECREASE_URGENCY
            reward = 0.0   # Neutral — evaluated indirectly

        # Penalty for redundant announcement (same action within 3 steps)
        if action == self._last_action and (self._step_count - self._last_action_time) < 3:
            if action in (0, 1, 2):  # Only penalize verbal announcements
                reward -= 1.0

        return reward


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Dueling DQN with:
      - Experience Replay (50K transitions)
      - Target network (updated every 500 steps)
      - ε-greedy exploration (1.0 → 0.05 over 10K steps)
      - Double DQN action selection
    """

    def __init__(self, state_dim: int = STATE_DIM, num_actions: int = NUM_ACTIONS,
                 gamma: float = 0.99, lr: float = 1e-4,
                 batch_size: int = 64, buffer_size: int = 50_000,
                 target_update_freq: int = 500, db_manager=None):
        if not TORCH_OK:
            raise RuntimeError("PyTorch required.")

        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from inference.cnn_ocr.model import DuelingDQN

        self.gamma           = gamma
        self.batch_size      = batch_size
        self.target_update   = target_update_freq
        self.num_actions     = num_actions
        self._step_count     = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online and target networks
        self.policy_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net = DuelingDQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(capacity=buffer_size, db_manager=db_manager)

        # Epsilon schedule
        self.eps_start = 1.0
        self.eps_end   = 0.05
        self.eps_decay = 10_000

    @property
    def epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self._step_count / self.eps_decay)

    def select_action(self, state: np.ndarray) -> Tuple[int, List[float]]:
        """ε-greedy action selection. Returns (action_idx, q_values)."""
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
            return action, []

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.policy_net(state_t).squeeze(0).cpu().numpy()
        return int(q_vals.argmax()), q_vals.tolist()

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(Transition(
            state=np.array(state, dtype=np.float32),
            action=action,
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32),
            done=done,
        ))

    def update(self) -> Optional[float]:
        """Sample a minibatch and update policy network. Returns loss or None."""
        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states      = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions     = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(batch.reward).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones       = torch.FloatTensor(batch.done).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1.0 - dones)

        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network
        self._step_count += 1
        if self._step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug(f"[RL] Target network updated at step {self._step_count}")

        return float(loss.item())

    def save(self, path: Path = None):
        path = path or (MODEL_DIR / "rl_agent.pt")
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state":  self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "step":          self._step_count,
        }, path)
        logger.info(f"[RL] Agent saved → {path}")

    def load(self, path: Path = None):
        path = path or (MODEL_DIR / "rl_agent.pt")
        if not path.exists():
            logger.warning(f"[RL] No checkpoint found at {path}")
            return
        ck = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ck["policy_state"])
        self.target_net.load_state_dict(ck["target_state"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self._step_count = ck.get("step", 0)
        logger.info(f"[RL] Agent loaded from {path} (step={self._step_count})")


# ── RL Trainer ────────────────────────────────────────────────────────────────

class RLTrainer:
    """Orchestrates DQN training over multiple episodes."""

    def __init__(self, agent: DQNAgent, env: BlindAidEnv, db_manager=None):
        self.agent = agent
        self.env   = env
        self.db    = db_manager

    def train(self, episodes: int = 500, log_interval: int = 50,
               save_interval: int = 100) -> List[float]:
        """
        Train for N episodes.
        Returns list of total rewards per episode.
        """
        episode_rewards = []
        best_reward     = float("-inf")

        logger.info(f"[RL] Starting training: {episodes} episodes, "
                    f"buffer_size={self.agent.buffer.buffer.maxlen}")

        for ep in range(1, episodes + 1):
            state   = self.env.reset()
            self.agent.buffer.new_episode()
            ep_reward = 0.0
            ep_losses = []

            while True:
                action, q_vals = self.agent.select_action(state)
                next_state, reward, done = self.env.step(action)

                self.agent.push(state, action, reward, next_state, done)
                loss = self.agent.update()
                if loss is not None:
                    ep_losses.append(loss)

                ep_reward += reward
                state = next_state

                if done:
                    break

            episode_rewards.append(ep_reward)

            if ep % log_interval == 0:
                avg_r = np.mean(episode_rewards[-log_interval:])
                avg_l = np.mean(ep_losses) if ep_losses else 0.0
                logger.info(f"[RL] Episode {ep:5d} | "
                            f"Avg Reward: {avg_r:+.2f} | "
                            f"Avg Loss: {avg_l:.4f} | "
                            f"ε: {self.agent.epsilon:.3f}")

                if self.db:
                    try:
                        self.db.insert_training_run(
                            run_id="rl_main",
                            phase="rl",
                            epoch=ep,
                            metrics={"loss": avg_l, "reward": avg_r},
                            notes=f"eps={self.agent.epsilon:.3f}",
                        )
                    except Exception:
                        pass

            if ep % save_interval == 0:
                self.agent.save()
                if ep_reward > best_reward:
                    best_reward = ep_reward
                    self.agent.save(MODEL_DIR / "rl_agent_best.pt")

        logger.info(f"[RL] Training complete. Best episode reward: {best_reward:.2f}")
        return episode_rewards


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 RL Training")
    parser.add_argument("--episodes",  type=int, default=500)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--load",      action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    if args.dry_run:
        print("=== RL Dry Run ===")
        env   = BlindAidEnv()
        state = env.reset()
        print(f"State shape: {state.shape}")
        for a in range(NUM_ACTIONS):
            ns, r, d = env.step(a)
            print(f"  Action {a} ({ACTION_NAMES[a]}): reward={r:.1f}")
        print("Dry run complete.")
    else:
        if not TORCH_OK:
            print("PyTorch not available. Cannot run RL training.")
            import sys; sys.exit(1)

        env   = BlindAidEnv()
        agent = DQNAgent()
        if args.load:
            agent.load()

        trainer = RLTrainer(agent, env)
        rewards = trainer.train(episodes=args.episodes)

        print(f"\nTraining summary:")
        print(f"  Total episodes : {len(rewards)}")
        print(f"  Best reward    : {max(rewards):.2f}")
        print(f"  Final avg(50)  : {np.mean(rewards[-50:]):.2f}")
