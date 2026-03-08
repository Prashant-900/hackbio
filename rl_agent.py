"""
rl_agent.py — Deep Q-Network (DQN) for bacterial decision-making.

Each bacterium observes a 14-dimensional state vector (local chemistry,
internal health, physics) and selects from 7 discrete actions.  A single
shared DQN brain is trained from the pooled experience of all bacteria —
evolutionary selection pressure acts as the implicit curriculum.

State space (14 dims):
  [0]  local_resource         S at (x,y) normalised
  [1]  local_antibiotic       AB at (x,y) normalised
  [2]  local_toxin            foreign toxin normalised
  [3]  local_signal           QS signal normalised
  [4]  local_biofilm          EPS density normalised
  [5]  biomass                own biomass / 3
  [6]  age_fraction           age / max_age
  [7]  phase                  phase int / 3
  [8]  resistance             antibiotic resistance [0,1]
  [9]  fitness                current fitness [0,1]
  [10] temperature_norm       Cardinal T normalised
  [11] pressure_norm          pressure normalised
  [12] z_position             depth normalised
  [13] biofilm_member         0 or 1

Action space (7 discrete):
  0  CHEMOTAXIS_RUN     Move toward nutrient gradient
  1  CHEMOTAXIS_TUMBLE  Random direction
  2  COOPERATE          Increase EPS / public-good secretion
  3  COMPETE            Increase bacteriocin production
  4  CONJUGATE          Attempt HGT with neighbour
  5  CONSERVE           Reduce metabolism to save energy
  6  GROW               Default metabolic behaviour

References:
  Mnih et al. (2015) Human-level control through deep RL, Nature.
  van Hasselt et al. (2016) Deep RL with Double Q-learning, AAAI.
"""

from __future__ import annotations

import random
from collections import deque
from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gpu_utils import get_device

# ── Constants ────────────────────────────────────────────────
STATE_DIM = 14
PHASE_INT_MAP = {"LAG": 0, "LOG": 1, "STATIONARY": 2, "DEATH": 3}


class BacterialAction(IntEnum):
    CHEMOTAXIS_RUN = 0
    CHEMOTAXIS_TUMBLE = 1
    COOPERATE = 2
    COMPETE = 3
    CONJUGATE = 4
    CONSERVE = 5
    GROW = 6


ACTION_DIM = len(BacterialAction)


# ── Neural Network ───────────────────────────────────────────
class DQNetwork(nn.Module):
    """Feed-forward Q-network: state → Q-values for each action."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Experience Replay ────────────────────────────────────────
class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ── State / Reward helpers ───────────────────────────────────
def extract_state(bact, env, cfg: dict) -> np.ndarray:
    """Build the 14-dim observation vector for a single bacterium."""
    res_max = cfg.get("resource", {}).get("max_concentration", 30.0)
    ab_max = cfg.get("antibiotic", {}).get("max_concentration", 8.0)
    max_age = cfg.get("bacterium", {}).get("max_age", 300)
    z_levels = cfg.get("grid", {}).get("z_levels", 10)

    return np.array(
        [
            min(1.0, env.resource[bact.y, bact.x] / max(res_max, 1e-9)),
            min(1.0, env.antibiotic[bact.y, bact.x] / max(ab_max, 1e-9)),
            min(1.0, env.get_foreign_toxin(bact.genotype.id, bact.x, bact.y) / 5.0),
            min(1.0, env.signal[bact.y, bact.x] / 2.0),
            min(1.0, env.biofilm[bact.y, bact.x] / 5.0),
            min(1.0, bact.biomass / 3.0),
            min(1.0, bact.age / max(max_age, 1)),
            PHASE_INT_MAP.get(bact.phase.name, 0) / 3.0,
            min(1.0, bact.genotype.antibiotic_resistance),
            min(1.0, max(0.0, bact.fitness)),
            min(1.0, max(0.0, (getattr(env, "temperature", 37.0) - 10.0) / 35.0)),
            min(1.0, max(0.0, (getattr(env, "pressure_atm", 1.0) - 0.5) / 5.0)),
            min(1.0, getattr(bact, "z", 0.0) / max(z_levels, 1)),
            float(bact.biofilm_member),
        ],
        dtype=np.float32,
    )


def extract_states_batch(agents: list, env, cfg: dict) -> np.ndarray:
    """Fully vectorised state extraction — no per-agent Python loop."""
    n = len(agents)
    if n == 0:
        return np.empty((0, STATE_DIM), dtype=np.float32)

    res_max = max(cfg.get("resource", {}).get("max_concentration", 30.0), 1e-9)
    ab_max = max(cfg.get("antibiotic", {}).get("max_concentration", 8.0), 1e-9)
    max_age = max(cfg.get("bacterium", {}).get("max_age", 300), 1)
    z_levels = max(cfg.get("grid", {}).get("z_levels", 10), 1)
    temp = getattr(env, "temperature", 37.0)
    pressure = getattr(env, "pressure_atm", 1.0)

    # Pre-extract arrays from agent objects
    xs = np.array([a.x for a in agents], dtype=np.intp)
    ys = np.array([a.y for a in agents], dtype=np.intp)

    states = np.empty((n, STATE_DIM), dtype=np.float32)
    states[:, 0] = np.minimum(1.0, env.resource[ys, xs] / res_max)
    states[:, 1] = np.minimum(1.0, env.antibiotic[ys, xs] / ab_max)

    # Foreign toxin — vectorised per-genotype
    gids = np.array([a.genotype.id for a in agents], dtype=np.intp)
    tox = np.zeros(n, dtype=np.float32)
    for gid in np.unique(gids):
        mask = gids == gid
        idxs = np.where(mask)[0]
        tox_sum = np.zeros(env.resource.shape, dtype=np.float64)
        for g, grid in env.toxin_grids.items():
            if g != gid:
                tox_sum += grid
        tox[idxs] = tox_sum[ys[idxs], xs[idxs]]
    states[:, 2] = np.minimum(1.0, tox / 5.0)

    states[:, 3] = np.minimum(1.0, env.signal[ys, xs] / 2.0)
    states[:, 4] = np.minimum(1.0, env.biofilm[ys, xs] / 5.0)
    states[:, 5] = np.minimum(1.0, np.array([a.biomass for a in agents], dtype=np.float32) / 3.0)
    states[:, 6] = np.minimum(1.0, np.array([a.age for a in agents], dtype=np.float32) / max_age)
    states[:, 7] = np.array([PHASE_INT_MAP.get(a.phase.name, 0) for a in agents], dtype=np.float32) / 3.0
    states[:, 8] = np.minimum(1.0, np.array([a.genotype.antibiotic_resistance for a in agents], dtype=np.float32))
    states[:, 9] = np.clip(np.array([a.fitness for a in agents], dtype=np.float32), 0.0, 1.0)
    states[:, 10] = np.clip((temp - 10.0) / 35.0, 0.0, 1.0)
    states[:, 11] = np.clip((pressure - 0.5) / 5.0, 0.0, 1.0)
    states[:, 12] = np.minimum(1.0, np.array([getattr(a, 'z', 0.0) for a in agents], dtype=np.float32) / z_levels)
    states[:, 13] = np.array([float(a.biofilm_member) for a in agents], dtype=np.float32)
    return states


def compute_reward(
    bact, prev_biomass: float, divided: bool, alive: bool
) -> float:
    """Compute single-step reward for RL training.

    +0.1  alive bonus (survival matters)
    +Δm×2 biomass gain (metabolic efficiency)
    +5.0  successful division (reproductive fitness)
    +0.3  biofilm membership (cooperation pays off)
    −10   death penalty
    """
    reward = 0.1
    reward += (bact.biomass - prev_biomass) * 2.0
    if divided:
        reward += 5.0
    if not alive:
        reward -= 10.0
    if bact.biofilm_member:
        reward += 0.3
    return reward


# ── DQN Controller ───────────────────────────────────────────
class BacterialDQN:
    """Shared DQN brain for all bacteria.  GPU-accelerated when available.

    Uses Double DQN (van Hasselt 2016) with soft target-network updates
    and experience replay.  Epsilon-greedy exploration decays over epochs.
    """

    def __init__(self, cfg: dict):
        rl = cfg.get("rl", {})
        self.device = get_device(rl.get("force_cpu", False))
        self.gamma: float = rl.get("gamma", 0.99)
        self.epsilon: float = rl.get("epsilon_start", 1.0)
        self.epsilon_min: float = rl.get("epsilon_min", 0.05)
        self.epsilon_decay: float = rl.get("epsilon_decay", 0.995)
        self.batch_size: int = rl.get("batch_size", 64)
        self.target_update_freq: int = rl.get("target_update_freq", 10)
        self.lr: float = rl.get("learning_rate", 1e-3)
        self.tau: float = rl.get("tau", 0.005)
        self.train_every: int = rl.get("train_every", 5)
        self.enabled: bool = rl.get("enabled", True)

        self.policy_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.replay = ReplayBuffer(rl.get("buffer_size", 50_000))
        self.steps: int = 0
        self._losses: list[float] = []

    # ── Action selection ─────────────────────────────────────
    def select_action(self, state: np.ndarray) -> int:
        if not self.enabled:
            return int(BacterialAction.GROW)
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.policy_net(t).argmax(1).item())

    def select_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """Batch inference on GPU for all alive bacteria."""
        if not self.enabled:
            return np.full(len(states), int(BacterialAction.GROW), dtype=np.int64)
        n = len(states)
        actions = np.zeros(n, dtype=np.int64)
        explore = np.random.random(n) < self.epsilon
        actions[explore] = np.random.randint(0, ACTION_DIM, size=int(explore.sum()))
        exploit = ~explore
        if exploit.any():
            with torch.no_grad():
                t = torch.FloatTensor(states[exploit]).to(self.device)
                actions[exploit] = self.policy_net(t).argmax(1).cpu().numpy()
        return actions

    # ── Training ─────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Single Double-DQN update from the replay buffer."""
        if len(self.replay) < self.batch_size:
            return None
        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best = self.policy_net(ns).argmax(1)
            q_next = self.target_net(ns).gather(1, best.unsqueeze(1)).squeeze(1)
            target = r + self.gamma * q_next * (1 - d)

        loss = nn.SmoothL1Loss()(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            for tp, pp in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        lv = loss.item()
        self._losses.append(lv)
        return lv

    def stats(self) -> dict:
        """Snapshot of RL training state for the dashboard."""
        return {
            "rl_enabled": self.enabled,
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.replay),
            "train_steps": self.steps,
            "avg_loss": (
                round(float(np.mean(self._losses[-100:])), 6)
                if self._losses
                else 0.0
            ),
            "device": str(self.device),
        }
