import os
import json
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


# --------------------------- CONFIG ---------------------------
@dataclass
class AgentConfig:
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 256
    buffer_size: int = 150_000
    min_buffer: int = 20_000

    # Softmax temperature
    tau_start: float = 5.0
    tau_end: float = 0.1
    tau_decay_steps: int = 5_000_000

    # PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 2_000_000
    per_eps: float = 1e-6

    polyak_tau: float = 0.01
    max_grad_norm: float = 1.0
    hidden_sizes: Tuple[int, ...] = (1024, 1024)
    update_every: int = 1
    num_envs: int = 64
    best_save_interval: int = 5000

    # Q-record while training
    q_record_interval: int = 2000  # record Q stats every N env steps
    q_record_batch: int = 512      # number of states to use when computing Q stats


# --------------------------- SUMTREE ---------------------------
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def update(self, idx: int, p: float):
        leaf_idx = idx + self.capacity - 1
        delta = p - self.tree[leaf_idx]
        self.tree[leaf_idx] = p
        if delta != 0:
            self._propagate(leaf_idx, delta)

    def add(self, p: float):
        idx = self.write
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return idx

    def _retrieve(self, idx: int, s: float):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def sample(self, batch_size: int):
        batch_idx = []
        batch_p = []
        segment = self.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx = self._retrieve(0, s)
            batch_idx.append(idx - self.capacity + 1)
            batch_p.append(self.tree[idx])
        return np.array(batch_idx, dtype=np.int64), np.array(batch_p, dtype=np.float32)

    def total(self) -> float:
        return float(self.tree[0])

    def __len__(self):
        return self.n_entries


# --------------------------- PER BUFFER ---------------------------
class PrioritizedReplay:
    def __init__(self, capacity: int, state_dim: int, alpha: float, eps: float):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def push(self, s, a, r, ns, d):
        idx = self.tree.add(self.max_priority ** self.alpha)
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = ns
        self.dones[idx] = d

    def sample(self, batch_size: int, beta: float):
        idxs, priorities = self.tree.sample(batch_size)
        total = self.tree.total()
        probs = priorities / total
        weights = (len(self.tree) * probs) ** (-beta)
        weights /= weights.max()
        batch = {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_states": self.next_states[idxs],
            "dones": self.dones[idxs].astype(np.float32),
            "idxs": idxs,
            "weights": weights.astype(np.float32),
        }
        return batch

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(idxs, priorities):
            self.tree.update(idx, p)
        self.max_priority = max(self.max_priority, float(priorities.max()))


# --------------------------- NETWORK ---------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(512, 512)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------- AGENT ---------------------------
class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 cfg: AgentConfig, device: torch.device, double: bool = False):
        self.cfg = cfg
        self.device = device
        self.double = double
        self.action_dim = action_dim
        self.tag = "ddqn" if double else "dqn"

        self.online = QNetwork(state_dim, action_dim, cfg.hidden_sizes).to(device)
        self.target = QNetwork(state_dim, action_dim, cfg.hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim = optim.AdamW(self.online.parameters(), lr=cfg.lr, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss(reduction='none')

        self.buffer = PrioritizedReplay(cfg.buffer_size, state_dim, cfg.per_alpha, cfg.per_eps)

        self.tau = cfg.tau_start
        self.tau_decay = (cfg.tau_start - cfg.tau_end) / max(1, cfg.tau_decay_steps)
        self.beta = cfg.per_beta_start
        self.beta_inc = (cfg.per_beta_end - cfg.per_beta_start) / max(1, cfg.per_beta_steps)

        self.env_steps = 0
        self.best_score = -1e9

        # Q-recording histories
        self.q_online_max_history: List[float] = []
        self.q_target_max_history: List[float] = []
        self.q_steps_history: List[int] = []

    def select_action(self, state_np: np.ndarray, explore: bool = True) -> int:
        s = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(s).squeeze(0)
        if not explore:
            return int(q.argmax().item())
        logits = q / max(self.tau, 1e-6)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        return int(np.random.choice(self.action_dim, p=probs))

    def push(self, s, a, r, ns, d):
        self.buffer.push(s, a, r, ns, d)

    def update(self):
        if len(self.buffer.tree) < self.cfg.min_buffer:
            return None

        batch = self.buffer.sample(self.cfg.batch_size, self.beta)
        s = torch.FloatTensor(batch["states"]).to(self.device)
        a = torch.LongTensor(batch["actions"]).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(batch["rewards"]).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(batch["next_states"]).to(self.device)
        d = torch.FloatTensor(batch["dones"]).unsqueeze(1).to(self.device)
        w = torch.FloatTensor(batch["weights"]).unsqueeze(1).to(self.device)
        idxs = batch["idxs"]

        q_sa = self.online(s).gather(1, a)

        with torch.no_grad():
            if self.double:
                na = self.online(ns).argmax(1, keepdim=True)
                next_q = self.target(ns).gather(1, na)
            else:
                next_q = self.target(ns).max(1, keepdim=True).values
            target = r + (1.0 - d) * self.cfg.gamma * next_q

        td_errors = (q_sa - target).detach().cpu().numpy().flatten()
        loss = (w * self.loss_fn(q_sa, target)).mean()

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.max_grad_norm)
        self.optim.step()

        self.buffer.update_priorities(idxs, td_errors)

        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(self.cfg.polyak_tau * op.data + (1.0 - self.cfg.polyak_tau) * tp.data)

        if self.tau > self.cfg.tau_end:
            self.tau = max(self.cfg.tau_end, self.tau - self.tau_decay)
        if self.beta < self.cfg.per_beta_end:
            self.beta = min(self.cfg.per_beta_end, self.beta + self.beta_inc)

        return float(loss.item())

    def maybe_save_best(self, recent_scores: List[float]):
        if (self.env_steps % self.cfg.best_save_interval != 0 or len(recent_scores) == 0):
            return
        window = min(50, len(recent_scores))
        score = float(np.mean(recent_scores[-window:]))
        if score > self.best_score:
            self.best_score = score
            path = os.path.join("models", f"{self.tag}_best.pt")
            torch.save({
                "state_dict": self.online.state_dict(),
                "score": score,
                "env_steps": self.env_steps
            }, path)
            print(f"  [Step {self.env_steps:,}] New best {self.tag.upper()}: {score:+8.2f} → {path}")

    def save(self, path: str):
        torch.save(self.online.state_dict(), path)

    # compute & store Q statistics using a random subset of states from the replay buffer
    def record_q_stats(self):
        n = len(self.buffer.tree)
        if n < max(self.cfg.batch_size, 128):
            return
        batch_size = min(self.cfg.q_record_batch, n)
        idxs = np.random.choice(n, size=batch_size, replace=False)
        states = torch.FloatTensor(self.buffer.states[idxs]).to(self.device)

        with torch.no_grad():
            q_online = self.online(states)  # (batch_size, action_dim)
            q_target = self.target(states)

            # mean of max Q across actions
            online_max = q_online.max(1).values.mean().item()
            target_max = q_target.max(1).values.mean().item()

        self.q_online_max_history.append(online_max)
        self.q_target_max_history.append(target_max)
        self.q_steps_history.append(self.env_steps)


# ----------------------- TRAINING LOOP -----------------------
def train_agent(env_id: str, agent: DQNAgent, episodes: int, max_steps: int, seed: int) -> List[float]:
    env = gym.vector.AsyncVectorEnv([lambda: gym.make(env_id) for _ in range(agent.cfg.num_envs)])
    obs, _ = env.reset(seed=seed)

    episode_returns = np.zeros(agent.cfg.num_envs, dtype=np.float32)
    completed: List[float] = []
    pbar = tqdm(total=episodes, desc=f"Train {agent.tag}")

    while len(completed) < episodes:
        acts = [agent.select_action(o, explore=True) for o in obs]
        next_obs, rew, term, trunc, _ = env.step(acts)
        done = np.logical_or(term, trunc)

        for i in range(agent.cfg.num_envs):
            agent.push(obs[i], acts[i], rew[i], next_obs[i], done[i])
            episode_returns[i] += rew[i]
            if done[i]:
                completed.append(float(episode_returns[i]))
                agent.maybe_save_best(completed)
                episode_returns[i] = 0.0
                pbar.update(1)

        agent.env_steps += agent.cfg.num_envs
        if agent.env_steps % agent.cfg.update_every == 0:
            agent.update()

        # record Q statistics periodically to visualize overestimation during training
        if agent.env_steps % agent.cfg.q_record_interval == 0:
            agent.record_q_stats()

        obs = next_obs

    # ensure final Q-record captured
    agent.record_q_stats()

    pbar.close()
    env.close()
    return completed


# -------------------------- PLOTTING --------------------------
def plot_reward_curve(returns: List[float], path: str, title: str):
    plt.figure(figsize=(8, 5))
    avg = np.convolve(returns, np.ones(100)/100, mode='valid')
    plt.plot(avg, label='100-episode MA', linewidth=2)
    plt.plot(returns, alpha=0.3, color='gray')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_q_values_per_action(q_values: List[np.ndarray], path: str, title: str):
    """q_values: list of (T, 4) arrays from evaluation episodes"""
    n_actions = 4
    action_names = ["Do nothing", "Left engine", "Main engine", "Right engine"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for a in range(n_actions):
        ax = axes[a]
        q_a = [qv[:, a] for qv in q_values]
        max_len = max(len(q) for q in q_a)
        q_padded = np.array([np.pad(q, (0, max_len - len(q)), constant_values=np.nan) for q in q_a])
        q_mean = np.nanmean(q_padded, axis=0)
        q_std = np.nanstd(q_padded, axis=0)

        steps = np.arange(len(q_mean))
        ax.plot(steps, q_mean, label=title, linewidth=2)
        ax.fill_between(steps, q_mean - q_std, q_mean + q_std, alpha=0.2)

        ax.set_title(f"Action {a}: {action_names[a]}")
        ax.set_ylabel("Q-value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time step in episode")
    plt.suptitle("Q-values per Action (averaged over 100 eval episodes)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(path, dpi=200)
    plt.close()


def plot_q_training(agent: DQNAgent, path: str, title: str):
    if len(agent.q_steps_history) == 0:
        return
    plt.figure(figsize=(10, 6))
    steps = np.array(agent.q_steps_history)
    online = np.array(agent.q_online_max_history)
    target = np.array(agent.q_target_max_history)

    plt.plot(steps, online, label=f'{agent.tag.upper()} Online max Q', linewidth=2)
    plt.plot(steps, target, label=f'{agent.tag.upper()} Target max Q', linewidth=2)
    plt.xlabel('Environment steps')
    plt.ylabel('Mean(max_a Q(s,a))')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -------------------------- EVAL + Q-RECORD --------------------------
def evaluate_and_record_q(env_id: str, agent: DQNAgent, episodes: int, max_steps: int, seed: int):
    env = gym.make(env_id)
    all_returns = []
    all_q_values = []  # List of (T, 4) per episode

    for ep in trange(episodes, desc=f"Eval+Record {agent.tag}", leave=False):
        s, _ = env.reset(seed=seed + ep)
        total = 0.0
        q_list = []

        for _ in range(max_steps):
            with torch.no_grad():
                q = agent.online(torch.from_numpy(s).float().unsqueeze(0).to(agent.device))
                q_np = q.squeeze(0).cpu().numpy()
            q_list.append(q_np)

            a = agent.select_action(s, explore=False)
            s, r, done, trunc, _ = env.step(a)
            total += r
            if done or trunc:
                break

        all_returns.append(total)
        all_q_values.append(np.stack(q_list))

    env.close()
    return np.array(all_returns, dtype=np.float32), all_q_values


# --------------------------- MAIN ---------------------------
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    env_id = "LunarLander-v3"
    tmp = gym.make(env_id)
    state_dim = tmp.observation_space.shape[0]
    action_dim = tmp.action_space.n
    tmp.close()

    cfg = AgentConfig()
    seed = 42

    # ---------- DQN ----------
    print("\n=== Training DQN ===")
    dqn = DQNAgent(state_dim, action_dim, cfg, device, double=False)
    dqn_returns = train_agent(env_id, dqn, episodes=20000, max_steps=1000, seed=seed)
    dqn.save("models/dqn_final.pt")
    np.savetxt("data/dqn_returns.csv", dqn_returns, delimiter=",", header="return", comments="")
    plot_reward_curve(dqn_returns, "plots/dqn_training_curve.png", "DQN – Training Return")
    plot_q_training(dqn, "plots/dqn_q_training.png", "DQN – Online vs Target max Q during training")
    # save q histories
    np.save("data/dqn_q_steps.npy", np.array(dqn.q_steps_history))
    np.save("data/dqn_q_online.npy", np.array(dqn.q_online_max_history))
    np.save("data/dqn_q_target.npy", np.array(dqn.q_target_max_history))

    # ---------- DDQN ----------
    print("\n=== Training DDQN ===")
    ddqn = DQNAgent(state_dim, action_dim, cfg, device, double=True)
    ddqn_returns = train_agent(env_id, ddqn, episodes=20000, max_steps=1000, seed=seed)
    ddqn.save("models/ddqn_final.pt")
    np.savetxt("data/ddqn_returns.csv", ddqn_returns, delimiter=",", header="return", comments="")
    plot_reward_curve(ddqn_returns, "plots/ddqn_training_curve.png", "DDQN – Training Return")
    plot_q_training(ddqn, "plots/ddqn_q_training.png", "DDQN – Online vs Target max Q during training")
    np.save("data/ddqn_q_steps.npy", np.array(ddqn.q_steps_history))
    np.save("data/ddqn_q_online.npy", np.array(ddqn.q_online_max_history))
    np.save("data/ddqn_q_target.npy", np.array(ddqn.q_target_max_history))

    # ---------- Evaluation + Q-recording ----------
    print("\n=== Evaluating & Recording Q-values ===")
    dqn_eval_rets, dqn_q_values = evaluate_and_record_q(env_id, dqn, episodes=100, max_steps=1000, seed=999)
    ddqn_eval_rets, ddqn_q_values = evaluate_and_record_q(env_id, ddqn, episodes=100, max_steps=1000, seed=999)

    # Save Q-value plots (SEPARATE for DQN and DDQN)
    plot_q_values_per_action(dqn_q_values, "plots/dqn_q_values_per_action.png", "DQN")
    plot_q_values_per_action(ddqn_q_values, "plots/ddqn_q_values_per_action.png", "DDQN")

    # Final results
    results = {
        "dqn": {"mean_reward": float(np.mean(dqn_eval_rets)), "std_reward": float(np.std(dqn_eval_rets))},
        "ddqn": {"mean_reward": float(np.mean(ddqn_eval_rets)), "std_reward": float(np.std(ddqn_eval_rets))}
    }

    with open("evaluation/evaluation_result.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("FINAL EVALUATION (100 episodes)")
    print("="*70)
    print(f"DQN  → Mean: {results['dqn']['mean_reward']:+8.2f} ± {results['dqn']['std_reward']:5.2f}")
    print(f"DDQN → Mean: {results['ddqn']['mean_reward']:+8.2f} ± {results['ddqn']['std_reward']:5.2f}")
    print("Plots:")
    print("  → plots/dqn_training_curve.png")
    print("  → plots/ddqn_training_curve.png")
    print("  → plots/dqn_q_training.png")
    print("  → plots/ddqn_q_training.png")
    print("  → plots/dqn_q_values_per_action.png")
    print("  → plots/ddqn_q_values_per_action.png")
    print("Best models → models/dqn_best.pt, models/ddqn_best.pt")
    print("="*70)


if __name__ == "__main__":
    main()
