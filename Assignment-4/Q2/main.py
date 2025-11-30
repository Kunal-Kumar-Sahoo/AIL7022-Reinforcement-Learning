import os
import pickle
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import trange, tqdm


# --------------------------- CONFIG ---------------------------
@dataclass
class Config:
    env_id: str = "InvertedPendulum-v5"
    num_envs: int = 64
    seed: int = 42
    gamma: float = 0.99
    lr: float = 3e-4
    vf_lr: float = 1e-3
    max_episodes: int = 40000
    target_avg_reward_min: float = 400.0
    target_avg_reward_max: float = 500.0
    hidden_sizes: tuple = (256, 256)
    batch_vf_updates: int = 10
    collect_trajectories_count: int = 500
    grad_repeat: int = 10
    sample_sizes: List[int] = field(default_factory=lambda: [20, 30, 40, 50, 60, 70, 80, 90, 100])
    stabilize_episodes: int = 300

    # device: torch.device = torch.device(
    #     "cuda" if torch.cuda.is_available() else
    #     "mps" if torch.backends.mps.is_available() else "cpu"
    # )
    device = 'cpu'

    save_dir: str = "reinforce_variance"
    q_plot_path: str = "plots/gradient estimate variance.png"


# --------------------------- UTIL ---------------------------
def make_vec_env(env_id: str, n: int, seed: int):
    def _make(_):
        def _thunk():
            env = gym.make(env_id)
            return env
        return _thunk

    return gym.vector.AsyncVectorEnv([_make(i) for i in range(n)])


def flat_grad(model: nn.Module) -> torch.Tensor:
    grads = []
    for p in model.parameters():
        g = p.grad
        if g is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(g.detach().view(-1))
    return torch.cat(grads).cpu()


# --------------------------- POLICY & VALUE NETS ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes=(256, 256), activation=nn.Tanh):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.mu_net = MLP(obs_dim, act_dim, hidden_sizes, activation=nn.Tanh)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def get_dist(self, obs: torch.Tensor):
        mu, std = self.forward(obs)
        return torch.distributions.Normal(mu, std)

    def act_and_logp(self, obs: np.ndarray, device: torch.device):
        obs_t = torch.FloatTensor(obs).to(device)
        dist = self.get_dist(obs_t)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        return action.detach().cpu().numpy(), logp.detach().cpu().numpy()

    def log_prob_from_data(self, obs: torch.Tensor, act: torch.Tensor):
        dist = self.get_dist(obs)
        return dist.log_prob(act).sum(axis=-1)


# --------------------------- TRAINING HELPERS ---------------------------
def compute_returns(rewards: List[float], gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    R = 0.0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns


# --------------------------- TRAINING (REINFORCE) ---------------------------
def train_reinforce_variant(cfg: Config, baseline_type: str, save_path: str):
    """
    baseline_type in {"none", "avg_reward", "reward_to_go", "value_fn"}
    """
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = cfg.device

    env = make_vec_env(cfg.env_id, cfg.num_envs, cfg.seed)
    tmp = gym.make(cfg.env_id)
    obs_dim = tmp.observation_space.shape[0]
    act_dim = tmp.action_space.shape[0]
    tmp.close()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    policy = GaussianPolicy(obs_dim, act_dim, cfg.hidden_sizes).to(device)
    vf = MLP(obs_dim, 1, cfg.hidden_sizes).to(device) if baseline_type == "value_fn" else None

    opt_policy = optim.Adam(policy.parameters(), lr=cfg.lr)
    opt_vf = optim.Adam(vf.parameters(), lr=cfg.vf_lr) if vf is not None else None

    recent_returns: List[float] = []
    episode_count = 0

    env_obs, _ = env.reset(seed=cfg.seed)
    env_episode_rewards = np.zeros(cfg.num_envs, dtype=np.float32)
    env_episode_steps = np.zeros(cfg.num_envs, dtype=np.int32)

    running_avg_reward = 0.0
    running_count = 0

    first_reached_at = None

    pbar = tqdm(total=cfg.max_episodes, desc=f"Train {baseline_type}", leave=True)
    while episode_count < cfg.max_episodes:
        trajectories = [[] for _ in range(cfg.num_envs)]
        done_mask = np.zeros(cfg.num_envs, dtype=bool)
        max_steps_per_ep = 2000
        steps = 0
        while not done_mask.all() and steps < max_steps_per_ep:
            actions, logps = policy.act_and_logp(env_obs, device)
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            dones = np.logical_or(terms, truncs)

            for i in range(cfg.num_envs):
                if done_mask[i]:
                    continue
                trajectories[i].append({
                    "s": env_obs[i].copy(),
                    "a": actions[i].copy(),
                    "r": float(rewards[i]),
                    "logp": float(logps[i])
                })
                env_episode_rewards[i] += rewards[i]
                env_episode_steps[i] += 1

                if dones[i]:
                    ep_return = float(env_episode_rewards[i])
                    recent_returns.append(ep_return)
                    if len(recent_returns) > 2000:
                        recent_returns.pop(0)
                    episode_count += 1
                    pbar.update(1)
                    if running_count > 0:
                        running_avg_reward = (running_avg_reward * running_count + ep_return) / (running_count + 1)
                    else:
                        running_avg_reward = ep_return
                    running_count += 1
                    env_episode_rewards[i] = 0.0
                    env_episode_steps[i] = 0
                    done_mask[i] = True

            env_obs = next_obs
            steps += 1

        episodes_data = []
        for traj in trajectories:
            if len(traj) == 0:
                continue
            states = np.stack([t["s"] for t in traj], axis=0).astype(np.float32)
            actions = np.stack([t["a"] for t in traj], axis=0).astype(np.float32)
            rewards = np.array([t["r"] for t in traj], dtype=np.float32)
            logps = np.array([t["logp"] for t in traj], dtype=np.float32)
            episodes_data.append({"states": states, "actions": actions, "rewards": rewards, "logps": logps})

        if len(episodes_data) == 0:
            continue

        all_states = []
        all_actions = []
        all_advantages = []

        for ep in episodes_data:
            returns = compute_returns(ep["rewards"].tolist(), cfg.gamma)
            if baseline_type == "none":
                adv = returns
            elif baseline_type == "avg_reward":
                adv = returns - running_avg_reward
            elif baseline_type == "reward_to_go":
                adv = returns
            elif baseline_type == "value_fn":
                s_t = torch.FloatTensor(ep["states"]).to(device)
                with torch.no_grad():
                    v_pred = vf(s_t).squeeze(-1).cpu().numpy()
                adv = returns - v_pred
            else:
                raise ValueError("Unknown baseline type")
            all_states.append(ep["states"])
            all_actions.append(ep["actions"])
            all_advantages.append(adv)

        states_cat = torch.FloatTensor(np.concatenate(all_states, axis=0)).to(device)
        actions_cat = torch.FloatTensor(np.concatenate(all_actions, axis=0)).to(device)
        advantages_cat = torch.FloatTensor(np.concatenate(all_advantages, axis=0)).to(device)

        # advantage normalization for training stability
        adv_mean = advantages_cat.mean()
        adv_std = advantages_cat.std() + 1e-8
        advantages_norm = (advantages_cat - adv_mean) / adv_std

        logp = policy.log_prob_from_data(states_cat, actions_cat)
        loss_policy = -(logp * advantages_norm).mean()

        opt_policy.zero_grad()
        loss_policy.backward()
        opt_policy.step()

        if vf is not None:
            returns_cat = torch.FloatTensor(
                np.concatenate([compute_returns(ep["rewards"].tolist(), cfg.gamma) for ep in episodes_data], axis=0)
            ).to(device)
            for _ in range(cfg.batch_vf_updates):
                opt_vf.zero_grad()
                v_pred = vf(states_cat).squeeze(-1)
                loss_v = nn.MSELoss()(v_pred, returns_cat)
                loss_v.backward()
                opt_vf.step()

        if len(recent_returns) >= 100:
            avg100 = float(np.mean(recent_returns[-100:]))
        else:
            avg100 = float(np.mean(recent_returns))

        pbar.set_postfix({"episodes": episode_count, "avg100": f"{avg100:.2f}"})

        if cfg.target_avg_reward_min <= avg100 <= cfg.target_avg_reward_max:
            if first_reached_at is None:
                first_reached_at = episode_count
            if episode_count >= first_reached_at + cfg.stabilize_episodes:
                torch.save(policy.state_dict(), save_path)
                if vf is not None:
                    torch.save(vf.state_dict(), save_path.replace(".pt", "_vf.pt"))
                print(f"\n[{baseline_type}] Reached and stabilized avg100 {avg100:.2f} at episode {episode_count}. Saved -> {save_path}")
                break

    pbar.close()
    env.close()
    return policy, vf


# --------------------------- TRAJECTORY COLLECTION ---------------------------
def collect_trajectories_for_policy(cfg: Config, policy: GaussianPolicy, count: int, save_file: str):
    device = cfg.device
    env = gym.make(cfg.env_id)
    trajectories = []
    while len(trajectories) < count:
        s, _ = env.reset(seed=np.random.randint(1_000_000))
        states = []
        actions = []
        rewards = []
        done = False
        steps = 0
        max_steps = 2000
        while not done and steps < max_steps:
            a, _ = policy.act_and_logp(np.expand_dims(s, axis=0), device)
            a = a[0]
            ns, r, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            states.append(s.copy())
            actions.append(a.copy())
            rewards.append(float(r))
            s = ns
            steps += 1
        trajectories.append({
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
        })
        if len(trajectories) % 50 == 0:
            print(f"  Collected {len(trajectories)}/{count} trajectories")
    env.close()
    with open(save_file, "wb") as f:
        pickle.dump(trajectories, f)
    return trajectories


# --------------------------- GRADIENT ESTIMATION ---------------------------
def gradient_estimate_from_trajectories(cfg: Config, policy: GaussianPolicy, trajectories: List[Dict[str, Any]],
                                        baseline_type: str):
    device = cfg.device
    policy.to(device)
    policy.train()
    opt_dummy = optim.SGD(policy.parameters(), lr=1e-8)
    opt_dummy.zero_grad()

    states_list = []
    actions_list = []
    advantages_list = []

    for ep in trajectories:
        rewards = ep["rewards"]
        returns = compute_returns(rewards.tolist(), cfg.gamma)
        if baseline_type == "none":
            adv = returns
        elif baseline_type == "avg_reward":
            avg_r = returns.mean()
            adv = returns - avg_r
        elif baseline_type == "reward_to_go":
            adv = returns
        elif baseline_type == "value_fn":
            if "v_preds" in ep:
                v_preds = ep["v_preds"]
                adv = returns - np.array(v_preds, dtype=np.float32)
            else:
                adv = returns
        else:
            raise ValueError("Unknown baseline type")
        states_list.append(ep["states"])
        actions_list.append(ep["actions"])
        advantages_list.append(adv)

    states_all = torch.FloatTensor(np.concatenate(states_list, axis=0)).to(device)
    actions_all = torch.FloatTensor(np.concatenate(actions_list, axis=0)).to(device)
    advs_all = torch.FloatTensor(np.concatenate(advantages_list, axis=0)).to(device)

    # no advantage normalization here to preserve baseline effect
    logp = policy.log_prob_from_data(states_all, actions_all)
    loss = -(logp * advs_all).mean()

    opt_dummy.zero_grad()
    loss.backward()
    grad_vec = flat_grad(policy).numpy()
    opt_dummy.zero_grad()
    return grad_vec


# --------------------------- MAIN: RUN ALL EXPERIMENTS ---------------------------
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    baselines = {
        "none": "reinforce_none.pt",
        "avg_reward": "reinforce_avg_reward.pt",
        "reward_to_go": "reinforce_reward_to_go.pt",
        "value_fn": "reinforce_value_fn.pt",
    }

    trained_policies = {}
    trained_valuefs = {}

    for bname, fname in baselines.items():
        model_path = os.path.join(cfg.save_dir, fname)
        vf_path = model_path.replace(".pt", "_vf.pt")
        print(f"\n=== Training baseline: {bname} ===")
        policy, vf = train_reinforce_variant(cfg, bname, model_path)
        trained_policies[bname] = policy
        if vf is not None:
            trained_valuefs[bname] = vf
            torch.save(vf.state_dict(), vf_path)

    all_trajectories = {}
    for bname, policy in trained_policies.items():
        traj_file = os.path.join(cfg.save_dir, f"traj_{bname}.pkl")
        print(f"\nCollecting trajectories for baseline {bname} -> {traj_file}")
        trajs = collect_trajectories_for_policy(cfg, policy, cfg.collect_trajectories_count, traj_file)
        if bname == "value_fn" and bname in trained_valuefs:
            vf = trained_valuefs[bname]
            vf.to(cfg.device)
            for ep in trajs:
                states = torch.FloatTensor(ep["states"]).to(cfg.device)
                with torch.no_grad():
                    v_pred = vf(states).squeeze(-1).cpu().numpy()
                ep["v_preds"] = v_pred
        all_trajectories[bname] = trajs

    results_stats = {b: {} for b in baselines.keys()}
    for bname, trajs in all_trajectories.items():
        print(f"\nEstimating gradients for baseline {bname}")
        policy = trained_policies[bname]
        policy.to(cfg.device)
        rng = np.random.RandomState(cfg.seed + (abs(hash(bname)) % 1000))
        for sample_size in cfg.sample_sizes:
            grad_norms = []
            grad_means = []
            for rep in range(cfg.grad_repeat):
                idxs = rng.choice(len(trajs), size=sample_size, replace=False)
                sampled = [trajs[i] for i in idxs]
                grad_vec = gradient_estimate_from_trajectories(cfg, policy, sampled, baseline_type=bname)
                l2 = float(np.linalg.norm(grad_vec))
                mean_abs = float(np.mean(np.abs(grad_vec)))
                grad_norms.append(l2)
                grad_means.append(mean_abs)
            results_stats[bname][sample_size] = {
                "l2_mean": float(np.mean(grad_norms)),
                "l2_std": float(np.std(grad_norms)),
                "meanabs_mean": float(np.mean(grad_means)),
                "meanabs_std": float(np.std(grad_means)),
                "raw_norms": grad_norms
            }
            print(
                f"  baseline={bname} sample={sample_size} "
                f"L2 mean={results_stats[bname][sample_size]['l2_mean']:.4f} "
                f"std={results_stats[bname][sample_size]['l2_std']:.4f}"
            )

    # plotting with shared y limits
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    order = ["none", "avg_reward", "reward_to_go", "value_fn"]

    # compute global y limits across all baselines
    global_min = float("inf")
    global_max = float("-inf")
    for bname in order:
        xs = sorted(results_stats[bname].keys())
        means = np.array([results_stats[bname][x]["l2_mean"] for x in xs])
        stds = np.array([results_stats[bname][x]["l2_std"] for x in xs])
        low = means - stds
        high = means + stds
        if low.min() < global_min:
            global_min = float(low.min())
        if high.max() > global_max:
            global_max = float(high.max())

    for ax, bname in zip(axes, order):
        xs = sorted(results_stats[bname].keys())
        means = np.array([results_stats[bname][x]["l2_mean"] for x in xs])
        stds = np.array([results_stats[bname][x]["l2_std"] for x in xs])

        ax.plot(xs, means, label=f"{bname} mean L2", linewidth=2)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.3)
        ax.set_title(f"Baseline: {bname}")
        ax.set_xlabel("Sample size (trajectories)")
        ax.set_ylabel("Gradient L2 magnitude")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(global_min, global_max)

    plt.suptitle("Gradient estimate magnitude vs sample size (±1 std shaded)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(cfg.q_plot_path, dpi=200)
    plt.close()
    print(f"\nSaved plot -> {cfg.q_plot_path}")

    stats_file = os.path.join(cfg.save_dir, "gradient_variance_stats.pkl")
    with open(stats_file, "wb") as f:
        pickle.dump(results_stats, f)
    print(f"Saved stats -> {stats_file}")

    trajs_file = os.path.join(cfg.save_dir, "all_trajectories.pkl")
    with open(trajs_file, "wb") as f:
        pickle.dump(all_trajectories, f)
    print(f"Saved trajectories -> {trajs_file}")


if __name__ == "__main__":
    main()