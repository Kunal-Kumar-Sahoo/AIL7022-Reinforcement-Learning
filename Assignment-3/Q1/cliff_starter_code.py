import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque, namedtuple
import copy
import csv
import json
from tqdm import trange
import matplotlib.pyplot as plt
import imageio
from math import exp
from cliff import MultiGoalCliffWalkingEnv

# Hyperparameters
NUM_SEEDS = 10
MAX_EPISODE_STEPS = 100
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
REPLAY_CAPACITY = 100000
START_TRAIN_SIZE = 1000
TARGET_UPDATE_FREQ = 1000
ALPHA_PER = 0.6
BETA_PER_START = 0.4
BETA_PER_FRAMES = 100000
EPS_PRIORITY = 1e-6
TEMPERATURE_START = 1.0
TEMPERATURE_END = 0.1
TEMPERATURE_DECAY_FRAMES = 200000
NUM_EPISODES_PER_SEED = 10000  # extended training
DEVICE_PRIORITIES = ['cuda', 'mps', 'cpu']

# Select device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except Exception:
        pass
    return torch.device('cpu')

DEVICE = get_device()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        prios = self.priorities if len(self.buffer) == self.capacity else self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(probs), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(probs)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        batch = Transition(*zip(*samples))
        return batch, indices, torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    def update_priorities(self, indices, priorities):
        for idx, pr in zip(indices, priorities):
            self.priorities[idx] = pr + EPS_PRIORITY

    def __len__(self):
        return len(self.buffer)

def state_to_tensor(state, env):
    grid_size = env.height * env.width
    checkpoint_status = state // grid_size
    position_index = state % grid_size
    y = position_index // env.width
    x = position_index % env.width
    checkpoints_binary = [(checkpoint_status >> i) & 1 for i in range(2)]
    features = [y / env.height, x / env.width] + checkpoints_binary
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# --- Larger Models ---
class LinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearDQN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 8)
        self.head = nn.Linear(output_dim * 8, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return self.head(x)

class NonLinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def boltzmann_action(q_values, temperature):
    scaled = q_values.detach().cpu().numpy().astype(np.float64) / (temperature + 1e-8)
    max_q = np.max(scaled, axis=1, keepdims=True)
    exp_q = np.exp(scaled - max_q)
    probs = exp_q / exp_q.sum(axis=1, keepdims=True)
    action = [np.random.choice(len(p), p=p) for p in probs]
    return int(action[0]), probs[0]

class DQNTrainer:
    def __init__(self, env, network, optimizer, seed=0, replay_capacity=REPLAY_CAPACITY):
        self.env = env
        self.net = network.to(DEVICE)
        self.target_net = copy.deepcopy(self.net).to(DEVICE)
        self.optimizer = optimizer
        self.seed = seed
        self.steps_done = 0
        self.replay = PrioritizedReplayBuffer(replay_capacity, alpha=ALPHA_PER)
        self.beta = BETA_PER_START
        self.loss_fn = nn.MSELoss(reduction='none')  
        set_seed(seed)

    def select_action(self, state_tensor, temperature):
        self.net.eval()
        with torch.no_grad():
            q_vals = self.net(state_tensor)
        action, probs = boltzmann_action(q_vals, temperature)
        return action, q_vals

    def optimize_model(self, batch_size):
        if len(self.replay) < batch_size:
            return 0.0
        self.beta = min(1.0, BETA_PER_START + (self.steps_done / BETA_PER_FRAMES) * (1.0 - BETA_PER_START))
        batch, indices, weights = self.replay.sample(batch_size, beta=self.beta)
        states = torch.cat(batch.state).to(DEVICE)
        actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_states = torch.cat(batch.next_state).to(DEVICE)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        q_values = self.net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            expected_q = rewards + (1.0 - dones) * GAMMA * next_q
        losses = self.loss_fn(q_values, expected_q)
        weighted_losses = (weights.unsqueeze(1) * losses).mean()
        self.optimizer.zero_grad()
        weighted_losses.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()
        td_errors = losses.detach().cpu().numpy().squeeze().tolist()
        if isinstance(td_errors, float): td_errors = [td_errors]
        new_prios = np.abs(td_errors) + EPS_PRIORITY
        self.replay.update_priorities(indices, new_prios)
        return weighted_losses.item()

    def train(self, num_episodes, save_csv_path=None):
        episode_rewards, losses = [], []
        best_mean = -float('inf')
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        if save_csv_path and not os.path.exists(save_csv_path):
            with open(save_csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(['seed','episode','loss','mean_reward_last20','std_reward_last20'])
        pbar = trange(num_episodes, desc=f"Seed {self.seed} training", leave=True)
        for ep in pbar:
            state, _ = self.env.reset()
            state_t = state_to_tensor(state, self.env)
            ep_reward, ep_losses = 0.0, []
            for t in range(MAX_EPISODE_STEPS):
                temp = TEMPERATURE_END + (TEMPERATURE_START - TEMPERATURE_END) * np.exp(-1.0 * self.steps_done / TEMPERATURE_DECAY_FRAMES)
                action, _ = self.select_action(state_t, temp)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state_t = state_to_tensor(next_state, self.env)
                self.replay.push(state_t, action, reward, next_state_t, float(done))
                self.steps_done += 1
                state_t = next_state_t
                ep_reward += reward
                if len(self.replay) >= START_TRAIN_SIZE:
                    ep_losses.append(self.optimize_model(BATCH_SIZE))
                if self.steps_done % TARGET_UPDATE_FREQ == 0:
                    self.target_net.load_state_dict(self.net.state_dict())
                if done: break
            episode_rewards.append(ep_reward)
            losses.append(np.mean(ep_losses) if ep_losses else 0.0)
            mean20 = float(np.mean(episode_rewards[-20:]))
            std20 = float(np.std(episode_rewards[-20:]))
            pbar.set_postfix({'mean20': mean20, 'loss': losses[-1]})
            if save_csv_path:
                with open(save_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([self.seed, ep+1, losses[-1], mean20, std20])
            if mean20 > best_mean:
                best_mean = mean20
        return {'episode_rewards': episode_rewards, 'losses': losses, 'best_mean': best_mean}

    def evaluate(self, num_episodes=100):
        self.net.eval()
        rewards = []
        with torch.no_grad():
            for _ in range(num_episodes):
                state, _ = self.env.reset()
                s_t = state_to_tensor(state, self.env)
                ep_r = 0.0
                for _ in range(MAX_EPISODE_STEPS):
                    qv = self.net(s_t)
                    action = int(qv.argmax(dim=1).item())
                    next_s, r, done, _, _ = self.env.step(action)
                    s_t = state_to_tensor(next_s, self.env)
                    ep_r += r
                    if done: break
                rewards.append(ep_r)
        return float(np.mean(rewards)), float(np.std(rewards))

def ensure_dirs():
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def plot_mean_std_across_seeds(seed_rewards, filename, title):
    n = max(len(r) for r in seed_rewards)
    for r in seed_rewards: 
        if len(r) < n: r += [r[-1]]*(n-len(r))
    rewards_arr = np.array(seed_rewards)
    mean = rewards_arr.mean(axis=0)
    std = rewards_arr.std(axis=0)
    plt.figure()
    plt.plot(mean, label='Mean Reward')
    plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_gif(env, model, save_path):
    frames = []
    state, _ = env.reset()
    s_t = state_to_tensor(state, env)
    for _ in range(MAX_EPISODE_STEPS):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        with torch.no_grad():
            qv = model(s_t)
            action = int(qv.argmax(dim=1).item())
        next_s, _, done, _, _ = env.step(action)
        s_t = state_to_tensor(next_s, env)
        if done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break
    imageio.mimsave(save_path, frames, fps=4)

def run_training_for_network(network_cls, model_name_prefix):
    ensure_dirs()
    env = MultiGoalCliffWalkingEnv(train=True)
    output_dim = env.action_space.n
    input_dim = 4
    seed_rewards = []
    best_overall = {'mean': -float('inf'), 'model_state': None, 'seed': None}
    for seed in range(NUM_SEEDS):
        set_seed(seed)
        env_train = MultiGoalCliffWalkingEnv(train=True)
        net = network_cls(input_dim, output_dim).to(DEVICE)
        optimizer = optim.Adam(net.parameters(), lr=LR)
        trainer = DQNTrainer(env_train, net, optimizer, seed=seed)
        csv_path = f"logs/{model_name_prefix}_seed{seed}_training_log.csv"
        res = trainer.train(NUM_EPISODES_PER_SEED, save_csv_path=csv_path)
        seed_rewards.append(res['episode_rewards'])
        if res['best_mean'] > best_overall['mean']:
            best_overall.update({'mean':res['best_mean'], 'model_state':trainer.net.state_dict(), 'seed':seed})
    torch.save(best_overall['model_state'], f"models/{model_name_prefix}.pt")
    plot_mean_std_across_seeds(seed_rewards, f"plots/cliff_average_rewards_{model_name_prefix.replace('best_','')}.png",
                               f"Training Performance ({model_name_prefix})")
    return f"models/{model_name_prefix}.pt"

def main():
    ensure_dirs()
    # print("Training Linear Agent...")
    # linear_model_path = run_training_for_network(LinearDQN, "best_linear")
    # print("Training Non-Linear Agent...")
    # nonlinear_model_path = run_training_for_network(NonLinearDQN, "best_nonlinear")

    eval_env = MultiGoalCliffWalkingEnv(train=False, render_mode='rgb_array')
    results = {}
    for key, model_file in [('linear','models/best_linear.pt'),('nonlinear','models/best_nonlinear.pt')]:
        model = LinearDQN(4, eval_env.action_space.n) if key=='linear' else NonLinearDQN(4, eval_env.action_space.n)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model.to(DEVICE)
        trainer = DQNTrainer(eval_env, model, optimizer=optim.Adam(model.parameters(), lr=LR))
        mean_r,std_r = trainer.evaluate(num_episodes=100)
        results[key]={'mean':mean_r,'std':std_r}
        gif_path=f"plots/cliff_{key}.gif"
        generate_gif(eval_env, model, gif_path)
    eval_env.close()

    with open('evaluation/cliff_evaluation_results.json','w') as f:
        json.dump(results,f,indent=4)

if __name__ == '__main__':
    main()
