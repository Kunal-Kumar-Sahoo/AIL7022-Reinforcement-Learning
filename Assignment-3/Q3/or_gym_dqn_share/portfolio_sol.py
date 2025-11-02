import numpy as np
import itertools
from collections import deque

import sys
import copy
import time
import random

import gymnasium as gym
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

import os
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(seed)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

######################################################################
# Patched Environments
######################################################################

class PatchedDiscretePortfolioOptEnv(DiscretePortfolioOptEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_length = 1 + self.num_assets + self.num_assets + 1 
        self.observation_space = gym.spaces.Box(-20000, 20000, shape=(self.obs_length,), dtype=np.float32)

    def _STEP(self, action):
        assert self.action_space.contains(action)
        asset_prices = self.asset_prices[:, self.step_count].copy()
        
        for idx, a in enumerate(action):
            a = int(a)
            if a == 0:
                continue
            elif a < 0:
                a = np.abs(a)
                if a > self.holdings[idx]:
                    a = self.holdings[idx]
                self.holdings[idx] -= a
                self.cash += (asset_prices[idx] - self.sell_cost[idx]) * a
            elif a > 0:
                if self.holdings[idx] + a <= self.holding_limit[idx]:
                    cost_per_unit = asset_prices[idx] + self.buy_cost[idx]
                    final_a = a
                    if cost_per_unit > 0:
                        affordable_a = np.floor(self.cash / cost_per_unit)
                        final_a = int(min(a, affordable_a))
                        final_a = max(0, final_a)
                    else:
                        final_a = 0
                    purchase_cost = cost_per_unit * final_a
                    self.holdings[idx] += final_a
                    self.cash -= purchase_cost
        if self.step_count + 1 == self.step_limit: 
            reward = np.dot(asset_prices, self.holdings) + self.cash
        else: 
            reward = 0 
        self.step_count += 1
        if self.step_count >= self.step_limit:
            done = True
        else:
            self._update_state()
            done = False
        return self.state, reward, done, {}
    def reset(self):
        return self._RESET()

class DiscretePortfolioOptEnvTask2(PatchedDiscretePortfolioOptEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _STEP(self, action):
        assert self.action_space.contains(action)
        prices_t = self.asset_prices[:, self.step_count].copy()
        for idx, a in enumerate(action):
            a = int(a)
            if a == 0:
                continue
            elif a < 0:
                a = np.abs(a)
                if a > self.holdings[idx]:
                    a = self.holdings[idx]
                self.holdings[idx] -= a
                self.cash += (prices_t[idx] - self.sell_cost[idx]) * a
            elif a > 0:
                if self.holdings[idx] + a <= self.holding_limit[idx]:
                    cost_per_unit = prices_t[idx] + self.buy_cost[idx]
                    final_a = a
                    if cost_per_unit > 0:
                        affordable_a = np.floor(self.cash / cost_per_unit)
                        final_a = int(min(a, affordable_a))
                        final_a = max(0, final_a)
                    else:
                        final_a = 0
                    purchase_cost = cost_per_unit * final_a
                    self.holdings[idx] += final_a
                    self.cash -= purchase_cost
        self.step_count += 1
        if self.step_count >= self.step_limit:
            done = True
            prices_t_plus_1 = self.asset_prices[:, self.step_limit - 1]
            reward = np.dot(prices_t_plus_1, self.holdings) + self.cash
        else:
            done = False
            self._update_state()
            prices_t_plus_1 = self.asset_prices[:, self.step_count]
            reward = np.dot(prices_t_plus_1, self.holdings) + self.cash
        return self.state, reward, done, {}

######################################################################
# DQN Network
######################################################################

class DQN_Network(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN_Network, self).__init__()
        self.layer1 = nn.Linear(n_observations, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

######################################################################
# Hyperparameters
######################################################################
BATCH_SIZE = 256
GAMMA = 1.0
TEMP_START = 5.0
TEMP_END = 0.1
TEMP_DECAY = 20000
TAU = 0.005
LR = 1e-4
MEMORY_CAPACITY = 10000
NUM_EPISODES = 10000
NUM_EVAL_SEEDS = 100

steps_done = 0

possible_actions = [-2, -1, 0, 1, 2]
ACTION_MAP = list(itertools.product(possible_actions, repeat=5))
N_ACTIONS = len(ACTION_MAP)

######################################################################
# DQN Helper Functions
######################################################################

def select_action(state, policy_net):
    global steps_done
    temp = TEMP_END + (TEMP_START - TEMP_END) * math.exp(-1. * steps_done / TEMP_DECAY)
    with torch.no_grad():
        q_values = policy_net(state).squeeze(0)
        q_values_stable = q_values - q_values.max()
        probs = F.softmax(q_values_stable / temp, dim=0)
        action_idx = torch.multinomial(probs, 1)
        return action_idx.view(1, 1)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return None
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()

def plot_loss(losses, task_name):
    plt.figure()
    plt.plot(losses)
    plt.title(f'Training Loss Curve - {task_name}')
    plt.xlabel('Optimization Step (x100)')
    plt.ylabel('Average Loss (MSELoss)')
    plt.grid(True)
    plot_filename = f'plots/loss_curve_{task_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_filename)
    print(f"Loss curve saved to {plot_filename}")
    plt.close()

def plot_wealth_evaluation(all_wealths_np, task_name):
    mean_wealth = np.mean(all_wealths_np, axis=0)
    std_wealth = np.std(all_wealths_np, axis=0)
    plt.figure()
    time_steps = range(len(mean_wealth))
    plt.plot(time_steps, mean_wealth, label='Mean Portfolio Wealth', color='blue')
    plt.fill_between(time_steps, mean_wealth - std_wealth, mean_wealth + std_wealth, alpha=0.2, color='blue', label='Standard Deviation')
    plt.title(f'Portfolio Wealth Over Time (100 Seeds) - {task_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Wealth')
    plt.legend()
    plt.grid(True)
    plot_filename = f'plots/wealth_plot_{task_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_filename)
    print(f"Wealth plot saved to {plot_filename}")
    plt.close()

######################################################################
# Training and Evaluation Functions
######################################################################

def train_dqn(env, num_episodes, task_name="Task"):
    global steps_done
    steps_done = 0
    n_observations = env.observation_space.shape[0]
    n_actions = N_ACTIONS
    policy_net = DQN_Network(n_observations, n_actions).to(device)
    target_net = DQN_Network(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_CAPACITY)
    losses = []
    total_optimization_steps = 0
    print(f"Starting training for {num_episodes} episodes...")
    for i_episode in tqdm(range(num_episodes), desc=f"Training ({task_name})"):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action_idx_tensor = select_action(state, policy_net)
            action_vec = np.array(ACTION_MAP[action_idx_tensor.item()], dtype=np.int32)
            observation, reward, done, _ = env.step(action_vec)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            memory.push(state, action_idx_tensor, next_state, reward)
            state = next_state
            loss = optimize_model(memory, policy_net, target_net, optimizer)
            if loss is not None:
                total_optimization_steps += 1
                if total_optimization_steps % 100 == 0:
                    losses.append(loss)
            steps_done += 1
            if total_optimization_steps % 1 == 0:
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
            if done:
                break
    print("Training finished.")
    return policy_net, losses

def evaluate_model(EnvClass, policy_net, num_seeds):
    all_wealths = []
    print(f"Starting evaluation on {num_seeds} seeds...")
    for i in tqdm(range(num_seeds), desc="Evaluation Progress"):
        eval_env = EnvClass()
        eval_env.seed(seed + i)
        state = eval_env.reset()
        wealth_over_time = []
        initial_wealth = eval_env.initial_cash
        wealth_over_time.append(initial_wealth)
        for t in range(eval_env.step_limit):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_idx = policy_net(state_tensor).max(1)[1].view(1, 1)
            action_vec = np.array(ACTION_MAP[action_idx.item()], dtype=np.int32)
            state, reward, done, _ = eval_env.step(action_vec)
            current_cash = state[0]
            current_prices = state[1 : 1 + eval_env.num_assets]
            current_holdings = state[1 + eval_env.num_assets : 1 + 2 * eval_env.num_assets]
            current_wealth = current_cash + np.dot(current_prices, current_holdings)
            wealth_over_time.append(current_wealth)
            if done:
                break
        all_wealths.append(wealth_over_time)
    print("Evaluation finished.")
    return np.array(all_wealths)

######################################################################
# Main Execution
######################################################################

if __name__=="__main__":
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)

    start_time=time.time()
    print("\n" + "="*50)
    print("Starting Task 1: Maximize Final Wealth (Sparse Reward)")
    print("="*50)
    env_task1 = PatchedDiscretePortfolioOptEnv()
    policy_net_task1, losses_task1 = train_dqn(env_task1, num_episodes=NUM_EPISODES, task_name="Task 1")
    model_path_t1 = 'models/dqn_portfolio_task1.pth'
    torch.save(policy_net_task1.state_dict(), model_path_t1)
    print(f"Task 1 model saved as '{model_path_t1}'")
    eval_wealth_task1 = evaluate_model(PatchedDiscretePortfolioOptEnv, policy_net_task1, NUM_EVAL_SEEDS)
    
    # --- Build Task 1 Report String ---
    report1_lines = []
    report1_lines.append("\n" + "-"*50)
    report1_lines.append("Task 1: Report")
    report1_lines.append("-"*50)
    report1_lines.append("1. Model and Hyperparameters:")
    report1_lines.append("   - Algorithm: DQN (Vanilla)")
    report1_lines.append(f"   - Network: 3-Layer MLP ({env_task1.observation_space.shape[0]} -> 1024 -> 1024 -> {N_ACTIONS})")
    report1_lines.append(f"   - Optimizer: AdamW (LR={LR})")
    report1_lines.append(f"   - Loss: MSELoss")
    report1_lines.append(f"   - Replay Memory: {MEMORY_CAPACITY}")
    report1_lines.append(f"   - Batch Size: {BATCH_SIZE}")
    report1_lines.append(f"   - Gamma: {GAMMA}")
    report1_lines.append(f"   - Exploration: Boltzmann (Temp: {TEMP_START} -> {TEMP_END})")
    report1_lines.append(f"   - Target Update: Soft (TAU={TAU})")
    report1_lines.append(f"   - Training Episodes: {NUM_EPISODES}")
    report1_lines.append("\n2. Plotting Loss Curve for Task 1...")
    plot_loss(losses_task1, "Task 1")
    report1_lines.append("\n3. Plotting Evaluation Wealth Curve for Task 1...")
    plot_wealth_evaluation(eval_wealth_task1, "Task 1")
    mean_final_wealth_t1 = np.mean(eval_wealth_task1[:, -1])
    std_final_wealth_t1 = np.std(eval_wealth_task1[:, -1])
    report1_lines.append(f"\n4. Mean Total Wealth at T=10: {mean_final_wealth_t1:.2f}")
    ratio_t1 = 0.0
    if std_final_wealth_t1 > 1e-6:
        ratio_t1 = mean_final_wealth_t1 / std_final_wealth_t1
    report1_lines.append(f"5. Ratio (Mean / Std Dev): {ratio_t1:.2f}")
    
    report1_text = "\n".join(report1_lines)
    print(report1_text) # Print to console

    print("\n" + "="*50)
    print("Starting Task 2: Maximize Wealth at All Steps (Dense Reward)")
    print("="*50)
    env_task2 = DiscretePortfolioOptEnvTask2()
    policy_net_task2, losses_task2 = train_dqn(env_task2, num_episodes=NUM_EPISODES, task_name="Task 2")
    model_path_t2 = 'models/dqn_portfolio_task2.pth'
    torch.save(policy_net_task2.state_dict(), model_path_t2)
    print(f"Task 2 model saved as '{model_path_t2}'")
    eval_wealth_task2 = evaluate_model(DiscretePortfolioOptEnvTask2, policy_net_task2, NUM_EVAL_SEEDS)
    
    # --- Build Task 2 Report String ---
    report2_lines = []
    report2_lines.append("\n" + "-"*50)
    report2_lines.append("Task 2: Report")
    report2_lines.append("-"*50)
    report2_lines.append("1. Model and Hyperparameters:")
    report2_lines.append("   - Algorithm: DQN (Vanilla)")
    report2_lines.append(f"   - Network: 3-Layer MLP ({env_task2.observation_space.shape[0]} -> 1024 -> 1024 -> {N_ACTIONS})")
    report2_lines.append(f"   - Optimizer: AdamW (LR={LR})")
    report2_lines.append(f"   - Loss: MSELoss")
    report2_lines.append(f"   - Training Episodes: {NUM_EPISODES}")
    report2_lines.append("\n2. Plotting Loss Curve for Task 2...")
    plot_loss(losses_task2, "Task 2")
    report2_lines.append("\n3. Plotting Evaluation Wealth Curve for Task 2...")
    plot_wealth_evaluation(eval_wealth_task2, "Task 2")
    mean_final_wealth_t2 = np.mean(eval_wealth_task2[:, -1])
    report2_lines.append(f"\n4. Mean Total Wealth at T=10: {mean_final_wealth_t2:.2f}")
    mean_all_wealth_t2 = np.mean(eval_wealth_task2[:, 1:])
    std_all_wealth_t2 = np.std(eval_wealth_task2[:, 1:])
    ratio_t2 = 0.0
    if std_all_wealth_t2 > 1e-6:
        ratio_t2 = mean_all_wealth_t2 / std_all_wealth_t2
    report2_lines.append(f"5. Mean Wealth: {mean_all_wealth_t2:.2f}")
    report2_lines.append(f"   Std Dev: {std_all_wealth_t2:.2f}")
    report2_lines.append(f"   Ratio (Mean / Std Dev): {ratio_t2:.2f}")

    report2_text = "\n".join(report2_lines)
    print(report2_text) # Print to console

    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    
    # --- Write Both Reports to File ---
    report_filename = "evaluation/training_report.txt"
    try:
        with open(report_filename, "w") as f:
            f.write(report1_text)
            f.write("\n\n")
            f.write(report2_text)
            f.write("\n\n" + "="*50)
            f.write(f"\nTotal execution time: {total_time_minutes:.2f} minutes\n")
        
        print("\n" + "="*50)
        print(f"Report successfully saved to '{report_filename}'")
    except IOError as e:
        print("\n" + "="*50)
        print(f"Error saving report to file: {e}")

    print(f"Total execution time: {total_time_minutes:.2f} minutes")