import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import imageio
from collections import deque
import time

from env import TreasureHunt_v2

BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPISODES = 1500
MAX_TIMESTEPS = 500
BUFFER_CAPACITY = 10000
POLYAK_TAU = 1e-3  

TEMP_START = 5.0
TEMP_END = 0.1
TEMP_DECAY = 0.995

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_FRAMES = NUM_EPISODES * MAX_TIMESTEPS
PER_EPS = 1e-6

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_idx = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _get_leaf(self, s):
        idx = 0
        while True:
            left_child_idx = 2 * idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                return idx
            
            if s <= self.tree[left_child_idx]:
                idx = left_child_idx
            else:
                s -= self.tree[left_child_idx]
                idx = right_child_idx

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        tree_idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)
        
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        idx = self._get_leaf(s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class ReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = PER_EPS
        self.p_max = 1.0

    def _get_priority(self, td_error):
        return (np.abs(td_error) + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.p_max, transition)

    def sample(self, batch_size, beta=0.4):
        indices = []
        transitions = []
        priorities = []
        
        segment_size = self.tree.total() / batch_size
        
        for i in range(batch_size):
            s = random.uniform(segment_size * i, segment_size * (i + 1))
            idx, p, data = self.tree.get(s)
            
            indices.append(idx)
            transitions.append(data)
            priorities.append(p)

        sampling_probs = np.array(priorities) / self.tree.total()
        is_weights = (self.tree.n_entries * sampling_probs) ** (-beta)
        is_weights /= is_weights.max()
        
        return transitions, indices, is_weights

    def update_priorities(self, tree_indices, td_errors):
        for idx, error in zip(tree_indices, td_errors):
            priority = self._get_priority(error)
            self.p_max = max(self.p_max, priority)
            self.tree.update(idx, priority)
            
    def __len__(self):
        return self.tree.n_entries

class QNetwork(nn.Module):
    def __init__(self, h=10, w=10, num_actions=4):
        super(QNetwork, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        flattened_size = 576
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class Agent:
    def __init__(self, device):
        self.device = device
        self.num_actions = 4
        
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), 
            lr=LR, 
            weight_decay=WEIGHT_DECAY
        )
        self.memory = ReplayBuffer(BUFFER_CAPACITY, alpha=PER_ALPHA)
        
        self.beta = PER_BETA_START
        self.beta_increment = (1.0 - PER_BETA_START) / PER_BETA_FRAMES
        
        self.temp = TEMP_START # Renamed from tau
        self.tau = POLYAK_TAU   # This is now for Polyak averaging

    def select_action(self, state):
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.policy_net(s)
            # Use self.temp for softmax temperature
            probs = F.softmax(q_values / self.temp, dim=1)
            action = torch.multinomial(probs, num_samples=1)
            return action.item()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        transitions, indices, weights = self.memory.sample(BATCH_SIZE, self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        states = torch.from_numpy(np.array(batch_state)).float().to(self.device)
        actions = torch.tensor(batch_action, dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(batch_reward, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.array(batch_next_state)).float().to(self.device)
        dones = torch.tensor(batch_done, dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            next_q_values_policy = self.policy_net(next_states)
            best_actions = next_q_values_policy.argmax(1).unsqueeze(1)
            
            next_q_values_target = self.target_net(next_states)
            next_q_values = next_q_values_target.gather(1, best_actions)
        
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        current_q_values = self.policy_net(states).gather(1, actions)
        
        is_weights_tensor = torch.tensor(weights, dtype=torch.float).view(-1, 1).to(self.device)
        loss = (is_weights_tensor * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()

        td_errors = (target_q_values - current_q_values).abs().detach().cpu().numpy().flatten()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
        
        # Call soft update *after* optimization step
        self.soft_update_target_net()
        
        return loss.item()

    # Replaced hard update with soft update
    def soft_update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

def plot_rewards(episode_rewards, save_path="plots/dqn_rewards.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward", alpha=0.7)
    
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(99, len(episode_rewards)), moving_avg, label="100-Episode MA", color='red')
        
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Rewards (Softmax, AdamW, Polyak Update)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Reward plot saved to {save_path}")

def check_done(state):
    ship_pos_flat = np.argmax(state[3])
    ship_pos = np.unravel_index(ship_pos_flat, (10, 10))
    return ship_pos == (9, 9)

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    env = TreasureHunt_v2()
    agent = Agent(device)
    
    print(f"Starting training on {device} for {NUM_EPISODES} episodes...")
    
    start_time = time.time()
    
    episode_rewards = []
    
    pbar = trange(NUM_EPISODES, desc="Training Episodes")
    
    for i_episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for t in range(MAX_TIMESTEPS):
            action = agent.select_action(state)
            
            next_state, reward = env.step(action)
            done = check_done(next_state)
            
            agent.memory.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            loss_value = agent.optimize_model()
            
            if loss_value is not None:
                episode_losses.append(loss_value)
            
            if done:
                break
                
        episode_rewards.append(episode_reward)
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        agent.temp = max(TEMP_END, TEMP_DECAY * agent.temp)

        pbar.set_postfix(
            avg_loss=f"{avg_loss:.4f}",
            episode_reward=f"{episode_reward:.2f}",
            temp=f"{agent.temp:.3f}" 
        )

        if (i_episode + 1) % 100 == 0:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            pbar.write(f"Episode {i_episode+1}/{NUM_EPISODES} | Avg Reward (Last 100): {avg_reward_100:.2f} | Temp: {agent.temp:.3f}")

    pbar.close()
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    model_path = "models/dqn_model.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plot_rewards(episode_rewards, "plots/dqn_rewards.png")

    print("Generating policy from Q-network...")
    
    with torch.inference_mode():
        all_states = env.get_all_states()
        all_states_tensor = torch.from_numpy(all_states).float().to(device)
        q_values_all = agent.policy_net(all_states_tensor)
        trained_policy = q_values_all.cpu().numpy()
    
    print("Generating trained agent gif...")
    env.visualize_policy_execution(trained_policy, "gifs/trained_treasurehunt.gif")
    
    print("Generating random agent gif...")
    random_policy = np.random.rand(env.env.num_states, 4)
    env.visualize_policy_execution(random_policy, "gifs/random_treasurehunt.gif")
    
    print("\nAll tasks complete.")
    print("Model saved in 'models/'")
    print("GIFs saved in 'gifs/'")
    print("Plot saved in 'plots/'")

if __name__ == "__main__":
    main()