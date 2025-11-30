import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import random
from collections import deque
import time
import json

# ====================== CONFIGURATION ======================
ENV_NAME = "LunarLander-v3"
SEED = 42
MAX_EPISODES = 1500
MAX_STEPS = 1000
GAMMA = 0.99

# Directories
BASE_DIR = "Q4"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
GIF_DIR = os.path.join(BASE_DIR, "gifs")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

for d in [CKPT_DIR, PLOT_DIR, GIF_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Hyperparameters
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
LR_DQN = 0.0003
HIDDEN_SIZE = 128

# Entropy
ENTROPY_START = 0.05
ENTROPY_END = 0.001
ENTROPY_DECAY = 0.995

# DQN settings
BUFFER_SIZE = 50000
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

MAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                           "mps" if torch.backends.mps.is_available() else 
                           "cpu")
AC_DEVICE = torch.device("cpu") 
print(f"DQN Device: {MAIN_DEVICE} | AC Device: {AC_DEVICE}")

# ====================== UTILITIES ======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def moving_average(data, window=10):
    return np.convolve(data, np.ones(window) / window, mode="valid")

# ====================== NETWORKS ======================

class BaseNet(nn.Module):
    """
    Base network with LayerNorm for stability.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.base = BaseNet(state_dim, HIDDEN_SIZE)
        self.head = nn.Linear(HIDDEN_SIZE, action_dim)
    
    def forward(self, x):
        x = self.base(x)
        return F.softmax(self.head(x), dim=-1)
    

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.base = BaseNet(state_dim, HIDDEN_SIZE)
        self.head = nn.Linear(HIDDEN_SIZE, 1)
    
    def forward(self, x):
        x = self.base(x)
        return self.head(x)
    

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.base = BaseNet(state_dim, HIDDEN_SIZE)
        self.head = nn.Linear(HIDDEN_SIZE, action_dim)

    def forward(self, x):
        x = self.base(x)
        return self.head(x)


# ====================== AGENTS ======================
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = ActorNetwork(state_dim, action_dim).to(AC_DEVICE)
        self.critic = CriticNetwork(state_dim).to(AC_DEVICE)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.ent_coef = ENTROPY_START

    def select_action(self, state):
        state_t = torch.FloatTensor(state).to(AC_DEVICE).unsqueeze(0)
        probs = self.actor(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state).to(AC_DEVICE).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).to(AC_DEVICE).unsqueeze(0)
        action_t = torch.tensor([action], device=AC_DEVICE)
        
        scaled_reward = reward / 100.0
        reward_t = torch.tensor([scaled_reward], device=AC_DEVICE, dtype=torch.float32)
        done_t = torch.tensor([float(done)], device=AC_DEVICE, dtype=torch.float32)

        curr_value = self.critic(state_t)
        
        with torch.no_grad():
            next_value = self.critic(next_state_t)
            
        td_target = reward_t + GAMMA * next_value * (1 - done_t)
        
        advantage = td_target - curr_value
        
        critic_loss = F.mse_loss(curr_value, td_target)
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) 
        self.critic_optim.step()
        
        probs = self.actor(state_t)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()
        
        actor_loss = -(log_prob * advantage.detach()) - (self.ent_coef * entropy)
        
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        return actor_loss.item(), critic_loss.item()
    
    def decay_entropy(self):
        self.ent_coef = max(ENTROPY_END, self.ent_coef * ENTROPY_DECAY)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=AC_DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = DQNNetwork(state_dim, action_dim).to(MAIN_DEVICE)
        self.target_net = DQNNetwork(state_dim, action_dim).to(MAIN_DEVICE)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR_DQN)
        self.memory = deque(maxlen=BUFFER_SIZE)
        
        self.epsilon = EPS_START
        self.action_dim = action_dim

    def select_action(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(MAIN_DEVICE)
            return self.q_net(state_t).argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(MAIN_DEVICE)
        actions = torch.LongTensor(actions).unsqueeze(1).to(MAIN_DEVICE)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(MAIN_DEVICE)
        next_states = torch.FloatTensor(np.array(next_states)).to(MAIN_DEVICE)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(MAIN_DEVICE)

        curr_q = self.q_net(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + GAMMA * max_next_q * (1 - dones)
        
        loss = F.mse_loss(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        tau = 0.005
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
    
    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=MAIN_DEVICE))


# ====================== TRAINING ======================
def train_ac(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCriticAgent(state_dim, action_dim)
    
    rewards_history = []
    actor_loss_history, critic_loss_history = [], []
    
    start_time = time.time()
    print("Starting Actor-Critic Training ...")
    
    best_reward = -float('inf')

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset(seed=SEED+episode)
        ep_reward = 0
        ep_a_loss, ep_c_loss = [], []
        
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            al, cl = agent.update(state, action, reward, next_state, done)
            
            ep_a_loss.append(al)
            ep_c_loss.append(cl)
            ep_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_entropy()
        
        # Logging
        rewards_history.append(ep_reward)
        avg_al = np.mean(ep_a_loss) if ep_a_loss else 0
        avg_cl = np.mean(ep_c_loss) if ep_c_loss else 0
        actor_loss_history.append(avg_al)
        critic_loss_history.append(avg_cl)
        
        avg_reward_10 = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else ep_reward
        
        if avg_reward_10 > best_reward:
            best_reward = avg_reward_10
            agent.save(os.path.join(CKPT_DIR, "best_actor_critic.pt"))
        
        if episode % 20 == 0:
            print(f"AC Ep {episode:4d} | R: {ep_reward:6.1f} | Avg10: {avg_reward_10:6.1f} | AL: {avg_al:.4f} | CL: {avg_cl:.4f} | Ent: {agent.ent_coef:.3f}")

        if avg_reward_10 > 220:
            print(f"Actor-Critic Solved in {episode} episodes!")
            break
            
    return agent, rewards_history, actor_loss_history, critic_loss_history, time.time() - start_time

def train_dqn(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    rewards_history = []
    loss_history = []
    
    start_time = time.time()
    print(f"\nStarting DQN Training ...")
    
    best_reward = -float('inf')

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset(seed=SEED+episode+10000)
        ep_reward = 0
        ep_loss = []
        
        for _ in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store(state, action, reward, next_state, done)
            loss = agent.update()
            
            ep_loss.append(loss)
            ep_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_epsilon()
        
        rewards_history.append(ep_reward)
        avg_loss = np.mean(ep_loss) if ep_loss else 0
        loss_history.append(avg_loss)
        
        avg_reward_10 = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else ep_reward

        if avg_reward_10 > best_reward:
            best_reward = avg_reward_10
            # Save DQN best model
            agent.save(os.path.join(CKPT_DIR, "best_dqn.pt"))
            
        if episode % 20 == 0:
            print(f"DQN Ep {episode:4d} | R: {ep_reward:6.1f} | Avg10: {avg_reward_10:6.1f} | Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.3f}")

        if avg_reward_10 > 220:
            print(f"DQN Solved in {episode} episodes!")
            break
            
    return agent, rewards_history, loss_history, time.time() - start_time

def save_gifs_and_return_rewards(agent, env_name, algo_name, episodes=5):
    env = gym.make(env_name, render_mode="rgb_array")
    rewards = []
    
    print(f"\nGenerating GIFs for {algo_name}...")
    for i in range(episodes):
        state, _ = env.reset(seed=i)
        frames = []
        ep_reward = 0
        
        while True:
            frames.append(env.render())
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, evaluate=True)
            else:
                action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        
        rewards.append(ep_reward)
        imageio.mimsave(os.path.join(GIF_DIR, f"{algo_name}_ep{i+1}.gif"), frames, fps=30)
        
    env.close()
    return rewards

def generate_plots(ac_data, dqn_data):
    ac_rew, ac_a_loss, ac_c_loss = ac_data
    dqn_rew, dqn_loss = dqn_data
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(ac_rew, color='blue', alpha=0.2, linewidth=1)
    plt.plot(moving_average(ac_rew), label='Actor-Critic (MA10)', color='blue', linewidth=2)
    
    plt.plot(dqn_rew, color='red', alpha=0.2, linewidth=1)
    plt.plot(moving_average(dqn_rew), label='DQN (MA10)', color='red', linewidth=2)
    
    plt.title('Training Reward Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "reward_comparison.png"))
    plt.close()
    
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(ac_a_loss, label='Actor Loss', color='red', alpha=0.7)
    plt.title('Actor Loss')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(ac_c_loss, label='Critic Loss', color='green', alpha=0.7)
    plt.title('Critic Loss')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(dqn_loss, label='DQN Loss', color='purple', alpha=0.7)
    plt.title('DQN Loss')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "losses.png"))
    plt.close()

def plot_evaluation_episodes(ac_rewards, dqn_rewards):
    episodes = np.arange(1, len(ac_rewards) + 1)
    
    ac_running_mean = [np.mean(ac_rewards[:i+1]) for i in range(len(ac_rewards))]
    ac_running_std = [np.std(ac_rewards[:i+1]) for i in range(len(ac_rewards))]
    
    dqn_running_mean = [np.mean(dqn_rewards[:i+1]) for i in range(len(dqn_rewards))]
    dqn_running_std = [np.std(dqn_rewards[:i+1]) for i in range(len(dqn_rewards))]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(episodes, ac_running_mean, marker='o', label='Actor-Critic (Running Mean)', color='blue', linewidth=2)
    plt.fill_between(episodes, 
                     np.array(ac_running_mean) - np.array(ac_running_std), 
                     np.array(ac_running_mean) + np.array(ac_running_std), 
                     color='blue', alpha=0.2)
    
    plt.plot(episodes, dqn_running_mean, marker='s', label='DQN (Running Mean)', color='red', linewidth=2)
    plt.fill_between(episodes, 
                     np.array(dqn_running_mean) - np.array(dqn_running_std), 
                     np.array(dqn_running_mean) + np.array(dqn_running_std), 
                     color='red', alpha=0.2)
    
    plt.title('Evaluation Stability (Running Mean & Std across Episodes)')
    plt.xlabel('Evaluation Episode Count')
    plt.ylabel('Cumulative Mean Reward')
    plt.xticks(episodes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "evaluation_episodes.png"))
    plt.close()

if __name__ == "__main__":
    set_seed(SEED)
    
    env_ac = gym.make(ENV_NAME)
    ac_agent, ac_rewards, ac_a_loss, ac_c_loss, ac_time = train_ac(env_ac)
    env_ac.close()
    
    env_dqn = gym.make(ENV_NAME)
    dqn_agent, dqn_rewards, dqn_loss, dqn_time = train_dqn(env_dqn)
    env_dqn.close()
    
    generate_plots((ac_rewards, ac_a_loss, ac_c_loss), (dqn_rewards, dqn_loss))
    
    ac_agent.load(os.path.join(CKPT_DIR, "best_actor_critic.pt"))
    ac_eval_rewards = save_gifs_and_return_rewards(ac_agent, ENV_NAME, "ActorCritic")
    
    dqn_agent.load(os.path.join(CKPT_DIR, "best_dqn.pt"))
    dqn_eval_rewards = save_gifs_and_return_rewards(dqn_agent, ENV_NAME, "DQN")
    
    plot_evaluation_episodes(ac_eval_rewards, dqn_eval_rewards)

    ac_mean = np.mean(ac_eval_rewards)
    ac_std = np.std(ac_eval_rewards)
    dqn_mean = np.mean(dqn_eval_rewards)
    dqn_std = np.std(dqn_eval_rewards)
    
    results = {
        "Actor-Critic": {
            "wall_time": float(ac_time),
            "eval_mean_reward": float(ac_mean),
            "eval_reward_std": float(ac_std)
        },
        "DQN": {
            "wall_time": float(dqn_time),
            "eval_mean_reward": float(dqn_mean),
            "eval_reward_std": float(dqn_std)
        }
    }
    
    with open(os.path.join(EVAL_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print(f"AC Time: {ac_time:.1f}s | DQN Time: {dqn_time:.1f}s")
    print(f"AC Eval: {ac_mean:.1f} +/- {ac_std:.1f}")
    print(f"DQN Eval: {dqn_mean:.1f} +/- {dqn_std:.1f}")
    print(f"Results saved to {EVAL_DIR}")
    print("="*50)