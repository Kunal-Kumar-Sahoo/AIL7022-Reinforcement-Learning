import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
import json
import imageio
import time
from tqdm import tqdm
import csv

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

USE_CUDA = torch.cuda.is_available()


CONFIG = {
    "env_id": "LunarLander-v3",
    "seed": 42,
    "num_episodes": 10000,
    "max_t": 1000,
    "gamma": 0.99,
    "tau": 1e-2,
    "temp_start": 5.0,
    "temp_end": 0.1,
    "temp_decay": 0.999,
    "hidden_layers": 3,      
    "hidden_neurons": 1024,  
    "eval_episodes": 100,
    "lr": 5e-4,
    "buffer_size": int(1e5),
    "batch_size": 64,
    "update_every": 4,
}


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_neurons):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        num_hidden_layers = CONFIG["hidden_layers"] 
        
        self.hidden_layers = nn.ModuleList()
        
        if num_hidden_layers == 0:
            self.output_layer = nn.Linear(state_size, action_size)
        else:
            self.hidden_layers.append(nn.Linear(state_size, hidden_neurons))
            
            for _ in range(num_hidden_layers - 1):
                self.hidden_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            
            self.output_layer = nn.Linear(hidden_neurons, action_size)

        self.num_hidden_layers = num_hidden_layers

    def forward(self, state):
        if self.num_hidden_layers == 0:
            return self.output_layer(state)
        
        x = state
        for layer in self.hidden_layers:
            x = F.silu(layer(x))
        
        return self.output_layer(x)


class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed, CONFIG["hidden_neurons"]).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, CONFIG["hidden_neurons"]).to(device)

        try:
            self.qnetwork_local = torch.compile(self.qnetwork_local)
            self.qnetwork_target = torch.compile(self.qnetwork_target)
            print("PyTorch 2.0 compilation enabled.")
        except Exception as e:
            print(f"PyTorch 2.0 compilation failed (safe to ignore): {e}")
        
        self.optimizer = optim.AdamW(self.qnetwork_local.parameters(), lr=CONFIG["lr"])
        self.memory = deque(maxlen=CONFIG["buffer_size"])
        self.t_step = 0
        self.batch_size = CONFIG["batch_size"]
        self.update_every = CONFIG["update_every"]

    def step(self, state, action, reward, next_state, next_action, done):
        self.memory.append((state, action, reward, next_state, next_action, done))
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = random.sample(self.memory, k=self.batch_size)
                self.learn(experiences)

    def act(self, state, temp=0.01):
        if temp <= 0.0:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        action_values_temp = action_values / temp
        probs = F.softmax(action_values_temp, dim=1).cpu()
        action = torch.multinomial(probs, 1).item()
        return action

    def learn(self, experiences):
        states, actions, rewards, next_states, next_actions, dones = self._unpack_experiences(experiences)

        q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        q_targets = rewards + (CONFIG["gamma"] * q_targets_next * (1 - dones))

        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(q_expected, q_targets)
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, CONFIG["tau"])

    def _unpack_experiences(self, experiences):
        states, actions, rewards, next_states, next_actions, dones = zip(*experiences)
        
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(device)
        actions = torch.from_numpy(np.array(actions, dtype=np.int64)).to(device).unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(device).unsqueeze(1)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(device)
        next_actions = torch.from_numpy(np.array(next_actions, dtype=np.int64)).to(device).unsqueeze(1)
        dones = torch.from_numpy(np.array(dones, dtype=np.uint8)).float().to(device).unsqueeze(1)
        
        return states, actions, rewards, next_states, next_actions, dones

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def train_agent():
    env = gym.make(CONFIG["env_id"])
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=CONFIG["seed"])

    scores_window = deque(maxlen=100)
    temp = CONFIG["temp_start"]
    best_eval_score = -np.inf
    log_file = 'evaluation/training_log.csv'

    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "average_score_100", "temperature"])

    pbar = tqdm(range(1, CONFIG["num_episodes"] + 1), desc="Training Progress")
    for i_episode in pbar:
        state, _ = env.reset(seed=CONFIG["seed"] + i_episode)
        score = 0
        
        action = agent.act(state, temp)
        done = False
        
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_action = agent.act(next_state, temp)
            
            agent.step(state, action, reward, next_state, next_action, done)
            
            state, action = next_state, next_action
            score += reward
            if done:
                break
                
        scores_window.append(score)
        temp = max(CONFIG["temp_end"], CONFIG["temp_decay"] * temp)
        
        pbar.set_postfix({
            "Avg Score": f"{np.mean(scores_window):.2f}",
            "Best Eval": f"{best_eval_score:.2f}",
            "Temp": f"{temp:.2f}",
            "Buffer": f"{len(agent.memory)}"
        })

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([i_episode, score, np.mean(scores_window), temp])
        
        if i_episode % 50 == 0:
            eval_score = evaluate(env, agent, num_episodes=10, is_training=True)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                torch.save(agent.qnetwork_local.state_dict(), 'models/best_deep_sarsa.pt')
                
    env.close()

def evaluate(env, agent, num_episodes=100, is_training=False):
    rewards = []
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state, temp=0.0) # Act greedily
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        rewards.append(episode_reward)

    mean_reward = np.mean(rewards)
    
    if not is_training:
        std_reward = np.std(rewards)
        print(f"\nFinal Evaluation -> Mean: {mean_reward:.2f} +/- {std_reward:.2f}")
        results = {"mean_reward": mean_reward, "std_reward": std_reward}
        with open('evaluation/lunarlander_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Evaluation results saved.")

    return mean_reward

def create_gif():
    env = gym.make(CONFIG["env_id"], render_mode='rgb_array')
    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=CONFIG["seed"])
    
    try:
        agent.qnetwork_local.load_state_dict(torch.load('models/best_deep_sarsa.pt', map_location=device))
    except FileNotFoundError:
        print("Model file not found. Could not create GIF. Run training first.")
        return
        
    agent.qnetwork_local.eval()

    frames = []
    state, _ = env.reset()
    done = False
    print("Creating GIF...")
    while not done:
        frames.append(env.render())
        action = agent.act(state, temp=0.0) 
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    env.close()
    imageio.mimsave('gifs/lunarlander.gif', frames, fps=30)
    print("GIF saved.")

if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if USE_CUDA:
        torch.cuda.manual_seed(CONFIG["seed"])

    os.makedirs("models", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    
    train_agent()
    
    print("\nStarting final evaluation...")
    eval_env = gym.make(CONFIG["env_id"])
    eval_agent = Agent(state_size=eval_env.observation_space.shape[0], action_size=eval_env.action_space.n, seed=CONFIG["seed"])
    try:
        eval_agent.qnetwork_local.load_state_dict(torch.load('models/best_deep_sarsa.pt', map_location=device))
        evaluate(eval_env, eval_agent, num_episodes=CONFIG["eval_episodes"])
    except FileNotFoundError:
        print("Could not load model for evaluation. Skipping.")
    eval_env.close()
    
    create_gif()
