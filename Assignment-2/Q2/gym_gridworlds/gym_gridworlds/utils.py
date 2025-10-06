import os
import random
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_environment():
    env_id = 'Gym-Gridworlds/Full-4x5-v0'
    register(
        id=env_id,
        entry_point='gym_gridworlds.gridworld:Gridworld',
        max_episode_steps=500,
        kwargs={'grid': '4x5_full'}
    )
    return env_id

def evaluate_policy(env, policy, n_episodes=100):
    rewards = []
    success_rate = 0
    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0
        steps = 0
       
        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        
        rewards.append(episode_reward)

        if episode_reward > 0.5:
            success_rate += 1
    
    mean_reward = np.mean(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    std_reward = np.std(rewards)
    final_success_rate = success_rate / n_episodes

    return mean_reward, min_reward, max_reward, std_reward, final_success_rate


