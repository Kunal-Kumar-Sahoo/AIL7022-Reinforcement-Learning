import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import imageio
import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import pandas as pd 
from dataclasses import dataclass
from collections import defaultdict
from frozenlake import DiagonalFrozenLake
from cliff import MultiGoalCliffWalkingEnv
import copy

from utils import *
from update_rules import *

environment_settings = {
    'MultiGoalCliffWalkingEnv': {
        'NUM_SEEDS': 10,
        'NUM_EPISODES': 10000,
        'MAX_STEPS': 1000,
        'DISCOUNT_FACTOR': 0.99,
        'START_TEMPERATURE': 5.0,
        'TEMPERATURE_DECAY': 0.999,
        'STEP_SIZE': 0.9,
        'LAST_K': 100
    },

    'DiagonalFrozenLake': {
        'FROZEN_SEED': 12345,
        'NUM_EPISODES': 1000000,
        'MAX_STEPS': 100,
        'DISCOUNT_FACTOR': 1.0,
        'START_TEMPERATURE': 1.0,
        'TEMPERATURE_DECAY': 0.999995,
        'STEP_SIZE': 0.5,
    }
}


def SARSA(env: MultiGoalCliffWalkingEnv):
    '''
    Implement the SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    num_seeds = environment_settings['MultiGoalCliffWalkingEnv']['NUM_SEEDS']
    num_episodes = environment_settings['MultiGoalCliffWalkingEnv']['NUM_EPISODES']
    max_steps = environment_settings['MultiGoalCliffWalkingEnv']['MAX_STEPS']
    discount_factor = environment_settings['MultiGoalCliffWalkingEnv']['DISCOUNT_FACTOR']
    start_temperature = environment_settings['MultiGoalCliffWalkingEnv']['START_TEMPERATURE']
    temperature_decay = environment_settings['MultiGoalCliffWalkingEnv']['TEMPERATURE_DECAY']
    step_size = environment_settings['MultiGoalCliffWalkingEnv']['STEP_SIZE']
    last_k = environment_settings['MultiGoalCliffWalkingEnv']['LAST_K']

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    all_rewards = np.zeros((num_seeds, num_episodes))
    all_safe = np.zeros(num_seeds)
    all_risky = np.zeros(num_seeds)
    Qs = []
    seed_last_k_means = []

    for seed in range(num_seeds):
        rng = np.random.default_rng(1000 + seed)
        np.random.seed(1000 + seed)
        random.seed(1000 + seed)

        Q = np.zeros((n_states, n_actions))
        seed_rewards = np.zeros(num_episodes)
        safe_count = 0
        risky_count = 0

        for episode in range(num_episodes):
            temperature = start_temperature * (temperature_decay ** episode)
            ep_reward, steps, visited, Q = run_episode(env, Q, update_sarsa, step_size, 
                                                       discount_factor, temperature, max_steps, rng)
            seed_rewards[episode] = ep_reward
            safe_count += visited['safe']
            risky_count += visited['risky']

        all_rewards[seed] = seed_rewards
        all_safe[seed] = safe_count
        all_risky[seed] = risky_count
        Qs.append(copy.deepcopy(Q))

        k = min(last_k, num_episodes)
        seed_last_k_means.append(np.mean(seed_rewards[-k:]))

    index = int(np.argmax(seed_last_k_means))
    best_Q = Qs[index]

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_safe = float(np.mean(all_safe))
    avg_risky = float(np.mean(all_risky))

    return best_Q, avg_rewards, avg_safe, avg_risky


def q_learning_for_cliff(env: MultiGoalCliffWalkingEnv):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    num_seeds = environment_settings['MultiGoalCliffWalkingEnv']['NUM_SEEDS']
    num_episodes = environment_settings['MultiGoalCliffWalkingEnv']['NUM_EPISODES']
    max_steps = environment_settings['MultiGoalCliffWalkingEnv']['MAX_STEPS']
    discount_factor = environment_settings['MultiGoalCliffWalkingEnv']['DISCOUNT_FACTOR']
    start_temperature = environment_settings['MultiGoalCliffWalkingEnv']['START_TEMPERATURE']
    temperature_decay = environment_settings['MultiGoalCliffWalkingEnv']['TEMPERATURE_DECAY']
    step_size = environment_settings['MultiGoalCliffWalkingEnv']['STEP_SIZE']
    last_k = environment_settings['MultiGoalCliffWalkingEnv']['LAST_K']

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    all_rewards = np.zeros((num_seeds, num_episodes))
    all_safe = np.zeros(num_seeds)
    all_risky = np.zeros(num_seeds)
    Qs = []
    seed_last_k_means = []

    for seed in range(num_seeds):
        rng = np.random.default_rng(1000 + seed)
        np.random.seed(1000 + seed)
        random.seed(1000 + seed)

        Q = np.zeros((n_states, n_actions))
        seed_rewards = np.zeros(num_episodes)
        safe_count = 0
        risky_count = 0

        for episode in range(num_episodes):
            temperature = start_temperature * (temperature_decay ** episode)
            ep_reward, steps, visited, Q = run_episode(env, Q, update_qlearning, step_size, 
                                                       discount_factor, temperature, max_steps, rng)
            seed_rewards[episode] = ep_reward
            safe_count += visited['safe']
            risky_count += visited['risky']

        all_rewards[seed] = seed_rewards
        all_safe[seed] = safe_count
        all_risky[seed] = risky_count
        Qs.append(copy.deepcopy(Q))

        k = min(last_k, num_episodes)
        seed_last_k_means.append(np.mean(seed_rewards[-k:]))

    index = int(np.argmax(seed_last_k_means))
    best_Q = Qs[index]

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_safe = float(np.mean(all_safe))
    avg_risky = float(np.mean(all_risky))

    return best_Q, avg_rewards, avg_safe, avg_risky

def expected_SARSA(env: MultiGoalCliffWalkingEnv):
    '''
    Implement the Expected SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    num_seeds = environment_settings['MultiGoalCliffWalkingEnv']['NUM_SEEDS']
    num_episodes = environment_settings['MultiGoalCliffWalkingEnv']['NUM_EPISODES']
    max_steps = environment_settings['MultiGoalCliffWalkingEnv']['MAX_STEPS']
    discount_factor = environment_settings['MultiGoalCliffWalkingEnv']['DISCOUNT_FACTOR']
    start_temperature = environment_settings['MultiGoalCliffWalkingEnv']['START_TEMPERATURE']
    temperature_decay = environment_settings['MultiGoalCliffWalkingEnv']['TEMPERATURE_DECAY']
    step_size = environment_settings['MultiGoalCliffWalkingEnv']['STEP_SIZE']
    last_k = environment_settings['MultiGoalCliffWalkingEnv']['LAST_K']

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    all_rewards = np.zeros((num_seeds, num_episodes))
    all_safe = np.zeros(num_seeds)
    all_risky = np.zeros(num_seeds)
    Qs = []
    seed_last_k_means = []

    for seed in range(num_seeds):
        rng = np.random.default_rng(1000 + seed)
        np.random.seed(1000 + seed)
        random.seed(1000 + seed)

        Q = np.zeros((n_states, n_actions))
        seed_rewards = np.zeros(num_episodes)
        safe_count = 0
        risky_count = 0

        for episode in range(num_episodes):
            temperature = start_temperature * (temperature_decay ** episode)
            ep_reward, steps, visited, Q = run_episode(env, Q, update_expected_sarsa, 
                                                       step_size, discount_factor, temperature, max_steps, rng)
            seed_rewards[episode] = ep_reward
            safe_count += visited["safe"]
            risky_count += visited["risky"]

        all_rewards[seed] = seed_rewards
        all_safe[seed] = safe_count
        all_risky[seed] = risky_count
        Qs.append(copy.deepcopy(Q))

        k = min(last_k, num_episodes)
        seed_last_k_means.append(np.mean(seed_rewards[-k:]))

    index = int(np.argmax(seed_last_k_means))
    best_Q = Qs[index]

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_safe = float(np.mean(all_safe))
    avg_risky = float(np.mean(all_risky))

    return best_Q, avg_rewards, avg_safe, avg_risky


def monte_carlo(env: DiagonalFrozenLake):
    '''
    Implement the Monte Carlo algorithm to find the optimal policy for the given environment.
    Return Q table.
    return: Q table -> np.array of shape (num_states, num_actions)
    return: episode_rewards -> []
    return: _ 
    return: _ 
    '''
    frozen_seed = environment_settings['DiagonalFrozenLake']['FROZEN_SEED']
    num_episodes = environment_settings['DiagonalFrozenLake']['NUM_EPISODES']
    max_steps = environment_settings['DiagonalFrozenLake']['MAX_STEPS']
    discount_factor = environment_settings['DiagonalFrozenLake']['DISCOUNT_FACTOR']
    start_temperature = environment_settings['DiagonalFrozenLake']['START_TEMPERATURE']
    temperature_decay = environment_settings['DiagonalFrozenLake']['TEMPERATURE_DECAY']

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rng = np.random.default_rng(frozen_seed)
    np.random.seed(frozen_seed)
    random.seed(frozen_seed)

    Q = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))
    episode_rewards = np.zeros(num_episodes, dtype=float)

    for episode in range(num_episodes):
        temperature = start_temperature * (temperature_decay ** episode)
        episode_list = []
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31-1)))
        state = int(obs)
        total_reward = 0.0

        for t in range(max_steps):
            action, _ = sample_softmax_action(Q, state, temperature, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = int(next_obs)
            episode_list.append((state, action, float(reward)))
            total_reward += reward
            state = next_state
            if done:
                break

        episode_rewards[episode] = float(total_reward)

        returns = 0.0
        visited_sa = set()

        for (state, action, reward) in reversed(episode_list):
            returns = reward + discount_factor * returns
            if (state, action) not in visited_sa:
                visited_sa.add((state, action))
                returns_count[state, action] += 1
                n = returns_count[state, action]
                Q[state, action] += (returns - Q[state, action]) / n

    return Q, episode_rewards

def q_learning_for_frozenlake(env: DiagonalFrozenLake):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: Q table -> np.array of shape (num_states, num_actions)
    return episode_rewards_for_one_seed -> []
    '''
    frozen_seed = environment_settings['DiagonalFrozenLake']['FROZEN_SEED']
    num_episodes = environment_settings['DiagonalFrozenLake']['NUM_EPISODES']
    max_steps = environment_settings['DiagonalFrozenLake']['MAX_STEPS']
    discount_factor = environment_settings['DiagonalFrozenLake']['DISCOUNT_FACTOR']
    start_temperature = environment_settings['DiagonalFrozenLake']['START_TEMPERATURE']
    temperature_decay = environment_settings['DiagonalFrozenLake']['TEMPERATURE_DECAY']
    step_size = environment_settings['DiagonalFrozenLake']['STEP_SIZE']

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rng = np.random.default_rng(frozen_seed)
    np.random.seed(frozen_seed)
    random.seed(frozen_seed)

    Q = np.zeros((n_states, n_actions))
    episode_rewards = np.zeros(num_episodes)

    for episode in range(num_episodes):
        temperature = start_temperature * (temperature_decay ** episode)
        episode_reward, steps, visited, Q = run_episode(env, Q, update_qlearning, step_size, 
                                                        discount_factor, temperature, max_steps, rng)
        episode_rewards[episode] = episode_reward
        
    return Q, episode_rewards