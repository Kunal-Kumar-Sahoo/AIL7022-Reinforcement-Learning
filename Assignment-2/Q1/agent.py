import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import imageio
import numpy as np
import math
import copy
import os
import random
import matplotlib.pyplot as plt
import pandas as pd 
from dataclasses import dataclass
from collections import defaultdict
from frozenlake import DiagonalFrozenLake
from cliff import MultiGoalCliffWalkingEnv


NUM_SEEDS = 10
NUM_EPISODES = 10000
MAX_STEPS = 1000
alpha = 0.5
gamma = 0.99
tau0 = 5.0
tau_decay = 0.999
last_k = 100  # choose best Q by mean of last_k episodes

# Helper: Softmax policy
def softmax_action_probs(Q_row, temperature):
    z = Q_row - np.max(Q_row)
    exp = np.exp(z / (temperature + 1e-12))
    probs = exp / (np.sum(exp) + 1e-12)
    return probs

def sample_softmax_action(Q, state, temperature, rng):
    probs = softmax_action_probs(Q[state], temperature)
    return rng.choice(len(probs), p=probs), probs


def update_sarsa(Q, state, action, reward, next_state, terminated,
                 alpha, gamma, temperature, rng):
    # Sample next action from softmax (on-policy)
    next_action, _ = sample_softmax_action(Q, next_state, temperature, rng)
    td_target = reward + gamma * Q[next_state, next_action] * (not terminated)
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return Q, next_action

def update_qlearning(Q, state, action, reward, next_state, terminated,
                     alpha, gamma, temperature, rng):
    td_target = reward + gamma * np.max(Q[next_state]) * (not terminated)
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return Q, None

def update_expected_sarsa(Q, state, action, reward, next_state, terminated,
                          alpha, gamma, temperature, rng):
    probs = softmax_action_probs(Q[next_state], temperature)
    expected_q = np.dot(probs, Q[next_state])
    td_target = reward + gamma * expected_q * (not terminated)
    td_error = td_target - Q[state, action]
    Q[state, action] += alpha * td_error
    return Q, None

# Generic episode runner with injected update function
def run_episode_generic(env, Q, update_fn, alpha, gamma, temperature, max_steps, rng):
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    state = int(obs)
    total_reward = 0.0
    visited = {"safe": 0, "risky": 0}

    # Initial behavior action
    action, _ = sample_softmax_action(Q, state, temperature, rng)

    for t in range(max_steps):
        obs2, reward, terminated, truncated, info = env.step(int(action))
        next_state = int(obs2)
        total_reward += reward

        goal_str = info.get("goal", None)
        if goal_str == "safe":
            visited["safe"] += 1
        elif goal_str == "risky":
            visited["risky"] += 1

        # Call injected update function
        Q, maybe_next_action = update_fn(Q, state, action, reward, next_state,
                                         terminated, alpha, gamma, temperature, rng)

        # Choose action for next timestep
        if maybe_next_action is not None:
            action = maybe_next_action
        else:
            action, _ = sample_softmax_action(Q, next_state, temperature, rng)

        state = next_state

        if terminated or truncated:
            break

    return total_reward, t + 1, visited, Q


def SARSA(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    all_rewards = np.zeros((NUM_SEEDS, NUM_EPISODES))
    all_safe = np.zeros(NUM_SEEDS)
    all_risky = np.zeros(NUM_SEEDS)
    Qs = []
    seed_last_k_means = []

    for s in range(NUM_SEEDS):
        rng = np.random.default_rng(1000 + s)
        np.random.seed(1000 + s)
        random.seed(1000 + s)

        Q = np.zeros((n_states, n_actions))
        seed_rewards = np.zeros(NUM_EPISODES)
        safe_count = 0
        risky_count = 0

        for ep in range(NUM_EPISODES):
            temperature = tau0 * (tau_decay ** ep)
            ep_reward, steps, visited, Q = run_episode_generic(env, Q, update_sarsa,
                                                               alpha, gamma, temperature, MAX_STEPS, rng)
            seed_rewards[ep] = ep_reward
            safe_count += visited["safe"]
            risky_count += visited["risky"]

        all_rewards[s] = seed_rewards
        all_safe[s] = safe_count
        all_risky[s] = risky_count
        Qs.append(copy.deepcopy(Q))

        # compute mean over last_k episodes (fallback to full mean if NUM_EPISODES < last_k)
        k = min(last_k, NUM_EPISODES)
        seed_last_k_means.append(np.mean(seed_rewards[-k:]))

    # Choose best Q as the seed with highest mean over last_k episodes
    best_idx = int(np.argmax(seed_last_k_means))
    best_Q = Qs[best_idx]

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_safe = float(np.mean(all_safe))
    avg_risky = float(np.mean(all_risky))

    return best_Q, avg_rewards, avg_safe, avg_risky

def q_learning(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    all_rewards = np.zeros((NUM_SEEDS, NUM_EPISODES))
    all_safe = np.zeros(NUM_SEEDS)
    all_risky = np.zeros(NUM_SEEDS)
    Qs = []
    seed_last_k_means = []

    for s in range(NUM_SEEDS):
        rng = np.random.default_rng(2000 + s)
        np.random.seed(2000 + s)
        random.seed(2000 + s)

        Q = np.zeros((n_states, n_actions))
        seed_rewards = np.zeros(NUM_EPISODES)
        safe_count = 0
        risky_count = 0

        for ep in range(NUM_EPISODES):
            temperature = tau0 * (tau_decay ** ep)
            ep_reward, steps, visited, Q = run_episode_generic(env, Q, update_qlearning,
                                                               alpha, gamma, temperature, MAX_STEPS, rng)
            seed_rewards[ep] = ep_reward
            safe_count += visited["safe"]
            risky_count += visited["risky"]

        all_rewards[s] = seed_rewards
        all_safe[s] = safe_count
        all_risky[s] = risky_count
        Qs.append(copy.deepcopy(Q))

        k = min(last_k, NUM_EPISODES)
        seed_last_k_means.append(np.mean(seed_rewards[-k:]))

    best_idx = int(np.argmax(seed_last_k_means))
    best_Q = Qs[best_idx]

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_safe = float(np.mean(all_safe))
    avg_risky = float(np.mean(all_risky))

    return best_Q, avg_rewards, avg_safe, avg_risky

def expected_SARSA(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    all_rewards = np.zeros((NUM_SEEDS, NUM_EPISODES))
    all_safe = np.zeros(NUM_SEEDS)
    all_risky = np.zeros(NUM_SEEDS)
    Qs = []
    seed_last_k_means = []

    for s in range(NUM_SEEDS):
        rng = np.random.default_rng(3000 + s)
        np.random.seed(3000 + s)
        random.seed(3000 + s)

        Q = np.zeros((n_states, n_actions))
        seed_rewards = np.zeros(NUM_EPISODES)
        safe_count = 0
        risky_count = 0

        for ep in range(NUM_EPISODES):
            temperature = tau0 * (tau_decay ** ep)
            ep_reward, steps, visited, Q = run_episode_generic(env, Q, update_expected_sarsa,
                                                               alpha, gamma, temperature, MAX_STEPS, rng)
            seed_rewards[ep] = ep_reward
            safe_count += visited["safe"]
            risky_count += visited["risky"]

        all_rewards[s] = seed_rewards
        all_safe[s] = safe_count
        all_risky[s] = risky_count
        Qs.append(copy.deepcopy(Q))

        k = min(last_k, NUM_EPISODES)
        seed_last_k_means.append(np.mean(seed_rewards[-k:]))

    best_idx = int(np.argmax(seed_last_k_means))
    best_Q = Qs[best_idx]

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_safe = float(np.mean(all_safe))
    avg_risky = float(np.mean(all_risky))

    return best_Q, avg_rewards, avg_safe, avg_risky


def monte_carlo(env):
    '''
    Implement the Monte Carlo algorithm to find the optimal policy for the given environment.
    Return Q table.
    return: Q table -> np.array of shape (num_states, num_actions)
    return: episode_rewards -> []
    return: _ 
    return: _ 
    '''
    pass

    return Q, episode_rewards, _, _