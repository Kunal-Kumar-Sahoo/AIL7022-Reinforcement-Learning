import sys
# sys.path.append('/Users/kunalkumarsahoo/Playground/IITD/AIL7022/Assignment-2/Q2/gym_gridworlds')

import gymnasium
import gym_gridworlds
import numpy as np
import random
import imageio
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from gymnasium.envs.registration import register
from behaviour_policies import create_behaviour
from utils import setup_environment, set_global_seed, evaluate_policy
from graphics import generate_policy_gif, plot_reward_curve


def monte_carlo_off_policy_control(env, behavior_Q, num_episodes, seed, gamma, initial_temperature=1.0):
    """
    Off-Policy Monte Carlo Control using Weighted Importance Sampling.
    Fixed weight update order and effective probabilities with environment noise.
    """
    n_actions = env.action_space.n
    n_states = env.observation_space.n
    
    Q = defaultdict(lambda: np.zeros(n_actions))
    C = defaultdict(lambda: np.zeros(n_actions))
    episode_rewards = []
    min_temp = 0.01
    EPS = 1e-12
    max_W = 1e9  

    p_rand = getattr(getattr(env, 'unwrapped', env), 'random_action_prob', 0.0)
    uniform_prob = 1.0 / float(n_actions)

    def get_target_probs(state, temp):
        """Calculates softmax action probabilities."""
        q_values = Q[state]
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / temp)
        denom = np.sum(exp_q)
        return exp_q / denom if denom > 0 else np.ones(n_actions) / n_actions

    def behavior_policy(state):
        """Samples an action from behavior policy distribution."""
        return np.random.choice(np.arange(n_actions), p=behavior_Q[state])

    # Main training loop
    for i in range(num_episodes):
        # 1. Generate an episode
        episode = []
        state, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state
        
        G = 0.0
        W = 1.0
        temp = max(min_temp, initial_temperature * (0.999 ** i))

        # 2. Backward update using Weighted IS
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G

            pi_probs = get_target_probs(state, temp)
            b_probs = behavior_Q[state]

            # account for env internal random_action_prob
            pi_eff = (1.0 - p_rand) * pi_probs[action] + p_rand * uniform_prob
            b_eff = (1.0 - p_rand) * b_probs[action] + p_rand * uniform_prob

            if b_eff <= EPS:
                break

            # update W BEFORE updating Q for correctness
            W *= (pi_eff / b_eff)
            if not np.isfinite(W) or W == 0.0:
                break
            W = min(W, max_W)

            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
        
        # Evaluation checkpoint
        if i % 200 == 0:
            current_policy = np.array([np.argmax(Q[s]) if s in Q else 0 for s in range(n_states)])
            mean_eval_reward, _, _, _, _ = evaluate_policy(env, current_policy, n_episodes=10)
            episode_rewards.append(mean_eval_reward)
            print(f"[Episode {i}] Mean Eval Reward: {mean_eval_reward:.3f}")

    final_policy = np.array([np.argmax(Q[s]) if s in Q else 0 for s in range(n_states)])
    return Q, final_policy, episode_rewards


if __name__ == '__main__':
    os.makedirs("plots", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    
    NOISE_LEVELS = [0.0, 0.1, 0.01]
    NUM_SEEDS = 10
    NUM_EPISODES = 50000
    GAMMA = 0.99
    INITIAL_TEMPERATURE = 5.0

    ENV_ID = setup_environment()
    env = gymnasium.make(ENV_ID, random_action_prob=0.1)
    
    json_filename = "evaluation/importance_sampling_evaluation_results.json"
    evaluation_results = {}

    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            evaluation_results = json.load(f)
        
    for noise in NOISE_LEVELS:
        print(f"\n{'='*50}")
        print(f"STARTING MONTE CARLO EXPERIMENT FOR NOISE LEVEL: {noise}")
        print(f"{'='*50}")
        
        behavior_Q = create_behaviour(noise=noise)
        all_seeds_rewards = []
        best_policy_for_noise = None
        best_eval_reward_for_noise = -np.inf

        for seed in range(NUM_SEEDS):
            print(f"\n--- [Noise: {noise}] Training Seed {seed + 1}/{NUM_SEEDS} ---")
            set_global_seed(seed)
            
            q_values, policy, reward_history = monte_carlo_off_policy_control(
                env, behavior_Q, NUM_EPISODES, seed, GAMMA, INITIAL_TEMPERATURE
            )
            all_seeds_rewards.append(reward_history)
            
            mean_reward, _, _, _, _ = evaluate_policy(env, policy, n_episodes=100)
            
            if mean_reward > best_eval_reward_for_noise:
                best_eval_reward_for_noise = mean_reward
                best_policy_for_noise = policy
                print(f"Found new best policy for this noise level.")

        plot_reward_curve(all_seeds_rewards, noise, 'Monte Carlo', 'plots')
        
        print(f"\nEvaluating best policy for noise level {noise}...")
        mean_rew, _, _, std_rew, _ = evaluate_policy(env, best_policy_for_noise, n_episodes=100)
        evaluation_results[f"mc_{noise}"] = {"mean": mean_rew, "std": std_rew}
        print(f"Final Evaluation: Mean={mean_rew:.3f}, Std={std_rew:.3f}")
        
        gif_filename = f"gifs/monte_carlo_gif_{noise}.gif"
        generate_policy_gif(env, best_policy_for_noise, gif_filename)

    with open(json_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"\nAll experiments complete. Evaluation results updated in {json_filename}")

    env.close()
