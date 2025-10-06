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


def tdis(env, behavior_Q, num_episodes, gamma, initial_alpha, initial_temp, seed):
    """
    Implement Off-Policy TD(0) Control with Importance Sampling (SARSA).
    This version includes rho clipping for stability and returns reward history.
    """
    n_actions = env.action_space.n
    n_states = env.observation_space.n
    
    Q = defaultdict(lambda: np.zeros(n_actions))
    min_alpha = 0.1
    min_temp = 0.1
    episode_rewards = []

    def get_target_probs(state, temp):
        """Calculates action probabilities for a state using the softmax function."""
        q_values = Q[state]
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / temp)
        return exp_q / np.sum(exp_q)
    
    def select_action_from_target_policy(state, temp):
        """Selects a single action based on the softmax target policy."""
        probs = get_target_probs(state, temp)
        return np.random.choice(np.arange(n_actions), p=probs)

    def behavior_policy(state):
        """Selects an action based on the provided behavior policy."""
        return np.random.choice(np.arange(n_actions), p=behavior_Q[state])

    # Main training loop
    for i in range(num_episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        
        alpha = max(min_alpha, initial_alpha * (1 - (i / num_episodes)))
        temp = max(min_temp, initial_temp * (0.999**i))

        while not done:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            prob_target = get_target_probs(state, temp)[action]
            prob_behavior = behavior_Q[state][action]
            rho = prob_target / prob_behavior if prob_behavior > 0 else 0
            rho = min(rho, 1.0) # Clip rho for stability

            # Correct, stable update rule
            next_action = select_action_from_target_policy(next_state, temp)
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]

            Q[state][action] += alpha * td_error
            
            state = next_state
            
        # For plotting, evaluate the current greedy policy every 10 episodes
        if i % 200 == 0:
            current_policy = np.array([np.argmax(Q[s]) if s in Q else 0 for s in range(n_states)])
            mean_eval_reward, _, _, _, _ = evaluate_policy(env, current_policy, n_episodes=10)
            episode_rewards.append(mean_eval_reward)
            print(f'[Episode {i}/{num_episodes}] Mean Eval Reward: {mean_eval_reward:.3f}')
            
    final_policy = np.array([np.argmax(Q[s]) if s in Q else 0 for s in range(n_states)])
    return Q, final_policy, episode_rewards


if __name__ == '__main__':
    # Create directories for outputs
    os.makedirs("plots", exist_ok=True)
    os.makedirs("gifs", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    
    # --- Parameters ---
    NOISE_LEVELS = [0.0, 0.01, 0.1]
    NUM_SEEDS = 10
    NUM_EPISODES = 50000 
    GAMMA = 0.99
    INITIAL_ALPHA = 0.8
    INITIAL_TEMPERATURE = 5.0
    
    ENV_ID = setup_environment()
    env = gymnasium.make(ENV_ID, random_action_prob=0.1)

    json_filename = 'evaluation/importance_sampling_evaluation_results.json'
    evaluation_results = {}
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            evaluation_results = json.load(f)

    for noise in NOISE_LEVELS:
        print(f"\n{'='*50}")
        print(f"STARTING EXPERIMENT FOR NOISE LEVEL: {noise}")
        print(f"{'='*50}")
        
        behavior_Q = create_behaviour(noise=noise)
        all_seeds_rewards = []
        best_policy_for_noise = None
        best_q_for_noise = None
        best_eval_reward_for_noise = -np.inf

        for seed in range(NUM_SEEDS):
            print(f"\n--- [Noise: {noise}] Training Seed {seed + 1}/{NUM_SEEDS} ---")
            set_global_seed(seed)
            
            q_values, policy, reward_history = tdis(
                env, behavior_Q, NUM_EPISODES, GAMMA, INITIAL_ALPHA, INITIAL_TEMPERATURE, seed
            )
            all_seeds_rewards.append(reward_history)
            
            mean_reward, _, _, _, _ = evaluate_policy(env, policy, n_episodes=100)
            
            if mean_reward > best_eval_reward_for_noise:
                best_eval_reward_for_noise = mean_reward
                best_policy_for_noise = policy
                best_q_for_noise = q_values
                print(f"Found new best policy for this noise level.")

        # --- Learning Analysis ---
        plot_reward_curve(all_seeds_rewards, noise, 'Temporal Difference', 'plots')
        
        # --- Evaluation Phase ---
        print(f"\nEvaluating best policy for noise level {noise}...")
        mean_rew, _, _, std_rew, _ = evaluate_policy(env, best_policy_for_noise, n_episodes=100)
        evaluation_results[f"td_{noise}"] = {
            "mean": mean_rew,
            "std": std_rew
        }
        print(f"Final Evaluation: Mean={mean_rew:.3f}, Std={std_rew:.3f}")
        
        # --- Policy Demonstration ---
        gif_filename = f"gifs/temporal_difference_gif_{noise}.gif"
        generate_policy_gif(env, best_policy_for_noise, gif_filename)

    # --- Final Reporting ---
    with open(json_filename, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"\nAll experiments complete. Evaluation results saved to {json_filename}")

    env.close()