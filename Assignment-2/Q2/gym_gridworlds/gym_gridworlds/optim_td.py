import sys; sys.path.append('/Users/kunalkumarsahoo/Playground/IITD/AIL7022/Assignment-2/Q2/gym_gridworlds')

import gymnasium
import gym_gridworlds
import numpy as np
import random
import os
from collections import defaultdict
from gymnasium.envs.registration import register
from behaviour_policies import create_behaviour
import itertools

# --- Environment Setup ---
GRID_ROWS = 4
GRID_COLS = 5
register(
    id="Gym-Gridworlds/Full-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={"grid": "4x5_full"},
)

# --- Helper Functions ---
def set_global_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

behavior_Q = create_behaviour()

# --- Core TD Learning Algorithm ---
def tdis(env, params):
    """
    Implement Off-Policy TD Control with Importance Sampling (Expected SARSA).
    This version includes decaying hyperparameters for more stable learning.
    
    Args:
        env: The environment instance.
        params (dict): A dictionary containing all hyperparameters for the run.
        
    Returns:
        Q (defaultdict): The learned action-value function.
    """
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Unpack hyperparameters
    num_episodes = params['num_episodes']
    gamma = params['gamma']
    initial_alpha = params['alpha']
    initial_temp = params['temperature']
    min_alpha = 0.1
    min_temp = 0.1

    def get_target_probs(state, temp):
        """Calculates action probabilities using the softmax function."""
        q_values = Q[state]
        max_q = np.max(q_values)
        exp_q = np.exp((q_values - max_q) / temp)
        return exp_q / np.sum(exp_q)

    def behavior_policy(state):
        """Selects an action based on the pre-loaded behavior policy."""
        return np.random.choice(np.arange(env.action_space.n), p=behavior_Q[state])

    # Main training loop
    for i in range(num_episodes):
        state, _ = env.reset(seed=params['seed'] + i)
        done = False
        
        # --- Decaying Hyperparameters ---
        # Linearly decay alpha
        alpha = max(min_alpha, initial_alpha * (1 - (i / num_episodes)))
        # Exponentially decay temperature
        temp = max(min_temp, initial_temp * (0.999**i))

        while not done:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Importance Sampling Ratio
            prob_target = get_target_probs(state, temp)[action]
            prob_behavior = behavior_Q[state][action]
            rho = prob_target / prob_behavior if prob_behavior > 0 else 0
            
            # Off-policy TD target (Expected SARSA)
            if done:
                expected_next_q = 0
            else:
                next_q_values = Q[next_state]
                next_action_probs = get_target_probs(next_state, temp)
                expected_next_q = np.sum(next_q_values * next_action_probs)

            td_target = reward + gamma * expected_next_q
            td_error = td_target - Q[state][action]
            
            Q[state][action] += alpha * rho * td_error
            state = next_state
            
    return Q

# --- Evaluation Function ---
def evaluate_policy(env, policy, n_episodes=100):
    """Evaluate a deterministic policy and return the mean reward."""
    total_reward = 0
    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        done = False
        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    return total_reward / n_episodes

# --- Main Tuning Script ---
if __name__ == '__main__':
    # --- Hyperparameter Search Space ---
    param_grid = {
        'alpha': [0.1, 0.2, 0.5, 0.8, 1.0],
        'temperature': [0.5, 1.0, 2.0],
        'gamma': [0.90, 0.95, 0.99, 1.0] # Usually kept fixed, but can be tuned
    }
    
    # --- Tuning Configuration ---
    num_episodes_per_run = 3000 # Fewer episodes for faster tuning
    seeds_per_combination = 5 # Use 5 seeds to get a stable average
    
    # --- Tracking Variables ---
    best_avg_reward = -np.inf
    best_params = {}
    
    # Create all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"--- Starting Hyperparameter Tuning ---")
    print(f"Testing {len(param_combinations)} combinations, each averaged over {seeds_per_combination} seeds.")
    
    # --- Grid Search Loop ---
    for i, params in enumerate(param_combinations):
        
        run_rewards = []
        print(f"\n[{i+1}/{len(param_combinations)}] Testing params: {params}")
        
        # Average results over several seeds
        for seed in range(seeds_per_combination):
            set_global_seed(seed)
            env = gymnasium.make('Gym-Gridworlds/Full-4x5-v0', random_action_prob=0.1)
            
            # Add static params for this run
            run_params = params.copy()
            run_params['num_episodes'] = num_episodes_per_run
            run_params['seed'] = seed

            # Train the agent
            learned_Q = tdis(env, run_params)
            
            # Create a deterministic policy from Q-values
            policy = np.zeros(env.observation_space.n, dtype=int)
            for s in range(env.observation_space.n):
                policy[s] = np.argmax(learned_Q[s])
                
            # Evaluate the policy
            mean_reward = evaluate_policy(env, policy)
            run_rewards.append(mean_reward)
            env.close()

        # Calculate the average performance for this set of hyperparameters
        avg_reward = np.mean(run_rewards)
        print(f"-> Average reward over {seeds_per_combination} seeds: {avg_reward:.3f}")
        
        # Check if this is the best combination so far
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_params = params
            print(f"*** New best parameter set found! ***")

    # --- Final Report ---
    print("\n\n--- Hyperparameter Tuning Complete ---")
    print(f"Best average reward achieved: {best_avg_reward:.3f}")
    print("Best hyperparameter set:")
    for key, value in best_params.items():
        print(f"  - {key}: {value}")
