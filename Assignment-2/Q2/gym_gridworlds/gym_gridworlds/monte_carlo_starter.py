import gymnasium
import gym_gridworlds
import numpy as np
import random
import imageio
import os
import matplotlib.pyplot as plt 
from collections import defaultdict
from gymnasium.envs.registration import register
from behaviour_policies import create_behaviour

GRID_ROWS = 4
GRID_COLS = 5

# Register the custom environment
register(
    id="Gym-Gridworlds/Full-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={"grid": "4x5_full"},
)

def set_global_seed(seed: int):
    """Set seed for reproducibility across modules."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def state_to_cord(state):
    """Convert state number to (row, col) coordinates."""
    return divmod(state, GRID_COLS)

def cord_to_state(row, col):
    """Convert (row, col) coordinates to state number."""
    return row * GRID_COLS + col

# Create the behavior policy
behavior_Q = create_behaviour()

def monte_carlo_off_policy_control(env, num_episodes, seed, gamma):
    """
    Implement Monte Carlo off-policy control using importance sampling.
    
    Args:
        env: The environment
        num_episodes: Number of episodes to train
        seed: Random seed for episode generation
        gamma: Discount factor
        
    Returns:
        Q: Action-value function (dict of state -> array of action values)
        final_policy: Final deterministic policy (array)
    """
    
    # TODO:
    # Hint: Use defaultdict for initialization
    Q = # YOUR CODE HERE
    C = # YOUR CODE HERE

    def target_policy(state):
        """
        TODO: Implement the target policy 
        
        Args:
            state: Current state
            
        Returns:
            action: Action to take (integer)
        """
        return # YOUR CODE HERE

    def behavior_policy(state):
        """
        TODO: Implement the behavior policy
        
        Args:
            state: Current state
            
        Returns:
            action: Action to take (integer)
        """
        return # YOUR CODE HERE

    def get_behavior_prob(state, action):
        """TODO: Get the probability of taking action in state under behavior policy."""
        pass

    def get_target_prob(state, action):
        """
        TODO: Get the probability of taking action in state under target policy
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            probability
        """
        return # YOUR CODE HERE

    # Main training loop
            
    return Q, final_policy

def evaluate_policy(env, policy, n_episodes=100, max_steps=500):
    """
    Evaluate a given policy by running it for multiple episodes.
    
    Args:
        env: The environment
        policy: Policy to evaluate (array of actions for each state)
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        
    Returns:
        tuple: (mean_reward, min_reward, max_reward, std_reward, success_rate)
    """
    rewards = []
    success_rate = 0

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
            
        rewards.append(episode_reward)
        
        # Consider episode successful if reward > 0.5
        if episode_reward > 0.5:
            success_rate += 1

    return (np.mean(rewards), np.min(rewards), np.max(rewards), 
            np.std(rewards), success_rate / n_episodes)

def generate_policy_gif(env, policy, filename='policy_run_mcmc.gif', max_steps=500):
    """Generate a GIF showing the policy in action."""
    # TODO:
    
    frames = []
    print(f"\nGenerating GIF... saving to {filename}")
    
    env_render = gymnasium.make(env.spec.id, render_mode='rgb_array', random_action_prob=0.1)
    state, _ = env_render.reset()
    done = False
    steps = 0
    
    while not done and steps < max_steps:
        frames.append(env_render.render())
        
        
    env_render.close()
    
    imageio.mimsave(filename, frames, fps=3)
    print(f"GIF saved successfully to {os.path.abspath(filename)}")

if __name__ == '__main__':
    num_seeds = 10  
    num_episodes = 1  # TODO:CHANGE TO SUITABLE VALUE
    
    env = gymnasium.make('Gym-Gridworlds/Full-4x5-v0', random_action_prob=0.1)
    
    best_policy = None
    best_q_values = None
    
    print(f"--- Starting training across {num_seeds} seeds ---")

    for seed in range(num_seeds):
        print(f"\n--- Training Seed {seed + 1}/{num_seeds} ---")
        set_global_seed(seed)
        
        # Train the policy
        q_values, policy = monte_carlo_off_policy_control(env, num_episodes, seed)
        
        # Evaluate the trained policy
        mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, policy)
        
        print(f"\nResults: Mean={mean_reward:.3f}, Min={min_reward:.3f}, "
              f"Max={max_reward:.3f}, Std={std_reward:.3f}, Success Rate={success_rate:.3f}")

        # Check if this is the best policy so far

    
    action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Stay'}
    
    print("\nBest Optimal Policy (Action to take in each state):")
    if best_policy is not None:
        policy_grid = np.array([action_map[i] for i in best_policy]).reshape(GRID_ROWS, GRID_COLS)
        print(policy_grid)
        
        print("\nQ-values for State 0 (top-left) from the best policy:")
        if 0 in best_q_values:
            for action, value in enumerate(best_q_values[0]):
                print(f"  Action: {action_map[action]}, Q-value: {value:.3f}")
        
        # Generate a GIF of the best policy
        generate_policy_gif(env, best_policy, filename='best_policy_run.gif')
    else:
        print("No successful policy was trained.")

    env.close()
