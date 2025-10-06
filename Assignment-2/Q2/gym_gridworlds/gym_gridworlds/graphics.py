import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_policy_gif(env, policy, filename, seed=42):
    frames = []
    print(f'Generating GIF: {filename}')

    random_prob = getattr(getattr(env, 'unwrapped', env), 'random_action_prob', 0.0)
    env_render = gym.make(env.spec.id, render_mode='rgb_array', random_action_prob=random_prob)

    state, _ = env_render.reset(seed=seed)
    done = False

    while not done:
        frames.append(env_render.render())
        action = policy[state]
        state, _, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated

    frames.append(env_render.render())
    env_render.close()

    imageio.mimsave(filename, frames, fps=3)
    print('GIF saved successfully.')

def plot_reward_curve(reward_histories, noise_level, algorithm_name, plots_dir='plots'):
    print(f'\nGenerating plot for {algorithm_name} with noise level: {noise_level}')
    
    if not reward_histories or not reward_histories[0]:
        print('Warning: No reward history to plot.')
        return
    
    min_len = min(len(h) for h in reward_histories)
    rewards_matrix = np.array([h[:min_len] for h in reward_histories])

    mean_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)

    episodes = np.arange(0, min_len * 200, 200)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, mean_rewards, label=f'Mean Reward (Noise: {noise_level})')
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label='Standard Deviation')
    plt.title(f'{algorithm_name} Learning Curve (Noise: {noise_level})')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Evaluation Reward')
    plt.grid(True)
    plt.legend()

    filename = f'{plots_dir}/{algorithm_name.lower().replace(" ", "_")}_reward_curve_{noise_level}.png'
    plt.savefig(filename)
    plt.close()

    print(f'Plot saved successfully to {filename}.')