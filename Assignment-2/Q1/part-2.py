import os
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt
from agent import monte_carlo, q_learning_for_frozenlake as q_learning
from frozenlake import DiagonalFrozenLake

from utils import makedirs
from graphics import save_reward_plot, eval_policy_save_gif


START_STATES = [(0, 3), (0, 5)]
NUM_EVAL_EPISODES = 100


def run_for_start(start_state):
    results_for_start = {}
    print(f'\nStart state {start_state}')

    env = DiagonalFrozenLake(render_mode='rgb_array', map_size=16, start_state=start_state)

    print('Training Monte Carlo (on-policy) ...')
    Q_mc, rewards_mc = monte_carlo(env)

    title = f'Monte Carlo On-policy - start {start_state}'
    plot_name = f'plots/frozen_lake_mc_{start_state}.png'
    save_reward_plot(rewards_mc, title, plot_name)

    gif_name = f'gifs/frozenlake_mc_{start_state}.gif'
    mc_eval_rewards = eval_policy_save_gif(env, Q_mc, gif_name, num_episodes=NUM_EVAL_EPISODES, max_steps=100)
    mc_eval_rewards = np.array(mc_eval_rewards)
    results_for_start['mc'] = {'mean': float(np.mean(mc_eval_rewards)), 'std': float(np.std(mc_eval_rewards))}
    print(f'MC eval mean: {results_for_start["mc"]["mean"]:.4f}, std: {results_for_start["mc"]["std"]:.4f}')
    print(f'Saved MC gif to {gif_name}')

    print('Training Q-Learning ...')
    Q_q, rewards_q = q_learning(env)
    rewards_q = np.array(rewards_q)

    title = f'Q-Learning - start {start_state}'
    plot_name_q = f'plots/frozenlake_qlearning_{start_state}.png'
    save_reward_plot(rewards_q, title, plot_name_q)
    print(f'Saved Q-learning reward plot to {plot_name_q}')

    gif_name_q = f'gifs/frozenlake_qlearning_{start_state}.gif'
    q_eval_rewards = eval_policy_save_gif(env, Q_q, gif_name_q, num_episodes=NUM_EVAL_EPISODES, max_steps=100)
    q_eval_rewards = np.array(q_eval_rewards)
    results_for_start['qlearning'] = {'mean': float(np.mean(q_eval_rewards)), 'std': float(np.std(q_eval_rewards))}
    print(f'Q eval mean: {results_for_start["qlearning"]["mean"]:.4f}, std: {results_for_start["qlearning"]["std"]:.4f}')
    print(f'Saved Q-learning gif to {gif_name_q}')

    env.close()
    return results_for_start


def main():
    makedirs()

    final_results = {}
    for start in START_STATES:
        res = run_for_start(start)
        final_results[str(start)] = res

    json_path = 'evaluation/frozenlake_variant_evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f'\nSaved evaluation results to {json_path}')
    print(json.dumps(final_results, indent=4))


if __name__ == '__main__':
    main()