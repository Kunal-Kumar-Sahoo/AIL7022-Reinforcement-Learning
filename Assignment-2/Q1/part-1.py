import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import json
from agent import q_learning_for_cliff as q_learning, SARSA, expected_SARSA
from cliff import MultiGoalCliffWalkingEnv
from utils import makedirs
from graphics import save_reward_plot, save_goal_bar_chart, eval_policy_save_gif


def evaluate_and_save_json(env, Qs):
    results = {}
    for name, Q in Qs.items():
        rewards = []
        for i in range(100):
            obs, _ = env.reset(seed=i)
            state = int(obs)
            total_r = 0.0
            for t in range(1000):
                action = int(np.argmax(Q[state]))
                next_obs, r, done, _, info = env.step(action)
                total_r += r
                state = int(next_obs)
                if done:
                    break
            rewards.append(total_r)
        rewards = np.array(rewards)
        results[name] = {'mean': float(np.mean(rewards)), 'std': float(np.std(rewards))}

    with open('evaluation/cliff_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results


def main():
    env = MultiGoalCliffWalkingEnv(render_mode='rgb_array')
    print('Training Q-learning...')
    q_Q, q_rewards, q_safe, q_risky = q_learning(env)

    print('Training SARSA...')
    sarsa_Q, sarsa_rewards, sarsa_safe, sarsa_risky = SARSA(env)

    print('Training Expected SARSA...')
    exp_Q, exp_rewards, exp_safe, exp_risky = expected_SARSA(env)

    makedirs()
    save_reward_plot(q_rewards, 'Q-Learning', 'plots/qlearning_plot.png')
    save_reward_plot(sarsa_rewards, 'SARSA', 'plots/sarsa_plot.png')
    save_reward_plot(exp_rewards, 'Expected SARSA', 'plots/expected_sarsa_plot.png')

    save_goal_bar_chart((q_safe, q_risky), (sarsa_safe, sarsa_risky), (exp_safe, exp_risky), 'plots/average_goal_visits.png')

    print('Generating GIFs and performing evaluations...')

    Qs = {
        'qlearning': q_Q,
        'sarsa': sarsa_Q,
        'expected_sarsa': exp_Q
    }

    eval_rewards_q = eval_policy_save_gif(env, q_Q, 'gifs/qlearning.gif', num_episodes=100)
    eval_rewards_sarsa = eval_policy_save_gif(env, sarsa_Q, 'gifs/sarsa.gif', num_episodes=100)
    eval_rewards_exp = eval_policy_save_gif(env, exp_Q, 'gifs/expected_sarsa.gif', num_episodes=100)

    results = evaluate_and_save_json(env, Qs)
    print('Evaluation results saved to `evaluation/cliff_evaluation_results.json`')
    print(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()