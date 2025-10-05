import imageio
import numpy as np
import matplotlib.pyplot as plt

from utils import moving_average


def save_reward_plot(rewards, title, filename, window=50):
    episodes = np.arange(len(rewards))
    ma = moving_average(rewards, window=window)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Episodic Reward', alpha=0.6)
    plt.plot(episodes, ma, label=f'Moving Average (w={window})')
    plt.xlabel('Episodes')
    plt.ylabel('Episodic Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_goal_bar_chart(q_stats, sarsa_stats, exp_stats, filename):
    labels = ['Q-Learning', 'SARSA', 'Expected SARSA']
    safe_values = [q_stats[0], sarsa_stats[0], exp_stats[0]]
    risky_values = [q_stats[1], sarsa_stats[1], exp_stats[1]]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(x - width/2, safe_values, width, label='Safe Goal visits')
    bars2 = plt.bar(x + width/2, risky_values, width, label='Risky Goal visits')

    for bar in bars1 + bars2:
        h = bar.get_height()
        plt.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
    
    plt.ylabel('Average number of goal visit (over training)')
    plt.title('Average Goal Visits by Algorithm (averaged across 10 seeds)')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def eval_policy_save_gif(env, Q, filename, num_episodes=100, max_steps=1000):
    rewards, frames = [], []

    for i in range(num_episodes):
        obs, _ = env.reset(seed=i)
        state = int(obs)
        total_r = 0.0

        for t in range(max_steps):
            action = int(np.argmax(Q[state]))
            next_obs, r, done, _, info = env.step(action)
            total_r += r
            state = int(next_obs)

            if i == 0:
                frame = env.render()
                frames.append(frame)
            if done:
                break
        rewards.append(total_r)

    if len(frames) > 0:
        frames_uint8 = [(frame.astype(np.uint8)) for frame in frames]
        imageio.mimsave(filename, frames_uint8, fps=4)
    
    return rewards