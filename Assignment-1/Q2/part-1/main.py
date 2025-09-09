import os
import sys
sys.path.append('/Users/kunalkumarsahoo/Playground/IITD/RL/Assignment-1/Q2/or_gym')

import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv


def bellman_backup(env, V, time_step, weight, index):
    q_vals = np.zeros(env.action_space.n)
    q_vals[0] = np.sum(env.item_probs * V[time_step + 1, weight, :])  # reject

    # Accept
    if weight + env.item_weights[index] <= env.max_weight:
        new_weight = weight + env.item_weights[index]
        q_vals[1] = env.item_values[index] + np.sum(env.item_probs * V[time_step + 1, new_weight, :]) 
    
    else:
        q_vals[1] = 0

    return q_vals


class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = env.N
        self.max_weight = env.max_weight
        self.T = env.step_limit
        self.V = np.zeros((self.T + 1, self.max_weight + 1, self.N))
        self.policy = np.zeros_like(self.V, dtype=int)

    def run(self, max_iters=1000):
        start = time.time()
        for _ in range(max_iters):
            delta = 0
            for time_step in reversed(range(self.T)):
                for weight in range(self.max_weight + 1):
                    for index in range(self.N):
                        q_vals = bellman_backup(self.env, self.V, time_step, weight, index)
                        new_val = np.max(q_vals)
                        delta = max(delta, abs(new_val - self.V[time_step, weight, index]))
                        self.V[time_step, weight, index] = new_val
                        self.policy[time_step, weight, index] = np.argmax(q_vals)
            
            if delta < self.epsilon:
                break
        
        print(f'Value Iteration - Horizon: {self.T} Time: {time.time() - start}s')
        return self.V, self.policy
    

class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-6):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.N = env.N
        self.max_weight = env.max_weight
        self.T = env.step_limit
        self.V = np.zeros((self.T + 1, self.max_weight + 1, self.N))
        self.policy = np.zeros_like(self.V, dtype=int)

    def policy_evaluation(self):
        while True:
            delta = 0
            for time_step in reversed(range(self.T)):
                for weight in range(self.max_weight + 1):
                    for index in range(self.N):
                        action = self.policy[time_step, weight, index]
                        q_vals = bellman_backup(self.env, self.V, time_step, weight, index)
                        v_new = q_vals[action]
                        delta = max(delta, abs(v_new - self.V[time_step, weight, index]))
                        self.V[time_step, weight, index] = v_new
            
            if delta < self.epsilon:
                break

    def policy_improvement(self):
        stable = True
        
        for time_step in reversed(range(self.T)):
            for weight in range(self.max_weight +1):
                for index in range(self.N):
                    old_action = self.policy[time_step, weight, index]
                    q_vals = bellman_backup(self.env, self.V, time_step, weight, index)
                    best_action = np.argmax(q_vals)
                    self.policy[time_step, weight, index] = best_action
                    if best_action != old_action:
                        stable = False
        
        return stable
    
    def run(self, max_iters=1000):
        start = time.time()
        for _ in range(max_iters):
            self.policy_evaluation()
            stable = self.policy_improvement()
            
            if stable:
                break
        
        print(f'Policy Iteration - Horizon: {self.T} Time: {time.time() - start}s')
        return self.V, self.policy
    
def evaluate_policy(env, policy, algorithm, horizon, num_episodes=5):
    trajectories = []
    final_values = []

    for seed in range(num_episodes):
        env.set_seed(seed)
        state = env.reset()

        current_weight = state['state'][0]
        current_item = state['state'][1]
        total_value = 0
        values = []

        for time_step in range(env.step_limit):
            action = policy[time_step, current_weight, current_item]
            next_state, reward, done, _ = env.step(action)
            total_value += reward
            values.append(total_value)

            current_weight = next_state['state'][0]
            current_item = next_state['state'][1]
            state = next_state

            if done:
                break
        
        trajectories.append(values)
        final_values.append(total_value)

    print('Final knapsack values across seeds:', final_values)

    plt.figure(figsize=(10, 6))
    for seed, trajectory in enumerate(trajectories):
        plt.plot(trajectory, label=f'Seed {seed}')
    
    plt.xlabel('Item presentation')
    plt.ylabel('Cumulative Knapsack Value')
    plt.title(f'Knapsack Value Trajectories for Different Seeds ({algorithm})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output_seeds', f'cumulative_values_{algorithm}_{horizon}.png'))


def plot_value_heatmaps(env, V, base_filename='value_heatmap'):
    frames_dict = {
        "weight": [],
        "value": [],
        "ratio": []
    }

    # --- Sorting indices ---
    sorted_by_weight = np.argsort(env.item_weights)
    sorted_by_value = np.argsort(env.item_values)
    ratios = env.item_weights / (env.item_values + 1e-6)
    sorted_by_ratio = np.argsort(ratios)

    sortings = {
        "weight": (sorted_by_weight, [f"{env.item_weights[i]}" for i in sorted_by_weight]),
        "value": (sorted_by_value, [f"{env.item_values[i]}" for i in sorted_by_value]),
        "ratio": (sorted_by_ratio, [f"{ratios[i]:.2f}" for i in sorted_by_ratio])
    }

    os.makedirs('frames', exist_ok=True)
    os.makedirs('output_seeds', exist_ok=True)

    global_min = np.min(V[:-1, :, :])
    global_max = np.max(V[:-1, :, :])

    for time_step in range(env.step_limit):
        slice_V = V[time_step, :, :]

        for key, (indices, labels) in sortings.items():
            slice_sorted = slice_V[:, indices]

            plt.figure(figsize=(16, 8))
            image = plt.imshow(slice_sorted, aspect='auto', origin='lower',
                               vmin=global_min, vmax=global_max)
            cbar = plt.colorbar(image, label='Value')
            cbar.set_label('Value')
            plt.xlabel('Items')
            plt.ylabel('Knapsack Weight')

            if key == "weight":
                title = f'Optimal Value Function (Weight-Sorted) at Time {time_step}'
            elif key == "value":
                title = f'Optimal Value Function (Value-Sorted) at Time {time_step}'
            else:
                title = f'Optimal Value Function (Weight/Value Ratio-Sorted) at Time {time_step}'
            plt.title(title)

            plt.xticks(ticks=np.arange(len(indices)), labels=labels,
                       rotation=90, fontsize=6)
            plt.tight_layout()

            frame_name = os.path.join('frames', f'{key}_frame_{time_step}.png')
            plt.savefig(frame_name)
            plt.close()

            frames_dict[key].append(imageio.imread(frame_name))

    # --- Save GIFs ---
    for key in frames_dict:
        gif_path = os.path.join('output_seeds', f'{base_filename}_{key}.gif')
        imageio.mimsave(gif_path, frames_dict[key], fps=1)
        print(f"Saved {gif_path}")



if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    log_file = os.path.join('logs', 'knapsack_results.csv')

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'algorithm', 'horizon', 'seed',
            'final_value', 'trajectory',
            'mean_final_value', 'std_final_value',
            'mean_trajectory', 'std_trajectory'
        ])

        horizon = 50
        env = OnlineKnapsackEnv()
        env.step_limit = horizon
        print(f'\nState space: {(env.step_limit, env.max_weight, env.N)}')

        vi = ValueIterationOnlineKnapsack(env)
        pi = PolicyIterationOnlineKnapsack(env)

        v_vi, pi_vi = vi.run()
        v_pi, pi_pi = pi.run()

        for algo, policy, value_fn in [('VI', pi_vi, v_vi), ('PI', pi_pi, v_pi)]:
            trajectories = []
            final_values = []

            for seed in range(5):
                env.set_seed(seed)
                state = env.reset()
                current_weight = state['state'][0]
                current_item = state['state'][1]
                total_value = 0
                values = []

                for time_step in range(env.step_limit):
                    action = policy[time_step, current_weight, current_item]
                    next_state, reward, done, _ = env.step(action)
                    total_value += reward
                    values.append(total_value)

                    current_weight = next_state['state'][0]
                    current_item = next_state['state'][1]
                    state = next_state

                    if done:
                        break

                trajectories.append(values)
                final_values.append(total_value)

                writer.writerow([algo, horizon, seed, total_value, values,
                                 '', '', '', ''])

            mean_final = float(np.mean(final_values))
            std_final = float(np.std(final_values))

            max_len = max(len(traj) for traj in trajectories)
            padded = np.array([traj + [traj[-1]] * (max_len - len(traj))
                               for traj in trajectories])
            mean_traj = np.mean(padded, axis=0).tolist()
            std_traj = np.std(padded, axis=0).tolist()

            writer.writerow([algo, horizon, 'ALL', '',
                             '', mean_final, std_final,
                             mean_traj, std_traj])

            print(f"\n=== {algo} (horizon={horizon}) ===")
            print("Final knapsack values across seeds:", final_values)
            print("Mean:", mean_final, "Std:", std_final)

            plt.figure(figsize=(10, 6))
            for seed, trajectory in enumerate(trajectories):
                plt.plot(trajectory, label=f'Seed {seed}')
            plt.plot(mean_traj, label="Mean", linewidth=2, color="black")
            plt.fill_between(range(len(mean_traj)),
                             np.array(mean_traj) - np.array(std_traj),
                             np.array(mean_traj) + np.array(std_traj),
                             alpha=0.3, label="Std Dev")
            plt.xlabel('Item presentation')
            plt.ylabel('Cumulative Knapsack Value')
            plt.title(f'Knapsack Value Trajectories ({algo}, Horizon={horizon})')
            plt.legend()
            plt.tight_layout()
            os.makedirs('output_seeds', exist_ok=True)
            plt.savefig(os.path.join('output_seeds', f'cumulative_values_{algo}_{horizon}.png'))
            plt.close()

            plot_value_heatmaps(env, value_fn, f'{algo.lower()}_heatmap_horizon_{horizon}')

        for horizon in [10, 50, 500]:
            env = OnlineKnapsackEnv()
            env.step_limit = horizon
            print(f'\nRunning Value Iteration with horizon {horizon}')

            vi = ValueIterationOnlineKnapsack(env)
            v_vi, pi_vi = vi.run()

            env.set_seed(0)
            state = env.reset()
            current_weight = state['state'][0]
            current_item = state['state'][1]
            total_value = 0
            values = []

            for time_step in range(env.step_limit):
                action = pi_vi[time_step, current_weight, current_item]
                next_state, reward, done, _ = env.step(action)
                total_value += reward
                values.append(total_value)

                current_weight = next_state['state'][0]
                current_item = next_state['state'][1]
                state = next_state

                if done:
                    break

            writer.writerow(['VI', horizon, 0, total_value, values,
                             '', '', '', ''])

            plot_value_heatmaps(env, v_vi, f'value_iteration_heatmap_horizon_{horizon}')
