import csv
import numpy as np
from utils import evaluate_policy
from env import FootballSkillsEnv


def bellman_backup(env, state, action, V, gamma, transition_calls, time_step=None):
    transitions = env.get_transitions_at_time(state, action, time_step)
    transition_calls[0] += 1
    q_val = 0

    for prob, next_state in transitions:
        reward = env._get_reward(next_state[:2], action, state[:2])
        if next_state[2] == 1:  # terminal state
            q_val += prob * reward
        else:
            q_val += prob * (reward + gamma * V[env.state_to_index(next_state)])
    
    return q_val


class ValueIteration:
    def __init__(self, env, gamma=0.95, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.observation_space[0].n * env.observation_space[1].n * env.observation_space[2].n
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros_like(self.V, dtype=int)
        self.transition_calls = [0]

    def run(self):
        while True:
            delta = 0
            for s_idx in range(self.n_states):
                state = self.env.index_to_state(s_idx)
                v = self.V[s_idx]
                q_values = [bellman_backup(self.env, state, a, self.V, self.gamma, self.transition_calls)
                            for a in range(self.env.action_space.n)]
                self.V[s_idx] = max(q_values)
                self.policy[s_idx] = np.argmax(q_values)
                delta = max(delta, abs(v - self.V[s_idx]))
            if delta < self.theta:
                break
        return self.policy, self.V
    

class TimeDependentValueIteration:
    def __init__(self, env, horizon=40, gamma=0.95, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.horizon = horizon
        self.n_base_states = env.observation_space[0].n * env.observation_space[1].n * env.observation_space[2].n
        self.n_states = self.n_base_states * (horizon + 1)
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros_like(self.V, dtype=int)
        self.transition_calls = [0]

    def state_time_to_index(self, state_idx, t):
        return t * self.n_base_states + state_idx
    
    def index_to_state_time(self, idx):
        t = idx // self.n_base_states
        s_idx = idx % self.n_base_states
        return s_idx, t
    
    def run(self):
        while True:
            delta = 0
            for idx in range(self.n_states):
                s_idx, t = self.index_to_state_time(idx)
                if t >= self.horizon:
                    continue
                state = self.env.index_to_state(s_idx)
                v = self.V[idx]

                q_values = []
                for a in range(self.env.action_space.n):
                    transitions = self.env.get_transitions_at_time(state, a, t)
                    self.transition_calls[0] += 1
                    q_val = 0
                    for prob, next_state in transitions:
                        reward = self.env._get_reward(next_state[:2], a, state[:2])
                        if next_state[2] == 1:
                            q_val += prob * reward
                        else:
                            next_idx = self.env.state_to_index(next_state)
                            next_aug_idx = self.state_time_to_index(next_idx, t+1)
                            q_val += prob * (reward + self.gamma * self.V[next_aug_idx])
                    q_values.append(q_val)
                
                self.V[idx] = max(q_values)
                self.policy[idx] = np.argmax(q_values)
                delta = max(delta, abs(v - self.V[idx]))
            
            if delta < self.theta:
                break
        
        return self.policy, self.V
    

if __name__ == '__main__':
    env_degraded = FootballSkillsEnv(render_mode='gif', degrade_pitch=True)
    horizon = 40

    file = open('non_stationary_football_env.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(['Algo', '# Calls', 'Reward'])

    tdvi = TimeDependentValueIteration(env_degraded, horizon=horizon, gamma=0.95)
    tdvi_policy, tdvi_v = tdvi.run()
    tdvi_policy = tdvi_policy.reshape((horizon + 1, -1))
    print(f'Time-Dependent VI transition calls: {tdvi.transition_calls[0]}')

    vi_stationary = ValueIteration(env_degraded, gamma=0.95)
    vi_stationary_policy, vi_stationary_v = vi_stationary.run()
    vi_stationary_policy = np.array([vi_stationary_policy for _ in range(horizon)])
    print(f'Stationary VI transition calls: {vi_stationary.transition_calls[0]}')

    tdvi_mean, tdvi_std = evaluate_policy(env_degraded, tdvi_policy, episodes=20, gamma=0.95, is_time_dependent=True)
    vi_mean, vi_std = evaluate_policy(env_degraded, vi_stationary_policy, episodes=20, gamma=0.95, is_time_dependent=True)
    
    writer.writerow(['TDVI', tdvi.transition_calls[0], f'{tdvi_mean} +/- {tdvi_std}'])
    writer.writerow(['VI', vi_stationary.transition_calls[0], f'{vi_mean} +/- {vi_std}'])

    print(f'Time-Dependent VI reward: {tdvi_mean} +/- {tdvi_std}')
    print(f'Stationary VI reward: {vi_mean} +/- {vi_std}')

    env_degraded.get_gif(tdvi_policy, seed=10, filename='tdvi_policy.gif')
    env_degraded.get_gif(vi_stationary_policy, seed=10, filename='vi_stationary_policy.gif')

    print(np.array_equal(tdvi_policy, vi_stationary_policy))