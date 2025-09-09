import csv
import numpy as np
from env import FootballSkillsEnv

def evaluate_policy(env, policy, episodes=20, gamma=0.95, is_time_dependent=False):
    rewards = []
    for seed in range(episodes):
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0
        t = 0
        
        while not done:
            state_index = env.state_to_index(state)
            if is_time_dependent:
                action = policy[t][state_index]
            else:
                action = policy[state_index]
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            t += 1
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)


def bellman_backup(env, state, action, V, gamma, transition_calls):
    transitions = env.get_transitions_at_time(state, action)
    transition_calls[0] +=1 
    q_val = 0

    for prob, next_state in transitions:
        reward = env._get_reward(next_state[:2], action, state[:2])
        if next_state[2] == 1:
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
        

class PrioritizedValueIteration:
    def __init__(self, env, gamma=0.95, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.observation_space[0].n * env.observation_space[1].n * env.observation_space[2].n
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros_like(self.V, dtype=int)
        self.transition_calls = [0]

    def run(self):
        priorities = np.ones(self.n_states)
        while np.max(priorities) > self.theta:
            s_idx = np.argmax(priorities)
            state = self.env.index_to_state(s_idx)
            v = self.V[s_idx]
            q_values = [bellman_backup(self.env, state, a, self.V, self.gamma, self.transition_calls)
                        for a in range(self.env.action_space.n)]
            self.V[s_idx] = max(q_values)
            self.policy[s_idx] = np.argmax(q_values)
            priorities[s_idx] = abs(v - self.V[s_idx])
        return self.policy, self.V        


if __name__ == '__main__':
    env_normal = FootballSkillsEnv(render_mode='gif')
    gamma = 0.95

    file = open('modified_vi_expt.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(['Algo', '# Calls', 'Reward'])

    pvi = PrioritizedValueIteration(env_normal, gamma)
    vi = ValueIteration(env_normal, gamma)

    pvi_policy, pvi_v = pvi.run()
    vi_policy, vi_v = vi.run()

    print(f'Prioritized VI transition calls: {pvi.transition_calls[0]}')
    print(f'Standard VI transition calls: {vi.transition_calls[0]}')

    pvi_mean, pvi_std = evaluate_policy(env_normal, pvi_policy, episodes=20, gamma=0.95)
    vi_mean, vi_std = evaluate_policy(env_normal, vi_policy, episodes=20, gamma=0.95)

    print(f'Prioritized VI reward: {pvi_mean} +/- {pvi_std}')
    print(f'Standard VI reward: {vi_mean} +/- {vi_std}')

    writer.writerows([['PVI', pvi.transition_calls[0], f'{pvi_mean} +/- {pvi_std}'],
                      ['VI', vi.transition_calls[0], f'{vi_mean} +/- {vi_std}']])

    env_normal.get_gif(pvi_policy, filename='pvi.gif')