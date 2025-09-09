import csv
import numpy as np
import matplotlib.pyplot as plt
from utils import evaluate_policy
from env import FootballSkillsEnv


def bellman_backup(env, state, action, V, gamma, transition_calls):
    transitions = env.get_transitions_at_time(state, action)
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
            
            for state_index in range(self.n_states):
                state = self.env.index_to_state(state_index)
                v = self.V[state_index]
                q_values = [bellman_backup(self.env, state, action, self.V, self.gamma, self.transition_calls)
                            for action in range(self.env.action_space.n)]
            
                self.V[state_index] = max(q_values)
                self.policy[state_index] = np.argmax(q_values)
                delta = max(delta, abs(v - self.V[state_index]))

            if delta < self.theta:
                break
        
        return self.policy, self.V
    

class PolicyIteration:
    def __init__(self, env, gamma=0.95, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.observation_space[0].n * env.observation_space[1].n * env.observation_space[2].n
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros_like(self.V, dtype=int)
        self.transition_calls = [0]

    def policy_evaluation(self):
        while True:
            delta = 0
            
            for state_index in range(self.n_states):
                state = self.env.index_to_state(state_index)
                v = self.V[state_index]
                a = self.policy[state_index]
                new_v = bellman_backup(self.env, state, a, self.V, self.gamma, self.transition_calls)
                self.V[state_index] = new_v
                delta = max(delta, abs(v - new_v))

            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True

        for state_index in range(self.n_states):
            state = self.env.index_to_state(state_index)
            old_action = self.policy[state_index]
            q_values = [bellman_backup(self.env, state, action, self.V, self.gamma, self.transition_calls)
                        for action in range(self.env.action_space.n)]
            
            self.policy[state_index] = np.argmax(q_values)
            if old_action != self.policy[state_index]:
                policy_stable = False

        return policy_stable
    
    def run(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
            
        return self.policy, self.V
    

if __name__ == '__main__':
    gammas = [0.3, 0.5, 0.95]
    env_normal = FootballSkillsEnv(render_mode='gif')

    file = open('stationary_football_env.csv', 'a')
    writer = csv.writer(file)
    writer.writerow(['Gamma', 'VI Calls', 'PI Calls', 'VI Reward', 'PI Reward', 'Policy Identical'])

    for gamma in gammas:
        print(f'\nStationary environment with gamma={gamma}')

        vi = ValueIteration(env_normal, gamma)
        pi = PolicyIteration(env_normal, gamma)

        vi_policy, vi_value = vi.run()
        pi_policy, pi_value = pi.run()

        print(f'Value iteration transition calls: {vi.transition_calls[0]}')
        print(f'Policy iteration transition calls: {pi.transition_calls[0]}')

        vi_mean, vi_std = evaluate_policy(env_normal, vi_policy, gamma=gamma)
        pi_mean, pi_std = evaluate_policy(env_normal, pi_policy, gamma=gamma)

        print(f'Value iteration reward: {vi_mean} +/- {vi_std}')
        print(f'Policy iteration reward: {pi_mean} +/- {pi_std}')

        identical = np.all(vi_policy == pi_policy)
        print('Policies identical?', identical)

        writer.writerow([gamma, vi.transition_calls[0], pi.transition_calls[0], 
                         f'{vi_mean} +/- {vi_std}', f'{pi_mean} +/- {pi_std}', identical])

        if gamma == 0.95:
            env_normal.get_gif(vi_policy, filename=f'value_iteration_gamma={gamma}.gif')
            env_normal.get_gif(pi_policy, filename=f'policy_iteration_gamma={gamma}.gif')  

    file.close()      