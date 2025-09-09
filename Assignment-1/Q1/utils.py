import numpy as np

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