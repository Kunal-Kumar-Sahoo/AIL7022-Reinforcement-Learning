import os
import numpy as np

epsilon = 1e-12  # Non-zero constant to avoid division by zero

def makedirs():
    os.makedirs('evaluation', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('gifs', exist_ok=True)

def moving_average(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode='same')

def softmax_action_probs(Q_row, temperature):
    z = Q_row - np.max(Q_row)
    exp = np.exp(z / (temperature + epsilon))
    probs = exp / (np.sum(exp) + epsilon)
    return probs

def sample_softmax_action(Q, state, temperature, rng):
    probs = softmax_action_probs(Q[state], temperature)
    return int(rng.choice(len(probs), p=probs)), probs

def run_episode(env, Q, update_fn, step_size, discount_factor,
                temperature, max_steps, rng):
    obs, _ = env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
    state = int(obs)
    total_reward = 0.0
    visited = {'safe': 0, 'risky': 0}

    # Initial behavior action
    action, _ = sample_softmax_action(Q, state, temperature, rng)

    for t in range(max_steps):
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = int(next_obs)
        total_reward += reward

        goal_str = info.get('goal', None)
        if goal_str == 'safe':
            visited['safe'] += 1
        elif goal_str == 'risky':
            visited['risky'] += 1

        Q, next_possible_action = update_fn(Q, state, action, reward, next_state, done, 
                                            step_size, discount_factor, temperature, rng)
        
        state = next_state
        if next_possible_action is None:
            action, _ = sample_softmax_action(Q, next_state, temperature, rng)
        else:
            action = next_possible_action

        if done:
            break

    return total_reward, t + 1, visited, Q        
