import numpy as np
from utils import sample_softmax_action, softmax_action_probs

def update_sarsa(Q, state, action, reward, next_state, done, 
                 step_size, discount_factor, temperature, rng):
    next_action, _ = sample_softmax_action(Q, next_state, temperature, rng)
    td_target = reward + discount_factor * Q[next_state, next_action] * (not done)
    td_error = td_target - Q[state, action]
    Q[state, action] += step_size * td_error
    return Q, next_action

def update_qlearning(Q, state, action, reward, next_state, done,
                     step_size, discount_factor, temperature, rng):
    td_target = reward + discount_factor * np.max(Q[next_state, :]) * (not done)
    td_error = td_target - Q[state, action]
    Q[state, action] += step_size * td_error
    return Q, None

def update_expected_sarsa(Q, state, action, reward, next_state, done,
                          step_size, discount_factor, temperature, rng):
    probs = softmax_action_probs(Q[next_state], temperature)
    expected_q = np.dot(probs, Q[next_state, :])
    td_target = reward + discount_factor * expected_q * (not done)
    td_error = td_target - Q[state, action]
    Q[state, action] += step_size * td_error
    return Q, None