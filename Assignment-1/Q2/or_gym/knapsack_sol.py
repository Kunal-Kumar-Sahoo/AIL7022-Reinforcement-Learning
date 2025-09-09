import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import matplotlib.pyplot as plt
from scipy.stats import mode


import numpy as np

class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4):
        pass


    def get_reward_and_done(self, current_weight, item_idx, action):
        pass

    def value_iteration(self, max_iterations=1000):
        pass

    def get_action(self, state):
        pass



class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4, eval_iterations=1000):
        pass

    def get_reward_and_done(self, current_weight, item_idx, action):
        pass
    
    def policy_evaluation(self):
        pass


    def policy_improvement(self):
        pass

    def run_policy_iteration(self, max_iterations=1000):
        pass

    def get_action(self, state):
        pass




if __name__=="__main__":
    env=OnlineKnapsackEnv()
    state=env._RESET()
    print(f"State for OnlineKnapsackEnv {state}")

    total_reward = 0
    done = False


    ###################### Random sampling #####################################################

    print("Starting Online Knapsack Simulation")
    print("Initial state:", state)

    # Run simulation until episode ends
    while not done:
        # Choose an action randomly: 0 (reject) or 1 (accept)
        action = env.sample_action()
        print(action)
        # Take a step in the environment
        next_state, reward, done, info = env._STEP(action)
        
        total_reward += reward
        print(f"Action: {'Accept' if action == 1 else 'Reject'} | Reward: {reward} | Next state: {next_state} | Done: {done}")
        exit()
    print(f"Episode finished. Total reward: {total_reward}")

