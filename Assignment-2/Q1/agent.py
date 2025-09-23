import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import imageio
import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import pandas as pd 
from dataclasses import dataclass
from collections import defaultdict
from frozenlake import DiagonalFrozenLake
from cliff import MultiGoalCliffWalkingEnv

def SARSA(env):
    '''
    Implement the SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    pass

    return Q, episode_rewards, safe_visits, risky_visits

def q_learning(env):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    pass

    return Q, episode_rewards, safe_visits, risky_visits

def expected_SARSA(env):
    '''
    Implement the Expected SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    pass

    return Q,  episode_rewards, safe_visits, risky_visits


def monte_carlo(env):
    '''
    Implement the Monte Carlo algorithm to find the optimal policy for the given environment.
    Return Q table.
    return: Q table -> np.array of shape (num_states, num_actions)
    return: episode_rewards -> []
    return: _ 
    return: _ 
    '''
    pass

    return Q, episode_rewards, _, _