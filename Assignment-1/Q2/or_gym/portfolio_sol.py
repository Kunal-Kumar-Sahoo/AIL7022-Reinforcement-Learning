import numpy as np
import itertools
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import mode
import sys
import copy
import time
import random

random.seed(42)       # Set seed for Python's built-in random module
np.random.seed(42) 
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv



if __name__=="__main__":
    start_time=time.time()


    ###Part 1 and Part 2
    ####Please train the value and policy iteration training algo for the given three sequences of prices
    ####Config1
    env = DiscretePortfolioOptEnv(prices=[1, 3, 5, 5 , 4, 3, 2, 3, 5, 8])

    ####Config2
    env = DiscretePortfolioOptEnv(prices=[2, 2, 2, 4 ,2, 2, 4, 2, 2, 2])

    ####Config3
    env = DiscretePortfolioOptEnv(prices=[4, 1, 4, 1 ,4, 4, 4, 1, 1, 4])



    ####Run the evaluation on the following prices and save the plots.



    ###Part 3. (Portfolio Optimizaton)
    env = DiscretePortfolioOptEnv(variance=1)
