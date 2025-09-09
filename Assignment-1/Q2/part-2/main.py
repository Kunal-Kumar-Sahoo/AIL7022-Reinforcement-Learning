import sys
sys.path.append('/Users/kunalkumarsahoo/Playground/IITD/RL/Assignment-1/Q2/or_gym')

import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

os.makedirs("plots/training", exist_ok=True)
os.makedirs("plots/trajectories", exist_ok=True)
os.makedirs("plots/variance", exist_ok=True)
os.makedirs("results", exist_ok=True)


def next_state(env, state, action, Cmax, Hmax):
    t, cash, h = state
    price = int(env.asset_prices[0, t])  # realized price
    nc, nh = cash, h

    # --- Apply action ---
    if action < 0:  # Sell
        sell_amt = min(-action, h)
        nh -= sell_amt
        nc += (price - env.sell_cost[0]) * sell_amt

    elif action > 0:  # Buy
        buy_amt = action
        if h + buy_amt > env.holding_limit[0]:
            buy_amt = max(0, env.holding_limit[0] - h)
        purchase_cost = (price + env.buy_cost[0]) * buy_amt
        if purchase_cost > nc:
            buy_amt = int(nc // (price + env.buy_cost[0]))
            purchase_cost = (price + env.buy_cost[0]) * buy_amt
        nh += buy_amt
        nc -= purchase_cost

    # Clamp both cash and holdings
    nc = max(0, min(int(nc), Cmax))
    nh = max(0, min(int(nh), Hmax))

    next_t = t + 1
    done = next_t == env.step_limit

    reward = 0
    if done:
        final_price = int(env.asset_prices[0, t])
        reward = nc + nh * final_price

    return (next_t, nc, nh), reward, done


def value_iteration(env, gamma=1.0, epsilon=1e-6):
    Tmax = env.step_limit
    Cmax = env.initial_cash + (int(env.asset_prices.max())+1) * env.holding_limit[0]
    Hmax = env.holding_limit[0]

    V = np.zeros((Tmax+1, Cmax+1, Hmax+1))
    policy = np.zeros((Tmax+1, Cmax+1, Hmax+1), dtype=int)
    training_curve = []

    while True:
        delta = 0
        for t in range(Tmax-1, -1, -1):
            for c in range(Cmax+1):
                for h in range(Hmax+1):
                    best_val, best_a = -1e9, 0
                    for a in range(-2, 3):  
                        (t2, c2, h2), r, done = next_state(env, (t,c,h), a, Cmax, Hmax)
                        val = r if done else r + gamma * V[t2, c2, h2]
                        if val > best_val:
                            best_val, best_a = val, a
                    delta = max(delta, abs(best_val - V[t,c,h]))
                    V[t,c,h] = best_val
                    policy[t,c,h] = best_a
        training_curve.append(V[0, env.initial_cash, 0])
        if delta < epsilon:
            break
    return V, policy, training_curve


def policy_iteration(env, gamma=1.0, max_iter=1000, epsilon=1e-6):
    Tmax = env.step_limit
    Cmax = env.initial_cash + (int(env.asset_prices.max())+1) * env.holding_limit[0]
    Hmax = env.holding_limit[0]

    V = np.zeros((Tmax+1, Cmax+1, Hmax+1))
    policy = np.zeros((Tmax+1, Cmax+1, Hmax+1), dtype=int)
    training_curve = []

    for it in range(max_iter):
        while True:
            delta = 0
            for t in range(Tmax-1, -1, -1):
                for c in range(Cmax+1):
                    for h in range(Hmax+1):
                        a = policy[t,c,h]
                        (t2, c2, h2), r, done = next_state(env, (t,c,h), a, Cmax, Hmax)
                        val = r if done else r + gamma*V[t2,c2,h2]
                        delta = max(delta, abs(val - V[t,c,h]))
                        V[t,c,h] = val
            if delta < epsilon:
                break
        training_curve.append(V[0, env.initial_cash, 0])

        stable = True
        for t in range(Tmax):
            for c in range(Cmax+1):
                for h in range(Hmax+1):
                    old_a = policy[t,c,h]
                    best_val, best_a = -1e9, old_a
                    for a in range(-2,3):
                        (t2, c2, h2), r, done = next_state(env, (t,c,h), a, Cmax, Hmax)
                        val = r if done else r + gamma*V[t2,c2,h2]
                        if val > best_val:
                            best_val, best_a = val, a
                    policy[t,c,h] = best_a
                    if best_a != old_a:
                        stable = False
        if stable:
            break
    return V, policy, training_curve


def simulate_policy(env, policy):
    Tmax = env.step_limit
    Cmax = env.initial_cash + (int(env.asset_prices.max())+1) * env.holding_limit[0]
    Hmax = env.holding_limit[0]
    state = (0, env.initial_cash, 0)
    wealth_traj, cash_traj, h_traj, actions = [], [], [], []
    while state[0] < Tmax:
        a = policy[state]
        actions.append(a)
        next_s, r, done = next_state(env, state, a, Cmax, Hmax)
        price = int(env.asset_prices[0, min(state[0],Tmax-1)])
        wealth = next_s[1] + next_s[2]*price
        wealth_traj.append(wealth)
        cash_traj.append(next_s[1])
        h_traj.append(next_s[2])
        state = next_s
        if done: break
    return wealth_traj, cash_traj, h_traj, actions


def policy_iteration_with_tracking(env, gamma=1.0, max_iter=1000, epsilon=1e-2):
    Tmax = env.step_limit
    Cmax = env.initial_cash + (int(env.asset_prices.max())+1) * env.holding_limit[0]
    Hmax = env.holding_limit[0]

    V = np.zeros((Tmax+1, Cmax+1, Hmax+1))
    policy = np.zeros((Tmax+1, Cmax+1, Hmax+1), dtype=int)
    diffs = []

    for it in range(max_iter):
        # Policy Evaluation
        while True:
            delta = 0
            for t in range(Tmax-1, -1, -1):
                for c in range(Cmax+1):
                    for h in range(Hmax+1):
                        a = policy[t,c,h]
                        (t2, c2, h2), r, done = next_state(env, (t,c,h), a, Cmax, Hmax)
                        val = r if done else r + gamma*V[t2,c2,h2]
                        delta = max(delta, abs(val - V[t,c,h]))
                        V[t,c,h] = val
            if delta < epsilon:
                break
        diffs.append(delta)

        # Policy Improvement
        stable = True
        for t in range(Tmax):
            for c in range(Cmax+1):
                for h in range(Hmax+1):
                    old_a = policy[t,c,h]
                    best_val, best_a = -1e9, old_a
                    for a in range(-2,3):
                        (t2, c2, h2), r, done = next_state(env, (t,c,h), a, Cmax, Hmax)
                        val = r if done else r + gamma*V[t2,c2,h2]
                        if val > best_val:
                            best_val, best_a = val, a
                    policy[t,c,h] = best_a
                    if best_a != old_a:
                        stable = False
        if stable:
            break
    return V, policy, diffs


if __name__=="__main__":
    log_file = "results/portfolio_results.csv"
    os.makedirs("results", exist_ok=True)

    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "SeqID", "Gamma", "Method", "FinalWealth", "ExecTime(s)",
            "TrainingCurve", "WealthTrajectory", "CashTrajectory",
            "HoldingsTrajectory", "MaxValueDiffs", "Variance", "Converged"
        ])

        # ======================
        # 1 & 2: Value Iteration & Policy Iteration
        # ======================
        price_seqs = [
            [1,3,5,5,4,3,2,3,5,8],
            [2,2,2,4,2,2,4,2,2,2],
            [4,1,4,1,4,4,4,1,1,4],
        ]

        for seq_id, seq in enumerate(price_seqs):
            env = DiscretePortfolioOptEnv(prices=seq)
            for gamma in [0.999, 1.0]:
                
                # ---------- Value Iteration ----------
                start = time.time()
                V_vi, pi_vi, curve_vi = value_iteration(env, gamma)
                t_vi = time.time()-start
                wealth_vi, cash_vi, h_vi, acts_vi = simulate_policy(env, pi_vi)

                writer.writerow([
                    seq_id, gamma, "VI", wealth_vi[-1], t_vi,
                    curve_vi, wealth_vi, cash_vi, h_vi, "", "", ""
                ])

                print(f"[Seq {seq_id}, γ={gamma}] VI final wealth={wealth_vi[-1]} time={t_vi:.2f}s")

                plt.plot(curve_vi)
                plt.title(f"Training Curve (VI, Seq {seq_id}, γ={gamma})")
                plt.xlabel("Iteration")
                plt.ylabel("Value of initial state")
                plt.savefig(f"plots/training/VI_training_seq{seq_id}_gamma{gamma}.png")
                plt.clf()

                plt.plot(wealth_vi,label="Wealth")
                plt.plot(cash_vi,label="Cash")
                plt.plot(h_vi,label="Holdings")
                plt.title(f"Trajectory (VI, Seq {seq_id}, γ={gamma})")
                plt.xlabel("Time step")
                plt.legend()
                plt.savefig(f"plots/trajectories/VI_trajectory_seq{seq_id}_gamma{gamma}.png")
                plt.clf()

                # ---------- Policy Iteration ----------
                start = time.time()
                V_pi, pi_pi, curve_pi = policy_iteration(env, gamma)
                t_pi = time.time()-start
                wealth_pi, cash_pi, h_pi, acts_pi = simulate_policy(env, pi_pi)

                writer.writerow([
                    seq_id, gamma, "PI", wealth_pi[-1], t_pi,
                    curve_pi, wealth_pi, cash_pi, h_pi, "", "", ""
                ])

                print(f"[Seq {seq_id}, γ={gamma}] PI final wealth={wealth_pi[-1]} time={t_pi:.2f}s")

                plt.plot(curve_pi)
                plt.title(f"Training Curve (PI, Seq {seq_id}, γ={gamma})")
                plt.xlabel("Iteration")
                plt.ylabel("Value of initial state")
                plt.savefig(f"plots/training/PI_training_seq{seq_id}_gamma{gamma}.png")
                plt.clf()

                plt.plot(wealth_pi,label="Wealth")
                plt.plot(cash_pi,label="Cash")
                plt.plot(h_pi,label="Holdings")
                plt.title(f"Trajectory (PI, Seq {seq_id}, γ={gamma})")
                plt.xlabel("Time step")
                plt.legend()
                plt.savefig(f"plots/trajectories/PI_trajectory_seq{seq_id}_gamma{gamma}.png")
                plt.clf()

        # ======================
        # 3: Variance Experiment
        # ======================
        env_var = DiscretePortfolioOptEnv(variance=1.0)
        V_var, pi_var, diffs = policy_iteration_with_tracking(env_var, gamma=1.0, max_iter=1000, epsilon=1e-2)

        converged = diffs[-1] < 1e-2

        # Run multiple episodes to get mean ± std wealth
        episodes = 50
        all_wealth = []
        for ep in range(episodes):
            env_var.reset()
            wealth_traj, _, _, _ = simulate_policy(env_var, pi_var)
            all_wealth.append(wealth_traj)
        all_wealth = np.array(all_wealth)
        mean_wealth = all_wealth.mean(axis=0)
        std_wealth = all_wealth.std(axis=0)

        writer.writerow([
            "Variance=1.0", 1.0, "PI", "", "", "",
            mean_wealth.tolist(), "", "", diffs, 1.0, converged
        ])

        print(f"Variance=1.0 PI converged: {converged}")

        plt.plot(diffs)
        plt.title("Max Value Difference vs Iterations (Variance=1.0, PI)")
        plt.xlabel("Iteration")
        plt.ylabel("Max Value Difference")
        plt.savefig("plots/variance/convergence_variance.png")
        plt.clf()

        timesteps = np.arange(len(mean_wealth))
        plt.plot(timesteps, mean_wealth, label="Mean Wealth", color="blue")
        plt.fill_between(timesteps, mean_wealth-std_wealth, mean_wealth+std_wealth,
                         color="blue", alpha=0.3, label="±1 Std Dev")
        plt.title("Wealth Evolution under Variance=1.0 (Policy Iteration)")
        plt.xlabel("Time step")
        plt.ylabel("Wealth")
        plt.legend()
        plt.savefig("plots/variance/wealth_variance.png")
        plt.clf()

    print(f"All results logged to {log_file}")
