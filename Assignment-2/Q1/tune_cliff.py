import os
import csv
import numpy as np
import optuna

# Import your existing environment and agent functions
from cliff import MultiGoalCliffWalkingEnv
from agent import SARSA, q_learning_for_cliff, expected_SARSA, environment_settings

# --- Create Directories for Saving Results ---
os.makedirs('tuning_optuna', exist_ok=True)
os.makedirs('tuning_optuna/plots', exist_ok=True)
LOG_FILE = None # Global variable for the CSV logger

# --- Objective Function for Optuna ---
def objective_cliff(trial, algorithm_func, env):
    """
    This function is called by Optuna for each trial.
    It suggests hyperparameters, runs the agent, and returns the performance score.
    """
    # 1. Suggest hyperparameters from the search space
    step_size = trial.suggest_float('step_size', 1e-3, 1.0, log=True)
    start_temperature = trial.suggest_float('start_temperature', 1.0, 10.0)
    temperature_decay = trial.suggest_float('temperature_decay', 0.99, 0.9999, log=True)

    # 2. Temporarily override global settings with the suggested values
    original_settings = environment_settings['MultiGoalCliffWalkingEnv'].copy()
    
    environment_settings['MultiGoalCliffWalkingEnv']['STEP_SIZE'] = step_size
    environment_settings['MultiGoalCliffWalkingEnv']['START_TEMPERATURE'] = start_temperature
    environment_settings['MultiGoalCliffWalkingEnv']['TEMPERATURE_DECAY'] = temperature_decay
    # Use fewer episodes/seeds for faster tuning
    environment_settings['MultiGoalCliffWalkingEnv']['NUM_EPISODES'] = 1000 
    environment_settings['MultiGoalCliffWalkingEnv']['NUM_SEEDS'] = 3

    # 3. Run the training
    _, avg_rewards, _, _ = algorithm_func(env)

    # 4. Calculate the performance score (objective value)
    # We want to maximize the average reward of the last 100 episodes.
    score = np.mean(avg_rewards[-100:])

    # 5. Restore original settings
    environment_settings['MultiGoalCliffWalkingEnv'] = original_settings
    
    # Pruning (optional): Stop unpromising trials early
    trial.report(score, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return score

# --- CSV Logging Callback ---
def csv_logger_callback(study, frozen_trial):
    """
    This function is called after each trial to log results to a CSV file.
    """
    params = frozen_trial.params
    score = frozen_trial.value
    
    # Create header if file doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(params.keys()) + ['score'])

    # Append new trial data
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(params.values()) + [score])


# --- Main Tuning Execution ---
if __name__ == '__main__':
    cliff_env = MultiGoalCliffWalkingEnv()

    algorithms_to_tune = {
        "SARSA": SARSA,
        "Q-Learning": q_learning_for_cliff,
        "Expected_SARSA": expected_SARSA
    }

    for name, func in algorithms_to_tune.items():
        print(f"\n--- Tuning {name} for Cliff Walking ---")

        # Set the log file for the current algorithm
        LOG_FILE = f'tuning_optuna/{name}_cliff_log.csv'
        if os.path.exists(LOG_FILE): # Clear old log
            os.remove(LOG_FILE)

        # Create a study object. We want to 'maximize' the score.
        study = optuna.create_study(direction='maximize')
        
        # Start the optimization
        study.optimize(
            lambda trial: objective_cliff(trial, algorithm_func=func, env=cliff_env),
            n_trials=50,  # Number of trials to run
            callbacks=[csv_logger_callback]
        )

        # --- Print Best Results ---
        print(f"\nBest score for {name}: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        # --- Generate and Save Visualizations ---
        # 1. Optimization History Plot
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f'tuning_optuna/plots/{name}_cliff_history.html')

        # 2. Parameter Importances Plot
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f'tuning_optuna/plots/{name}_cliff_importance.html')
        
        # 3. Slice Plot (shows each hyperparameter's effect)
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(f'tuning_optuna/plots/{name}_cliff_slice.html')

        print(f"Saved logs and plots for {name} in 'tuning_optuna/' directory.")

    cliff_env.close()