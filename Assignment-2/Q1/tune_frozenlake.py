import os
import csv
import numpy as np
import optuna

# Import your existing environment and agent functions
from frozenlake import DiagonalFrozenLake
from agent import monte_carlo, q_learning_for_frozenlake, environment_settings

# --- Create Directories ---
os.makedirs('tuning_optuna', exist_ok=True)
os.makedirs('tuning_optuna/plots', exist_ok=True)
LOG_FILE = None # Global variable for the CSV logger

# --- Objective Function ---
def objective_frozenlake(trial, algorithm_name, algorithm_func):
    """
    Objective function for FrozenLake agents.
    Suggests different parameters based on the algorithm name.
    """
    # 1. Suggest hyperparameters based on the algorithm
    if algorithm_name == "MonteCarlo":
        start_temperature = trial.suggest_float('start_temperature', 0.1, 5.0)
        temperature_decay = trial.suggest_float('temperature_decay', 0.999, 0.999999, log=True)
    else: # Q-Learning
        step_size = trial.suggest_float('step_size', 1e-3, 1.0, log=True)
        start_temperature = trial.suggest_float('start_temperature', 1.0, 10.0)
        temperature_decay = trial.suggest_float('temperature_decay', 0.999, 0.999999, log=True)

    # 2. Temporarily override global settings
    original_settings = environment_settings['DiagonalFrozenLake'].copy()
    
    settings = environment_settings['DiagonalFrozenLake']
    settings['START_TEMPERATURE'] = start_temperature
    settings['TEMPERATURE_DECAY'] = temperature_decay
    if 'step_size' in trial.params:
        settings['STEP_SIZE'] = trial.params['step_size']
    
    # Use fewer episodes for faster tuning
    settings['NUM_EPISODES'] = 50000 

    # 3. Run training
    env = DiagonalFrozenLake()
    _, avg_rewards = algorithm_func(env)
    env.close()

    # 4. Calculate score
    score = np.mean(avg_rewards[-1000:])

    # 5. Restore settings
    environment_settings['DiagonalFrozenLake'] = original_settings
    
    # Pruning (optional)
    trial.report(score, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return score

# --- CSV Logging Callback (can be reused) ---
def csv_logger_callback(study, frozen_trial):
    params = frozen_trial.params
    score = frozen_trial.value
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(params.keys()) + ['score'])
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(params.values()) + [score])

# --- Main Tuning Execution ---
if __name__ == '__main__':
    algorithms_to_tune = {
        "MonteCarlo": monte_carlo,
        "Q-Learning-FrozenLake": q_learning_for_frozenlake
    }

    for name, func in algorithms_to_tune.items():
        print(f"\n--- Tuning {name} for Frozen Lake ---")

        LOG_FILE = f'tuning_optuna/{name}_frozenlake_log.csv'
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)

        study = optuna.create_study(direction='maximize')
        
        study.optimize(
            lambda trial: objective_frozenlake(trial, algorithm_name=name, algorithm_func=func),
            n_trials=40,
            callbacks=[csv_logger_callback]
        )

        print(f"\nBest score for {name}: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")

        # --- Generate and Save Visualizations ---
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f'tuning_optuna/plots/{name}_frozenlake_history.html')

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f'tuning_optuna/plots/{name}_frozenlake_importance.html')
        
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(f'tuning_optuna/plots/{name}_frozenlake_slice.html')

        print(f"Saved logs and plots for {name} in 'tuning_optuna/' directory.")