import argparse
import yaml
import os
import gymnasium as gym
import numpy as np
import imageio
import torch as torch  # Required for evaluating 'torch.nn.Tanh' in config
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

def parse_policy_kwargs(kwargs_str):
    """
    Parses the string representation of the dictionary from YAML.
    Example input: "dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[256, 256], vf=[256, 256]))"
    """
    if not kwargs_str:
        return {}
    # We allow the use of 'torch' in the eval string so the config can reference torch.nn.Tanh
    return eval(kwargs_str, {"dict": dict, "torch": torch})

def train_and_evaluate(env_id, algo_name, hyperparams_path, seeds=[1, 2, 3]):
    # 1. Load Configuration
    with open(hyperparams_path, 'r') as f:
        config = yaml.safe_load(f)

    if algo_name not in config or env_id not in config[algo_name]:
        raise ValueError(f"Configuration for {algo_name} on {env_id} not found in {hyperparams_path}")

    params = config[algo_name][env_id]
    
    # 2. Extract Training Parameters
    # We default n_envs to 1 if not specified in config (Important for PPO vs A2C difference)
    n_envs = params.pop('n_envs', 1) 
    total_timesteps = params.pop('total_timesteps')
    policy_type = params.pop('policy_type')
    
    # Parse policy_kwargs string into a real dictionary
    if 'policy_kwargs' in params:
        params['policy_kwargs'] = parse_policy_kwargs(params['policy_kwargs'])

    # 3. Setup Directories
    log_dir = f"tensorboard_log/{env_id}"
    gif_dir = f"gifs/{env_id}"
    model_dir = f"models/{env_id}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"--- Starting Training: {algo_name} on {env_id} | Envs: {n_envs} ---")

    for seed in seeds:
        run_name = f"{algo_name}_{env_id}_seed_{seed}"
        print(f"Training Seed: {seed}")

        # 4. Create Vectorized & Normalized Environment
        # Training env uses n_envs (e.g., 16 for A2C) to stabilize gradients
        env = make_vec_env(env_id, n_envs=n_envs, seed=seed)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        # 5. Initialize Algorithm
        if algo_name == "A2C":
            model = A2C(policy_type, env, verbose=0, tensorboard_log=log_dir, seed=seed, **params)
        elif algo_name == "PPO":
            model = PPO(policy_type, env, verbose=0, tensorboard_log=log_dir, seed=seed, **params)
        
        # 6. Train the Agent
        model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, progress_bar=True)
 
        # 7. Save the Model
        save_path = f"{model_dir}/{run_name}"
        model.save(save_path)
        print(f"✅ Model saved to: {save_path}.zip")
        
        # 8. Evaluation & GIF Generation
        # We must use a separate environment for evaluation (n_envs=1) because
        # we want to record a single continuous episode, not 16 fragmented ones.
        print(f"Generating GIF for seed {seed}...")
        
        eval_env = make_vec_env(env_id, n_envs=1, seed=seed, env_kwargs=dict(render_mode='rgb_array'))
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
        
        # Sync the normalization stats from the training env to the eval env
        # This ensures the agent sees the world in the same way it was trained.
        eval_env.training = False 
        eval_env.obs_rms = env.obs_rms 

        frames = []
        obs = eval_env.reset()
        done = False
        
        # Record one episode
        while not done:
            frame = eval_env.render()
            frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = eval_env.step(action)
            
            # Handle VecEnv output where done is a numpy array
            if isinstance(done, np.ndarray):
                done = done[0]

        # Save GIF
        gif_path = f"{gif_dir}/{algo_name}_seed_{seed}.gif"
        imageio.mimsave(gif_path, frames, fps=30)
        print(f"📹 GIF saved to: {gif_path}")
        
        env.close()
        eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True, help="Environment ID (e.g., Hopper-v5)")
    parser.add_argument("--algo", type=str, required=True, choices=["A2C", "PPO"], help="Algorithm to use")
    parser.add_argument("--hyperparams", type=str, default="config.yaml", help="Path to yaml config file")
    
    args = parser.parse_args()
    
    train_and_evaluate(args.env, args.algo, args.hyperparams)