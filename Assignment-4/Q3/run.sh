#!/bin/bash

# Define the config file path
CONFIG="config.yaml"

echo "=========================================="
echo "Starting Batch Training for Stable Baselines3"
echo "=========================================="

# ---------------- A2C Experiments ----------------
echo ">>> Running A2C Experiments..."

# python train_sb3.py --env InvertedPendulum-v5 --algo A2C --hyperparams $CONFIG
# python train_sb3.py --env Hopper-v5 --algo A2C --hyperparams $CONFIG
# python train_sb3.py --env HalfCheetah-v5 --algo A2C --hyperparams $CONFIG

# ---------------- PPO Experiments ----------------
echo ">>> Running PPO Experiments..."

# python train_sb3.py --env InvertedPendulum-v5 --algo PPO --hyperparams $CONFIG
# python train_sb3.py --env Hopper-v5 --algo PPO --hyperparams $CONFIG
python train_sb3.py --env HalfCheetah-v5 --algo PPO --hyperparams $CONFIG

echo "=========================================="
echo "All experiments completed."
echo "Tensorboard logs available in: tensorboard_log/"
echo "GIFs available in: gifs/"
echo "Run 'tensorboard --logdir tensorboard_log' to view plots."
echo "=========================================="