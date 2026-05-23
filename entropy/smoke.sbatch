#!/usr/bin/env bash
#
#SBATCH --job-name=smoke_brc
#SBATCH --time=0-00:20:00
#SBATCH --partition=a100
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --output="logs/slurm-%j-%x.txt"
#SBATCH --error="logs/slurm-%j-%x.txt"

set -eux

source entropy/.env_entropy

uv run train.py \
  --env_names=cheetah-run \
  --arch=bro \
  --max_steps=50 \
  --start_training=10 \
  --replay_buffer_size=5000 \
  --batch_size=64 \
  --updates_per_step=1 \
  --eval_interval=100000 \
  --offline_evaluation=False \
  --log_to_wandb=False \
  --render=False \
  --log_interval=1