#!/usr/bin/env bash
#
#SBATCH --job-name=brc
#SBATCH --time=0-20:00:00
#SBATCH --partition=h100
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output="logs/slurm-%j-%x.txt"
#SBATCH --error="logs/slurm-%j-%x.txt"

set -eux

source entropy/.env_entropy

# uv run train.py \
#     --env_names=DMC_DOGS \
#     --max_steps=500000 \
#     --arch=simbaV2 \
#     --seed=4 \
uv run python entropy/run_experiments.py --norender
