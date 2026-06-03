#!/usr/bin/env bash
#
#SBATCH --job-name=brc
#SBATCH --time=0-07:00:00
#SBATCH --partition=a100
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --output="logs/slurm-%j-%x.txt"
#SBATCH --error="logs/slurm-%j-%x.txt"

set -eux

source entropy/.env_entropy

uv run train.py \
    --env_names=DMC_DOGS \
    --max_steps=500000 \
    --arch=simbaV2 \
