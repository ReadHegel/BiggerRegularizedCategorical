#!/usr/bin/env bash
#
#SBATCH --job-name=brc
#SBATCH --time=30:00:00
#SBATCH --partition=a100
#SBATCH --qos=aleksanderwojsz_a100
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --output="logs/slurm-%j-%x.txt"
#SBATCH --error="logs/slurm-%j-%x.txt"

set -eux

source entropy/.env_entropy

mkdir -p "$STORAGE/tmp"
export TMPDIR="$STORAGE/tmp"

uv run run_experiments.py
