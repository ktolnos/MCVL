#!/bin/bash
# Launch the main MCVL gridworld experiment (run_experiment.py).
#
# The SBATCH directives below are provided as an example for Slurm clusters;
# they are ignored when the script is run directly with `bash run.sh`.
#SBATCH --job-name=train_rew_tamp
#SBATCH --cpus-per-task=10
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

set -e

# Work from the repository root (the directory containing this script).
cd "$(dirname "$0")"

# Activate a local virtual environment if one exists.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

python run_experiment.py
