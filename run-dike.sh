#!/bin/bash
#
#SBATCH --job-name=job
#SBATCH --output=log/result-%j.out
#SBATCH --error=log/rresult-%j.err
#
#SBATCH --nodes=1
#SBATCH --nodelist=komputasi06
#SBATCH --time=20:00:00

source .venv/bin/activate
srun python3 status.py