#!/bin/bash
#
#SBATCH --job-name=job
#SBATCH --output=log/result-%j.out
#SBATCH --error=log/rresult-%j.err
#
#SBATCH --nodes=2
#SBATCH --nodelist=komputasi03,komputasi05
#SBATCH --time=20:00:00

source .venv/bin/activate
srun python3 main.py