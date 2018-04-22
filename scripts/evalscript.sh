#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=small_eval
#SBATCH --output=eval_slurm_%j.out
#SBATCH --error=eval_slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=5GB
#SBATCH --gres=gpu:1
 
cd /home/yhh303/AI-writer_Data2Doc/train/ 
module load pytorch/python3.6/0.3.0_4
python3 small_evaluate.py
