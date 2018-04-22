#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=training
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
 
cd /home/yhh303/AI-writer_Data2Doc/train/ 
module load pytorch/python3.6/0.3.0_4
python3 train.py
