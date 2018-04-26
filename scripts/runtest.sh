#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --pa333rtition=p1080_4
 
cd /home/yhh303/AI-writer_Data2Doc/train/ 
module load pytorch/python3.6/0.3.0_4
python3 -u train.py
