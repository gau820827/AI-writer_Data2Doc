#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=training
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:p100:1

#SBATCH --mail-type=end  # email me when the job ends

# Change the home directory
cd /home/yhh303/AI-writer_Data2Doc/train/
module load pytorch/python3.6/0.3.0_4
python3 -u train.py -embed 600 -lr 0.01 -batch 1 -getloss 20 -encoder HierarchicalRNN -decoder HierarchicalRNN -epochsave 5 -copy True -copyplayer True -gradclip 5 -layer 2 -epoch 10