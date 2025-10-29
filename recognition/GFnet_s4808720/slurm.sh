#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1

# set name of job
#SBATCH --job-name=GFNet

# total runtime, enable to use GPUs with max times
#SBATCH --time=1-00:00:00 #1 days

# output file
#SBATCH --output=GFNetADNI.out

conda activate torch
python train.py 
python predict.py