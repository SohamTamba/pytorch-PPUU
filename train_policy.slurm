#!/bin/bash

#SBATCH --job-name=train_policy
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mail-type=END
#SBATCH --mail-user=sgt287@nyu.edu

#SBATCH --mem=32G
#SBATCH -c 4

. ~/.bashrc
conda activate PPUU_FW

echo $PWD
which python

python train_MPUR.py