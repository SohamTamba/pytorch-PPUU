#!/bin/bash

#SBATCH --job-name=eval_policy
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2080ti:1
#SBATCH --mail-type=END
#SBATCH --mail-user=sgt287@nyu.edu

#SBATCH --mem=32G
#SBATCH -c 4

. ~/.bashrc
conda activate PPUU_FW

echo $PWD
which python


python eval_policy.py -policy_model MPUR-policy-deterministic-threshhold_fm=True-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=False-seed=1-novaluestep45000.model
python eval_policy.py -policy_model MPUR-policy-deterministic-threshhold_fm=False-model=vae-zdropout=0.5-nfeature=256-bsize=6-npred=30-ureg=0.05-lambdal=0.2-lambdaa=0.0-gamma=0.99-lrtz=0.0-updatez=0-inferz=False-learnedcost=False-seed=1-novaluestep45000.model


