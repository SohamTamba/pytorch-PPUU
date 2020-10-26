#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=12:00:00
##SBATCH --gres=gpu:1080ti:1
#SBATCH --gres=gpu:0
#SBATCH --mail-type=END
#SBATCH --mail-user=sgt287@nyu.edu

#SBATCH --mem=32G
#SBATCH -c 4

. ~/.bashrc
conda activate PPUU_FW

echo $PWD
which python

##for t in 0 1 2; do python generate_trajectories.py -map i80 -time_slot $t; done
python generate_trajectories.py -map i80 -time_slot 2
python extract_car_size.py
