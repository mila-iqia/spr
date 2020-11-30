#!/bin/bash
source $HOME/.bashrc
echo Running on $HOSTNAME
nvidia-smi
date

cd /scratch/schwarzm/Github/AMBLe
conda activate pytorch
wandb off
cp scripts/run.py ./
cp scripts/config.yaml ./

python run.py $DL_ARGS
