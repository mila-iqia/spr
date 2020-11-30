#!/bin/bash
source $HOME/.bashrc
echo Running on $HOSTNAME
nvidia-smi
date

cd ~/Github/AMBLe/
conda activate pytorch
wandb off

echo ${DL_ARGS}

python -u -m scripts.run ${DL_ARGS}
