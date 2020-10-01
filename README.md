# Data-Efficient Reinforcement Learning with Self-Predictive Representations

XXX

This repo provides code for implementing the SPR paper.

* [📦 Install ](#install) -- Install relevant dependencies and the project
* [🔧 Usage ](#usage) -- Commands to run different experiments from the paper

## Install 
To install the requirements, follow these steps:
```bash
# PyTorch
conda install pytorch torchvision -c pytorch
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Install requirements
pip install -r requirements.txt
```

## Usage:

* To run SPR with augmentation
```bash
python -m scripts.run --public --game pong --momentum-tau 1.
```

* To run SPR without augmentation
```bash
python -m scripts.run --public --game pong --augmentation none --target-augmentation 0 --momentum-tau 0.01 --dropout 0.5
```

When reporting scores, we average across 10 seeds. 

## What does each file do? 

    .
    ├── scripts
    │   └── run.py                # The main runner script to launch jobs.
    ├── src                     
    │   ├── agent.py              # Implements the Agent API for action selection 
    │   ├── algos.py              # Distributional RL loss
    │   ├── models.py             # Network architecture and forward passes.
    │   ├── rlpyt_atari_env.py    # Slightly modified Atari env from rlpyt
    │   ├── rlpyt_utils.py        # Utility methods that we use to extend rlpyt's functionality
    │   └── utils.py              # Command line arguments and helper functions 
    │
    └── requirements.txt          # Dependencies
