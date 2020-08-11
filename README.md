# Data-Efficient Reinforcement Learning with Self-Predictive Representations

*Max Schwarzer\*, Ankesh Anand\*, Rishab Goel, R Devon Hjelm, Aaron Courville, Philip Bachman*

This repo provides code for implementing the [SPR paper](https://arxiv.org/abs/2007.05929)

## Installation 
To install the requirements, follow these steps:
```bash
# PyTorch
conda install pytorch torchvision -c pytorch dill
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Install requirements
pip install -r requirements.txt

# Login to W&B
wandb login {wandb_key}

# Finally, install the project
pip install --user -e git+git://github.com/ankeshanand/abstract-world-models
```

## Usage:
The default branch for the latest and stable changes is `release`. 

* Sample run script with augmentation
```bash
python -m scripts.run_pizero --grayscale --game ms_pacman --num-envs 64 --num-trainers 3 --no-gpu-0-train 
```

## What does each file do? 

    .
    ├── scripts
    │   └── run_pizero.py         # The main runner script to launch jobs.
    ├── src                     
    │   ├── async_mcts.py         # Legacy / Defunct. Ignore this. 
    │   ├── async_reanalyze.py    # An async worker that performs ReAnalyze MCTS. Episodes are read and written to disk.
    │   ├── encoders.py           # Legacy / Defunt. Contains old nature conv encoders
    │   ├── logging.py            # Utils for logging metrics during training
    │   ├── mcts_memory.py        # Extends rlpyt's buffer for storing additional items such as the prior policy.
    │   ├── model_trainer.py      # The main file that contrains training code for everything. Also includes all the network architectures for MuZero. 
    │   ├── pizero.py             # Legacy / Defunct. Old MCTS implemetation.
    │   └── utils.py              # Command line arguments and helper functions 
    │
    └── ms_run_pizero.yaml        # YAML file to run experiments on Philly
