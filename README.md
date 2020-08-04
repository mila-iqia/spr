# abstract-world-models

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
pip install --user -e git+git://ithub.com/ankeshanand/abstract-world-models
```

## Usage:
The default branch for the latest and stable changes is `release`. 

* Sample run script
```bash
python -m scripts.run_pizero --grayscale --game ms_pacman --num-envs 64 --num-trainers 3 --no-gpu-0-train 
```
This will launch a MuZero run with NCE enabled on 4GPUs where GPU0 is used only for search / reanalyze. 
For all our experiments, we are using `--batch-size-per-worker 200` too, but this can change depending on the amount of GPU memory.

Here are some options that might be useful when running experiments:
* For disabling NCE loss during training, pass the flag `--no-nce`
* For running purely Q-learning (no search), use the following options: 
```bash
python -m scripts.run_pizero --grayscale --game ms_pacman --num-envs 64 --q-learning --no-nce --policy-loss-weight 0. --reward-loss-weight 0. --no-search-value-targets --local-target-net --num-simulations 0 --eval-simulations 0 --jumps 0  --num-trainers 3 --no-gpu-0-train 
```
* To run Q-learning with search, use the following options:
```bash
python -m scripts.run_pizero --grayscale --game ms_pacman --num-envs 64 --q-learning --no-nce --policy-loss-weight 0. --reward-loss-weight 0. --no-search-value-targets --local-target-net --c1 0.25 --num-trainers 3 --no-gpu-0-train 
```
A WIP implementation of the C51 Q-learning in currently located in the `c51` branch.


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
