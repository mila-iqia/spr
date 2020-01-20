# abstract-world-models

## Dependencies: 
* PyTorch 
* wandb (logging tool)
* gym

To install the requirements, follow these steps:
```bash
# PyTorch
conda install pytorch torchvision -c pytorch
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# Install requirements
pip install --user opencv-python wandb matplotlib scikit-learn 'gym[atari]' plotly recordclass pyprind psutil
pip install --user git+git://github.com/mila-iqia/atari-representation-learning.git
pip install --user git+git://github.com/astooke/rlpyt

# Login to W&B
wandb login {wandb_key}
```

## Usage:

* Sample run script
```bash
python -m scripts.run_pizero --grayscale --game alien --use-all-targets --training-interval 32 --num-envs 16
```

## References:
Kaixin's Rainbow implementation: https://github.com/Kaixhin/Rainbow

MBPO: https://arxiv.org/abs/1906.08253
