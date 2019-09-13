# abstract-world-models

## Dependencies: 
* PyTorch 
* wandb (logging tool)
* gym

To install the requirements, follow these steps:
```bash
# PyTorch
conda install pytorch torchvision -c pytorch

# Other requirements
pip install -r requirements.txt
```

## Usage:
* Add the project directory to your `$PYTHONPATH`

* Run the training script
```bash
python -m scripts.train_all
```

## References:
Kaixin's Rainbow implementation: https://github.com/Kaixhin/Rainbow

MBPO: https://arxiv.org/abs/1906.08253
