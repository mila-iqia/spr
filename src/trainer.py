import torch


class Trainer(torch.nn.Module):
    def __init__(self, encoder, wandb, device=torch.device('cpu')):
        super().__init__()
        self.encoder = encoder
        self.wandb = wandb
        self.device = device

    def generate_batch(self, episodes):
        raise NotImplementedError

    def train(self, episodes):
        raise NotImplementedError

    def log_results(self, epoch_idx, epoch_loss, accuracy):
        raise NotImplementedError

