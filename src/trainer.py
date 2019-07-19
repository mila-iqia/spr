# Dependencies
import torch


class Trainer():
    r"""Implements a trainer class."""

    def __init__(self, encoder, wandb, device=torch.device('cpu')):
        r"""The constructor.

        inputs:
        -------
        encoder: The encoder torch model.

        wandb: Weights and Biases handle for visualization data.

        device=torch.device('cpu'): The device on which to load the model.

        outputs:
        --------
        """
        self.encoder = encoder
        self.wandb = wandb
        self.device = device

    def generate_batch(self, episodes):
        r"""Generates batches from input episodes.

        inputs:
        -------
        episodes: The episodes in appropriate format.

        outputs:
        --------
        """
        raise NotImplementedError

    def train(self, episodes):
        r"""Train the model based on the input episodes.

        inputs:
        -------
        episodes: The episodes in appropriate format.

        outputs:
        --------
        """
        raise NotImplementedError

    def log_results(self, epoch_idx, epoch_loss, accuracy):
        r"""Logs the results like loss and accuracy per epoch.

        inputs:
        -------
        epoch_idx: The epoch.

        epoch_loss: Loss in the epoch.

        accuracy: The accuracy in the epoch.

        outputs:
        --------
        """
        raise NotImplementedError

