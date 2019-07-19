# Dependencies
import torch
import wandb


class ForwardModel():
    r"""Implements the forward model for RL."""
    
    def __int__(self, latent_size=256):
        r"""The constructor.

        inputs:
        -------
        latent_size=256: The latent size for the model.

        outputs:
        --------
        """
        self.latent_size = 256

    def train(self, train_eps):
        r"""Implements training of the forward model.

        inputs:
        -------
        train_eps: The training episodes in appropriate format.

        outputs:
        --------
        """
        pass

    def predict(self, z, a):
        r"""Performs the predictions for the next instant.

        inputs:
        -------
        z: The current latent representation.

        a: The current input (action?)

        outputs:
        --------
        """
        pass

    def log_metrics(self):
        r"""Logs information of results and metrics.

        inputs:
        -------

        outputs:
        --------
        """
        pass
