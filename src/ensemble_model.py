# Dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import wandb

import sys

from src.episodes import get_framestacked_transition


###############################################################################


class ForwardModel(nn.Module):
    r"""Holds the ensemble of environment models. The ensemble is used to
    model the epistemic uncertainty in the environment models, whereas the 
    individual model's stochastic predictions, if incorporated, would model 
    the aleatoric uncertainty.
    """

    # [TODO] Please verify the docstring for encoder.
    def __init__(self, args, encoder, num_ensemble=5, **kwargs):
        r"""The constructor.

        inputs:
        -------
        args: The args that should be passed to the constructor of the 
            individual forward model.

        encoder: The encoder that needs to be passed to the constructor of the
            individual forward model. Ideally, it should be the unique encoder
            that gets passed to all the forward models.

        num_ensemble=5: The number of forward models in the ensemble.
        """
        super(ForwardModel, self).__init__()
        # Set attributes.
        self.args = args
        self.encoder = encoder
        self.num_ensemble = num_ensemble
        self.kwargs = kwargs
        # Set the prediction strategy.
        self.set_prediction_strategy()
        # Create the num_ensemble-many forward models.
        self.model = nn.ModuleList()
        for _ in range(self.num_ensemble):
            self.model.append(
                EnvModel(args=self.args, encoder=self.encoder))

    ## [TODO] Maintain the support for prediction strategies.
    def set_prediction_strategy(self):
        r"""Sets the prediction strategy. As per MBPO, the ensemble's 
        prediction is defined as the output of a randomly chosen individual
        environment model. However, this is not the only strategy. We can also
        combine the (stochastic) predictions of the individuals to obtain a 
        joint/weighted sample. SUPPORTS: 'Deterministic-IndexBased', 
        'Deterministic-RandomIndex', 'Stochastic-AllIndices'. The default is 
        chosen to be the same as that in MBPO: 'Deterministic-RandomIndex'.

        inputs:
        -------

        outputs:
        --------
        """
        if hasattr(self.args, 'prediction_strategy'):
            self.prediction_strategy = self.args.prediction_strategy
        elif 'prediction_strategy' in self.kwargs:
            self.prediction_strategy = self.kwargs['prediction_strategy']
        else:
            print('[WARNING] Prediction strategy of ForwardModel',
                'not provided.', 'Setting to the default:', 
                'Deterministic-RandomIndex.')
            self.prediction_strategy = 'Deterministic-RandomIndex'

    #
    def train(self, real_transitions):
        r"""Implements training method using real transitions. Note that the
        training method calls the training method of each of the individual
        models, which then select their own batches with repetitions. Thus,
        the setup mimics the standard boot-strapping.

        inputs:
        -------
        real_transitions: The set of real transitions, each being an instance 
            of the Namespace Transition from src.episodes.

        outputs:
        --------
        """
        # Train all the environmental models one-by-one.
        for idx in range(self.num_ensemble):
            self.model[idx].train(
                real_transitions=real_transitions)

    # [TODO] Maintain support for the prediction strategies and batch data.
    def predict(self, z, a):
        r"""Method to predict for a state in latent space and an action the 
        next latent space.

        inputs:
        -------
        z: The state representation in latent space. SHAPE: 
            [<batch_size>, <feature_size>*4].
        a: The actions. SHAPE: [<batch_size>, 1].

        outputs:
        --------
        z_next: The next state representations prediction. SHAPE: 
            [<batch_size>, <hidden_size>].
        r_next: The next reward prediction. SHAPE: [<batch_size>, 1].
        """
        if self.prediction_strategy == 'Deterministic-RandomIndex':
            assert(len(list(z.shape)) == 1)
            # Select a random model for prediction.
            idx = np.random.randint(0, self.num_ensemble)
            z_next, r_next = self.model[idx].predict(z, a)
            return z_next, r_next
        else:
            print('[ERROR] Support for prediction strategy:', 
                self.prediction_strategy, 'not added.')
            sys.exit()


###############################################################################


class EnvModel(nn.Module):
    r"""Holds an individual environment model."""

    #
    def __init__(self, args, encoder, **kwargs):
        r"""The constructor.

        inputs:
        -------
        args: The args required for the environment model.

        encoder: The encoder used in an epoch of the environment model.

        outputs:
        --------
        """
        super(EnvModel, self).__init__()
        # Set attributes.
        self.args = args
        self.device = args.device
        self.encoder = encoder
        hidden_size = args.forward_hidden_size
        self.hidden = nn.Linear(args.feature_size * 4, hidden_size).to(self.device)
        self.sd_predictor = nn.Linear(hidden_size, args.feature_size).to(self.device)
        self.reward_predictor = nn.Linear(hidden_size, 1).to(self.device)
        # Set the learning rate.
        self.set_learning_rate()
        # Create a list for all the model parameters
        self.parameters_list = []
        self.parameters_list += list(self.hidden.parameters())\
                + list(self.sd_predictor.parameters())\
                + list(self.reward_predictor.parameters())
        # Create one optimizer for all the model parameters.
        self.optimizer = torch.optim.Adam(
            self.parameters_list, lr=self.learning_rate)

    ##
    def set_learning_rate(self):
        r"""Sets the learning rate for the optimizer. The default is 1e-3.
        
        inputs:
        -------

        outputs:
        --------
        """
        if hasattr(self.args, 'learning_rate'):
            self.learning_rate = self.args.learning_rate
        elif 'learning_rate' in self.kwargs:
            self.learning_rate = self.kwargs['learning_rate']
        else:
            print('[WARNING] Learning rate of ForwardModel not provided.',
                'Setting to the default: 1e-3.')
            self.learning_rate = 1e-3

    # [TODO] Fill in the output tensor shapes correctly.
    def generate_batch(self, transitions):
        r"""Defines a generator for batches of data. Each batch consists of
        the stacked current frame state, the action taken, the next state and 
        the corresponding reward. These batches are used to train the forward 
        model via the training loop of the ensemble. Note that the selection 
        of the data points is boot-strapped, which here means that the samples
        are picked with replacement.

        inputs:
        -------
        transitions: The list of all the transitions, each of which is an 
            instance of the Namespace Transition from src.episodes.

        outputs:
        --------
        state_tensor (implicit): The torch tensor of stacked-states. SHAPE:
            [<batch_size>, <stack=4>, <num_channels>, <height>, <width>].
        next_state_tensor (implicit): The torch tensor of next state. SHAPE:
            [<batch_size>, <num_channels>, <height>, <width>]. 
        action_tensor (implicit): The torch tensor of the actions. SHAPE: 
            [<batch_size>, 1].
        reward_tensor (implicit): The torch tensor of the rewards. SHAPE:
            [<batch_size>, 1].
        """
        # Compute the total steps that can be exploited.
        total_steps = len(transitions)
        # Loop for each of the iterations.
        for idx in range(total_steps // self.args.batch_size):
            # Get a batch of transition indices.
            indices = np.random.randint(
                0, total_steps-1, size=self.args.batch_size)
            # Initialize the required dataholders.
            s_t, x_t_next, a_t, r_t = [], [], [], []
            # For each of the transition...
            for t in indices:
                # (Update untill we) get a non-terminal transition index.
                while transitions[t].nonterminal is False:
                    t = np.random.randint(0, total_steps-1)
                # Get frame-stacked transition corresponding to the index.
                framestacked_transition = get_framestacked_transition(t, transitions, device=self.device)
                s_t.append(torch.stack([trans.state for trans in framestacked_transition]))
                x_t_next.append(transitions[t + 1].state)
                a_t.append(framestacked_transition[-1].action)
                r_t.append(framestacked_transition[-1].reward)
            # Yield the data to convert the method into a generator.
            yield torch.stack(s_t).float().to(self.device) / 255.,\
                torch.stack(x_t_next).float().to(self.device) / 255.,\
                torch.tensor(a_t).unsqueeze(-1).float().to(self.device),\
                torch.tensor(r_t).unsqueeze(-1).float().to(self.device)

    # [TODO] Cite PETS, and the paper that encourages modeling delta s_t.
    def do_one_epoch(self, epoch, episodes):
        r"""Runs one epoch of the training of the individual forward model.

        inputs:
        -------
        epoch: The epoch count, used mainly only for logging purposes.

        episodes: The set of transitions, each being an instance of the 
            Namespace Transition from src.episodes.

        outputs:
        --------
        """
        # Get a batch-generator for the episodes.
        data_generator = self.generate_batch(episodes)
        # Initialize terms for tracking.
        epoch_loss, epoch_sd_loss, epoch_reward_loss, steps = 0., 0., 0., 0
        # For each batch...
        for s_t, x_t_next, a_t, r_t in data_generator:
            # Obtain encodings for the stacked-states.
            s_t = s_t.view(
                self.args.batch_size * 4, 1, s_t.shape[-2], s_t.shape[-1])
            with torch.no_grad():
                f_t, f_t_next = self.encoder(s_t), self.encoder(x_t_next)
                f_t = f_t.view(self.args.batch_size, 4, -1)
            f_t_last = f_t[:, -1, :]
            f_t = f_t.view(self.args.batch_size, -1)
            hiddens = self.hidden(f_t * a_t)
            sd_predictions = self.sd_predictor(hiddens)
            reward_predictions = self.reward_predictor(hiddens)
            # Predict |s_{t+1} - s_t| instead of s_{t+1} directly
            sd_loss = F.mse_loss(sd_predictions, f_t_next - f_t_last)
            reward_loss = F.mse_loss(reward_predictions, r_t)

            loss = sd_loss + reward_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_sd_loss += sd_loss.detach().item()
            epoch_reward_loss += reward_loss.detach().item()
            epoch_loss += loss.detach().item()
            steps += 1
        self.log_metrics(epoch, epoch_loss / steps, epoch_reward_loss / steps,
            epoch_reward_loss / steps)

    #
    def train(self, real_transitions):
        r"""Implements training method using real transitions.

        inputs:
        -------
        real_transitions: The set of real transitions, each being an instance 
            of the Namespace Transition from src.episodes.

        outputs:
        --------
        """
        for e in range(self.args.epochs):
            self.do_one_epoch(e, real_transitions)

    #
    def predict(self, z, a):
        r"""Method to predict for a state in latent space and an action the 
        next latent space.

        inputs:
        -------
        z: The state representation in latent space. SHAPE: 
            [<batch_size>, <feature_size>*4].
        a: The actions. SHAPE: [<batch_size>, 1] or just an int.

        outputs:
        --------
        z_next: The next state representations prediction. SHAPE: 
            [<batch_size>, <hidden_size>].
        r_next: The next reward prediction. SHAPE: [<batch_size>, 1].
        """
        hidden = self.hidden(z * a)
        return self.sd_predictor(hidden), self.reward_predictor(hidden)

    #
    def log_metrics(self, epoch_idx, epoch_loss, sd_loss, reward_loss):
        r"""Logs the different metrics of performance to weights and biases.

        inputs:
        -------
        epoch_idx: 

        outputs:
        --------
        """
        print("Epoch: {}, Epoch Loss: {}, SD Loss: {}, Reward Loss: {}".
              format(epoch_idx, epoch_loss, sd_loss, reward_loss))
        wandb.log({'Dynamics loss': epoch_loss,
                   'SD Loss': sd_loss,
                   'Reward Loss': reward_loss}, step=epoch_idx)


# Pseudo-main.
if __name__ == '__main__':
    
    from dotmap import DotMap

    # Create a dummy class for args. DotMap fails!
    class Configs(object):
        pass

    env_args = Configs()
    env_args.device = torch.device('cpu')
    env_args.forward_hidden_size = 10
    env_args.feature_size = 256
    # Attempt creating an EnvModel instance.
    env_model = EnvModel(
        args=env_args, 
        encoder=None)
    print('env_model:\n', env_model)

    ensemble_env_args = Configs()
    ensemble_env_args.device = torch.device('cpu')
    ensemble_env_args.forward_hidden_size = 10
    ensemble_env_args.feature_size = 256
    # Attempt creating an ForwardModel instance.
    ensemble_env_model = ForwardModel(
        args=ensemble_env_args, 
        encoder=None,
        num_ensemble=10)
    print('ensemble_env_model:\n', ensemble_env_model)