import torch
import torch.nn as nn
import numpy as np

import wandb

from src.encoders import NatureCNN, ImpalaCNN
from src.forward_model import ForwardModel
from src.stdim import InfoNCESpatioTemporalTrainer
from src.utils import get_argparser
from src.episodes import get_random_agent_episodes


def train_policy(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

    tr_eps, val_eps = get_random_agent_episodes(args, device, args.pretraining_steps)
    encoder = train_encoder(args, tr_eps, val_eps)
    forward_model = train_model(args, encoder, tr_eps, val_eps)
    # train a PPO policy using this model


def train_encoder(args, tr_eps, val_eps):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

    observation_shape = tr_eps[0][0].shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)
    encoder.to(device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack
    if args.method == "infonce-stdim":
        trainer = InfoNCESpatioTemporalTrainer(encoder, config, device=device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    trainer.train(tr_eps, val_eps)
    return encoder


def train_model(args, encoder, tr_eps, val_eps):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    forward_model = ForwardModel(args, encoder, device)
    forward_model.train(tr_eps)
    return forward_model


if __name__ == '__main__':
    wandb.init()
    parser = get_argparser()
    args = parser.parse_args()
    tags = []
    wandb.init(project=args.wandb_proj, entity="abs-world-models", tags=tags)
    train_policy(args)
