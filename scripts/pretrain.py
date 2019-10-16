from collections import deque
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import wandb

from src.agent import Agent
from src.memory import ReplayMemory
from src.encoders import NatureCNN, ImpalaCNN
from src.envs import Env
from src.eval import test
from src.forward_model import ForwardModel
from src.stdim import InfoNCESpatioTemporalTrainer
from src.dqn_multi_step_stdim_with_actions import MultiStepActionInfoNCESpatioTemporalTrainer
from src.stdim_with_actions import ActionInfoNCESpatioTemporalTrainer
from src.utils import get_argparser, log, set_learning_rate
from src.episodes import get_random_agent_episodes, Transition, sample_real_transitions
from src.memory import blank_trans


def pretrain(args):
    env = Env(args)
    env.train()
    dqn = Agent(args, env)

    # get initial exploration data
    real_transitions = get_random_agent_episodes(args)
    val_transitions = get_random_agent_episodes(args)
    encoder, encoder_trainer = init_encoder(args,
                                            real_transitions,
                                            num_actions=env.action_space(),
                                            agent=dqn)

    encoder_trainer.train(real_transitions,
                          val_transitions,
                          epochs=args.pretrain_epochs)

    if args.integrated_model:
        forward_model = encoder_trainer
    if not args.integrated_model:
        forward_model = train_model(args,
                                    encoder,
                                    real_transitions,
                                    env.action_space(),
                                    init_epochs=args.pretrain_epochs,
                                    val_eps=val_transitions)
        forward_model.args.epochs = args.epochs // 2
        encoder_trainer.epochs = args.epochs // 2

    visualize_temporal_prediction_accuracy(forward_model, val_transitions, args)


def visualize_temporal_prediction_accuracy(model, transitions, args):
    with torch.no_grad():
        model.minimum_length = args.visualization_jumps
        model.maximum_length = args.visualization_jumps + 1
        model.dense_supervision = True

        model.encoder.eval(), model.classifier.eval()
        model.do_one_epoch(transitions, plots=True)
        model.encoder.train(), model.classifier.train()

        model.minimum_length = args.min_jump_length
        model.maximum_length = args.max_jump_length
        model.dense_supervision = args.dense_supervision

def init_encoder(args,
                 transitions,
                 num_actions,
                 agent=None):
    if args.integrated_model:
        trainer = MultiStepActionInfoNCESpatioTemporalTrainer
    else:
        trainer = InfoNCESpatioTemporalTrainer

    observation_shape = transitions[0].state.shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)
    encoder.to(args.device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack
    config['num_actions'] = num_actions  # weird hack
    if args.method == "infonce-stdim":
        if args.online_agent_training:
            trainer = trainer(encoder,
                              config,
                              device=args.device,
                              wandb=wandb,
                              agent=agent)
        else:
            trainer = trainer(encoder, config, device=args.device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)
    return encoder, trainer


def train_model(args,
                encoder,
                real_transitions,
                num_actions,
                val_eps=None,
                init_epochs=None):
    forward_model = ForwardModel(args, encoder, num_actions)
    forward_model.train(real_transitions, init_epochs)
    if val_eps is not None:
        forward_model.sd_predictor.eval()
        forward_model.hidden.eval()
        forward_model.reward_predictor.eval()
        forward_model.train(val_eps, init_epochs)
        forward_model.sd_predictor.train()
        forward_model.hidden.train()
        forward_model.reward_predictor.train()
    return forward_model


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    tags = ["Pretraining"]
    if len(args.name) > 0:
        wandb.init(project=args.wandb_proj, tags=tags, name=args.name, entity="abs-world-models")
    else:
        wandb.init(project=args.wandb_proj, tags=tags, entity="abs-world-models")
    wandb.config.update(vars(args))

    results_dir = os.path.join('results', args.id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')
    pretrain(args)
