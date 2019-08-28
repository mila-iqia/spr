from itertools import chain

import torch
import torch.nn as nn
import numpy as np

import wandb

from agent import Agent
from main import dqn
from memory import ReplayMemory
from src.encoders import NatureCNN, ImpalaCNN
from src.envs import make_vec_envs, Env
from src.forward_model import ForwardModel
from src.stdim import InfoNCESpatioTemporalTrainer
from src.utils import get_argparser
from src.episodes import get_random_agent_episodes, sample_state, Transition


def train_policy(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    env = Env(args)
    env.train()

    # get initial exploration data
    real_transitions = get_random_agent_episodes(args)
    model_transitions = ReplayMemory(args, args.fake_buffer_capacity)
    j, rollout_length = 0, args.rollout_length
    dqn = Agent(args, args.env)

    state, done = env.reset(), False
    while j * args.env_steps_per_epoch < args.total_steps:
        # Train encoder and forward model on real data
        encoder = train_encoder(args, real_transitions)
        forward_model = train_model(args, encoder, real_transitions)

        timestep, done = 0, True
        for e in range(args.env_steps_per_epoch):
            if done:
                state, done = env.reset(), False
            state = state[-1].mul(255).to(dtype=torch.uint8,
                                          device=torch.device('cpu')) # Only store last frame and discretise to save memory
            # Take action in env acc. to current policy, and add to real_transitions
            action = dqn.act(encoder(state))
            next_state, reward, done = env.step(action)
            real_transitions.append(Transition(timestep, state, action, reward, not done))
            state = next_state
            timestep = 0 if done else timestep + 1

            for m in range(args.num_model_rollouts // args.rollout_batch_size):
                # sample a state uniformly from real_transitions
                s = sample_state(real_transitions)

                # Perform k-step model rollout starting from s using current policy
                # Add imagined data to model_transitions
                for k in range(rollout_length):
                    action = dqn.act(s)
                    next_obs, reward = forward_model.predict(s, action)
                    # figure out what to do about terminal state here
                    model_transitions.append(state, action, reward, False)

            # Update policy parameters on model data
            for g in range(args.updates_per_epoch):
                dqn.learn(model_transitions)
        j += 1


def train_encoder(args, transitions, val_eps=None):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")

    observation_shape = transitions[0].state.shape
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

    trainer.train(transitions, val_eps)
    return encoder


def train_model(args, encoder, tr_eps, val_eps=None):
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
