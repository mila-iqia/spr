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
from src.framestack_stdim_with_actions import FramestackActionInfoNCESpatioTemporalTrainer
from src.stdim_with_actions import ActionInfoNCESpatioTemporalTrainer
from src.utils import get_argparser, log, set_learning_rate
from src.episodes import get_random_agent_episodes, Transition, sample_real_transitions


def train_policy(args):
    env = Env(args)
    env.train()

    # get initial exploration data
    real_transitions = get_random_agent_episodes(args)
    model_transitions = ReplayMemory(args, args.fake_buffer_capacity)
    encoder, encoder_trainer = train_encoder(args,
                                             real_transitions,
                                             num_actions=env.action_space(),
                                             init_epochs=args.pretrain_epochs)
    if args.integrated_model:
        forward_model = encoder_trainer
    else:
        forward_model = train_model(args,
                                    encoder,
                                    real_transitions,
                                    env.action_space(),
                                    init_epochs=args.pretrain_epochs)
        forward_model.args.epochs = args.epochs // 2
        encoder_trainer.epochs = args.epochs // 2

    j = 0
    dqn = Agent(args, env)
    dqn.train()
    results_dir = os.path.join('results', args.id)
    metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}

    state, done = env.reset(), False
    while j * args.env_steps_per_epoch < args.total_steps:
        # Train encoder and forward model on real data
        if j != 0:
            # encoder_lr = max(args.encoder_lr / (j*5), 1e-6)
            # set_learning_rate(encoder_trainer.optimizer, encoder_lr)
            if args.integrated_model:
                encoder_trainer.train(real_transitions)
            else:
                forward_model.train(real_transitions)

        steps = j * args.env_steps_per_epoch
        if steps % args.evaluation_interval == 0:
            dqn.eval()  # Set DQN (online network) to evaluation mode
            avg_reward = test(args, steps, dqn, encoder, metrics, results_dir, evaluate=True)  # Test
            log(steps, avg_reward)
            dqn.train()  # Set DQN (online network) back to training mode

        timestep, done = 0, True
        for e in range(args.env_steps_per_epoch):
            if done:
                state, done = env.reset(), False
            # Take action in env acc. to current policy, and add to real_transitions
            real_z = encoder(state).view(-1)
            action = dqn.act(real_z)
            next_state, reward, done = env.step(action)
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
            real_transitions.append(Transition(timestep, state, action, reward, not done))
            state = next_state
            timestep = 0 if done else timestep + 1

            # sample states from real_transitions
            samples = sample_real_transitions(real_transitions, args.num_model_rollouts).to(args.device)
            samples = samples.flatten(0, 1)
            H, N = args.history_length, args.num_model_rollouts
            with torch.no_grad():
                z = encoder(samples).view(H, N, -1)
            state_deque = deque(maxlen=4)
            for s in z.unbind():
                state_deque.append(s)

            # Perform k-step model rollout starting from s using current policy
            for k in range(args.rollout_length):
                z = torch.stack(list(state_deque))
                z = z.view(N, H, -1).view(N, -1)  # take a second look at this later
                actions = dqn.act(z, batch=True)
                actions_one_hot = F.one_hot(actions, num_classes=env.action_space())
                with torch.no_grad():
                    next_z, rewards = forward_model.predict(z, actions_one_hot.float())
                z = z.view(N, H, -1)

                actions, rewards = actions.tolist(), rewards.tolist()
                # Add imagined data to model_transitions
                for i in range(N):
                    model_transitions.append(z[i], actions[i], rewards[i], True)
                state_deque.append(next_z)

            if j >= 1:
                # Update policy parameters on model data
                for g in range(args.updates_per_step):
                    dqn.learn(model_transitions)

            dqn.reset_noise()

        if j > 0:
            dqn.log(env_steps=(j+1) * args.env_steps_per_epoch)
        # Update target network
        if steps % args.target_update == 0:
            dqn.update_target_net()

        j += 1


def train_encoder(args, transitions, num_actions, val_eps=None, init_epochs=None):
    if args.integrated_model:
        trainer = FramestackActionInfoNCESpatioTemporalTrainer
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
        trainer = trainer(encoder, config, device=args.device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)

    trainer.train(transitions, val_eps, epochs=init_epochs)
    return encoder, trainer


def train_model(args, encoder, real_transitions, num_actions, val_eps=None, init_epochs=None):
    forward_model = ForwardModel(args, encoder, num_actions)
    forward_model.train(real_transitions, init_epochs)
    return forward_model


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    tags = []
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
    train_policy(args)
