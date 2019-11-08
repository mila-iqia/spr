from collections import deque

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
from src.utils import get_argparser, log
from src.episodes import get_random_agent_episodes, Transition, sample_real_transitions, get_current_policy_episodes


def train_policy(args):
    env = Env(args)
    env.train()
    dqn = Agent(args, env.action_space())

    # get initial exploration data
    transitions = get_random_agent_episodes(args)
    real_transitions = ReplayMemory(args, args.real_buffer_capacity, images=True,
                                    priority_weight=args.model_priority_weight,
                                    priority_exponent=args.model_priority_exponent)
    for t in transitions:
        state, action, reward, terminal = t[1:]
        real_transitions.append(state, action, reward, not terminal)
    model_transitions = ReplayMemory(args, args.fake_buffer_capacity,
                                     no_overshoot=True)
    encoder, encoder_trainer = train_encoder(args,
                                             transitions,
                                             real_transitions,
                                             num_actions=env.action_space(),
                                             init_epochs=args.pretrain_epochs,
                                             agent=dqn)

    # General outline of validation set:
    # 1.  Append new steps to val_transitions.  Use a fixed # of episodes, to
    #     facilitate merging the buffers.  Start with random episodes.
    # 2.  Have ES tolerance of ~100 steps
    # 3.  If ES is triggered, the weight of the encoder shifts to
    #     DQN-only and the normal prediction/reward objectives are not trained.
    # 4.  This lasts until a set number of new transitions have been integrated,
    #     at which point a new validation set is created and the ES counter
    #     is reset.  Merge the old validation set into the training set.
    #     Wait until an episode boundary is reached to do this, so that the
    #     integrity of the buffer is preserved

    val_buffer = ReplayMemory(args,
                              args.val_buffer_capacity,
                              priority_weight=0,
                              priority_exponent=0,
                              images=True)

    val_transitions = get_current_policy_episodes(args, args.val_episodes, dqn,
                                                  encoder_trainer, encoder, 1.)
    old_val_transitions = []
    global val_losses
    val_losses = []
    for t in val_transitions:
        state, action, reward, terminal = t[1:]
        val_buffer.append(state, action, reward, not terminal)

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

    j = 1 if args.integrated_model else 0
    dqn.train()
    results_dir = os.path.join('results', args.id)
    metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}

    env_steps = args.initial_exp_steps + len(val_transitions)
    state, done = env.reset(), False
    while j * args.env_steps_per_epoch < args.total_steps:
        # Train encoder and forward model on real data
        if j != 0:
            # if args.integrated_model:
            #     if args.online_agent_training:
            #         dqn.update_target_net()
            #         encoder_trainer.update_target_net()
            #     encoder_trainer.train(real_transitions)
            if not args.integrated_model:
                forward_model.train(real_transitions)
        if env_steps % args.evaluation_interval == 0:
            dqn.eval()  # Set DQN (online network) to evaluation mode
            avg_reward = test(args, env_steps, dqn, encoder_trainer, encoder, metrics, results_dir, evaluate=True)  # Test
            log(env_steps, avg_reward)
            dqn.train()  # Set DQN (online network) back to training mode

        timestep, done = 0, True
        dqn.update_target_net()
        for e in range(args.env_steps_per_epoch):
            if done:
                state, done = env.reset(), False
                if len(old_val_transitions) > 0:
                    for t in old_val_transitions:
                        state, action, reward, terminal = t[1:]
                        real_transitions.append(state,
                                                action,
                                                reward,
                                                not terminal)

            # Take action in env acc. to current policy, and add to real_transitions
            real_z = encoder(state).view(-1)
            action = dqn.act_with_planner(real_z, encoder_trainer, length=args.planning_horizon,
                                          shots=args.planning_shots, epsilon=0.)
            next_state, reward, done = env.step(action)
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
            real_transitions.append(state, action, reward, done)
            transitions.append(Transition(timestep, state, action, reward, not done))
            state = next_state
            timestep = 0 if done else timestep + 1
            env_steps += 1

            # sample states from real_transitions
            all_zs = []
            all_actions = []
            all_rewards = []
            samples, actions, rewards = sample_real_transitions(transitions, args.num_model_rollouts)
            samples = samples.flatten(0, 1).to(args.device)
            H, N = args.history_length, args.num_model_rollouts
            with torch.no_grad():
                z = encoder(samples).view(H, N, -1)
            state_deque = deque(maxlen=4)
            for s in z.unbind():
                state_deque.append(s)

            for i in range(4):
                all_zs.append(z[i, :])
                all_actions.append(actions[:, i])
                all_rewards.append(rewards[:, i])

            # Perform k-step model rollout starting from s using current policy
            for k in range(args.rollout_length):
                z = torch.stack(list(state_deque))
                z = z.view(N, H, -1).view(N, -1)  # take a second look at this later
                actions = dqn.act(z, batch=True)
                with torch.no_grad():
                    next_z, rewards = forward_model.predict(z, actions)

                actions, rewards = actions.tolist(), rewards.tolist()
                state_deque.append(next_z)
                all_zs.append(next_z)
                all_actions.append(actions)
                all_rewards.append(rewards)

            # Add imagined data to model_transitions
            for i in range(N):
                for k in range(args.rollout_length+4):
                    priority = 0 if k < 3 else None
                    model_transitions.append(all_zs[k][i].unsqueeze(0),
                                             all_actions[k][i],
                                             all_rewards[k][i],
                                             False,
                                             timestep=k,
                                             init_priority=priority)

            if j >= 1:
                # Update policy parameters on model data
                for g in range(args.updates_per_step):
                    dqn.learn(model_transitions)

            if e % args.check_val_every == 0\
                    and not encoder_trainer.early_stopper.early_stop:
                with torch.no_grad():
                    encoder_trainer.encoder.eval()
                    encoder_trainer.prediction_module.eval()
                    encoder_trainer.reward_module.eval()
                    encoder_trainer.do_one_epoch(val_buffer)
                    encoder_trainer.encoder.train()
                    encoder_trainer.prediction_module.train()
                    encoder_trainer.reward_module.train()

            if not encoder_trainer.early_stopper.early_stop:
                encoder_trainer.do_one_epoch(real_transitions,
                                             iterations=args.model_updates_per_step)


        if j*args.env_steps_per_epoch % args.update_val_every == 0:
            val_losses = []
            old_val_transitions = val_transitions
            val_buffer = ReplayMemory(args,
                                      args.val_buffer_capacity,
                                      priority_weight=0,
                                      priority_exponent=0)
            encoder_trainer.log_results("val")

            val_transitions = get_current_policy_episodes(args,
                                                          args.val_episodes,
                                                          dqn,
                                                          encoder_trainer,
                                                          1.)
            env_steps += len(val_transitions)
            val_losses = []
            for t in val_transitions:
                state, action, reward, terminal = t[1:]
                val_buffer.append(state, action, reward, not terminal)
            encoder_trainer.reset_es()
            print("Encoder stopping has reset.")

        if j > 0:
            dqn.log(env_steps=env_steps)
        encoder_trainer.log_results("train")
        encoder_trainer.epochs_till_now += 1
        j += 1

def train_encoder(args,
                  transitions,
                  buffer,
                  num_actions,
                  val_eps=None,
                  init_epochs=None,
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

    trainer.train(buffer, val_eps, epochs=init_epochs)
    return encoder, trainer


def train_model(args, encoder, real_transitions, num_actions, init_epochs=None):
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
