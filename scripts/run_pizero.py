from collections import deque
from multiprocessing import Queue

from src.async_mcts import AsyncMCTS
from src.mcts_memory import LocalBuffer
from src.model_trainer import TrainingWorker
from src.pizero import PiZero
from src.async_reanalyze import AsyncReanalyze
from src.utils import get_args

import os
import copy
import torch
import wandb
import numpy as np
from apex import amp
import time
import gym

from src.vectorized_mcts import VectorizedMCTS


def run_pizero(args):
    pizero = PiZero(args)

    env_steps = 0
    training_worker = TrainingWorker(args, model=pizero.network)

    if args.target_update_interval > 0:
        target_network = copy.deepcopy(pizero.network)
    else:
        target_network = pizero.network
    target_network.share_memory()

    if args.fp16:
        amp.initialize([pizero.network, target_network],
                       pizero.network.optimizer)
    if args.reanalyze:
        async_reanalyze = AsyncReanalyze(args, target_network)

    local_buf = LocalBuffer()
    eprets = np.zeros(args.num_envs, 'f')
    episode_rewards = deque(maxlen=10)
    wandb.log({'env_steps': 0})

    env = gym.vector.make('atari-v0', num_envs=args.num_envs, args=args,
                          asynchronous=not args.sync_envs)
    # TODO return int observations
    obs = env.reset()
    vectorized_mcts = VectorizedMCTS(args, env.action_space[0].n, args.num_envs, target_network)
    total_episodes = 0.
    total_train_steps = 0
    while env_steps < args.total_env_steps:
        obs = torch.from_numpy(obs)

        # Run MCTS for the vectorized observation
        actions, policies, values = vectorized_mcts.run(obs)
        policy_probs = policies.probs
        next_obs, reward, done, infos = env.step(actions.cpu().numpy())
        reward, done = torch.from_numpy(reward).float(), torch.from_numpy(done).float()
        obs, actions, reward, done, policy_probs, values = obs.cpu(), actions.cpu(), reward.cpu(),\
                                                           done.cpu(), policy_probs.cpu(), values.cpu()

        eprets += np.array(reward)

        for i in range(args.num_envs):
            if done[i]:
                episode_rewards.append(eprets[i])
                wandb.log({'Episode Reward': eprets[i],
                           'env_steps': env_steps})
                eprets[i] = 0

        if args.reanalyze:
            async_reanalyze.store_transitions(
                (obs[:, -1]*255).byte(),
                actions,
                reward,
                done,
            )

            total_episodes += torch.sum(done)

            # Add the reanalyzed transitions to the real data.
            new_samples = async_reanalyze.get_transitions(total_episodes)
            obs = torch.cat([obs, new_samples[0]], 0)
            actions = torch.cat([actions, new_samples[1]], 0)
            reward = torch.cat([reward, new_samples[2]], 0)
            done = torch.cat([done, new_samples[3]], 0)
            policy_probs = torch.cat([policy_probs, new_samples[4]], 0)
            values = torch.cat([values, new_samples[5]], 0)

        local_buf.append(obs, actions, reward, done, policy_probs, values)

        if env_steps % args.jumps == 0 and env_steps > 0:
            # Send transitions from the local buffer to the replay buffer
            samples_to_buffer = training_worker.samples_to_buffer(*local_buf.stack())
            training_worker.buffer.append_samples(samples_to_buffer)
            local_buf.clear()

        if env_steps % args.training_interval == 0 and env_steps > args.num_envs*20:
            target_train_steps = env_steps // args.training_interval
            steps = target_train_steps - total_train_steps
            training_worker.train(steps)  # TODO: Make this async
            training_worker.log_results()

            # Need to be careful when we check whether or not to reset:
            if (args.target_update_interval > 0 and
                (total_train_steps % args.target_update_interval >
                 target_train_steps % args.target_update_interval or
                 steps > args.target_update_interval)):

                target_network.load_state_dict(pizero.network.state_dict())
            total_train_steps = target_train_steps

        if env_steps % args.log_interval == 0 and len(episode_rewards) > 0:
            print('Env Steps: {}, Mean Reward: {}, Median Reward: {}'.format(env_steps, np.mean(episode_rewards),
                                                                             np.median(episode_rewards)))
            wandb.log({'Mean Reward': np.mean(episode_rewards), 'Median Reward': np.median(episode_rewards),
                       'env_steps': env_steps})

        if env_steps % args.evaluation_interval == 0 and env_steps > 0:
            avg_reward = pizero.evaluate()
            print('Env steps: {}, Avg_Reward: {}'.format(env_steps, avg_reward))
            wandb.log({'env_steps': env_steps, 'avg_reward': avg_reward})

        obs = next_obs
        env_steps += args.num_envs


if __name__ == '__main__':
    args = get_args()
    tags = []
    try:
        if len(args.name) == 0:
            run = wandb.init(project=args.wandb_proj,
                             entity="abs-world-models",
                             tags=tags,
                             config=vars(args))
        else:
            run = wandb.init(project=args.wandb_proj,
                             name=args.name,
                             entity="abs-world-models",
                             tags=tags,
                             config=vars(args))
    except wandb.run_manager.LaunchError as e:
        print(e)
        pass

    if len(args.savedir) == 0:
        args.savedir = os.environ["SLURM_TMPDIR"]
    args.savedir = "{}/{}".format(args.savedir, run.id)
    os.makedirs(args.savedir, exist_ok=True)

    print("Saving episode data in {}".format(args.savedir))

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
    else:
        args.device = torch.device('cpu')

    run_pizero(args)
