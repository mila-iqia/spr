from collections import deque
from multiprocessing import Queue

from src.async_mcts import AsyncMCTS
from src.mcts_memory import LocalBuffer
from src.model_trainer import TrainingWorker
from src.pizero import PiZero, ReanalyzeWorker
from src.utils import get_args

import os
import torch
import wandb
import numpy as np


def run_pizero(args):
    pizero = PiZero(args)
    async_mcts = AsyncMCTS(args, pizero.network)
    env_steps = 0

    sample_queue = Queue()
    reanalyze_queue = Queue()
    reanalyze_worker = ReanalyzeWorker(args,
                                       sample_queue,
                                       reanalyze_queue)
    reanalyze_worker.start()
    training_worker = TrainingWorker(args, model=pizero.network)
    local_buf = LocalBuffer()
    eprets = np.zeros(args.num_envs, 'f')
    episode_rewards = deque(maxlen=10)
    wandb.log({'env_steps': 0})

    total_episodes = 0
    if args.profile:
        import pprofile
        prof = pprofile.Profile()
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    while env_steps < args.total_env_steps:
        # Run MCTS for the vectorized observation
        obs, actions, reward, done, policy_probs, values = async_mcts.run_mcts()

        if total_episodes > 0 and args.reanalyze:
            new_samples = reanalyze_queue.get()
            obs = torch.cat([obs, new_samples[0]], 0)  # Append reanalyze candidates to new observations

        actions = actions[:args.num_envs]  # Cut out any reanalyzed actions.

        eprets += np.array(reward)

        for i in range(args.num_envs):
            if done[i]:
                episode_rewards.append(eprets[i])
                wandb.log({'Episode Reward': eprets[i],
                           'env_steps': env_steps,
                           "Reanalyze queue length:": reanalyze_queue.qsize()})
                eprets[i] = 0

        if args.reanalyze:
            # Still haven't concluded an episode
            if total_episodes == 0:
                # to preserve expectations for the buffer, just pad with
                # the current examples
                obs = torch.cat([obs]*5, 0)
                obs[args.num_envs:] = 0
                actions = actions + [0]*(4*len(actions))
                reward = np.concatenate([reward]*5, 0)
                reward[args.num_envs:] = 0
                done = np.concatenate([done]*5, 0)
                done[args.num_envs:] = 0
                policy_probs = policy_probs * 5
                done[args.num_envs:] = 0
                values = values + [torch.zeros_like(values[0])]*(4*len(values))

            else:
                # Add the reanalyzed transitions to the real data.
                # Obs, policy_probs and values are already handled above.
                actions = actions + new_samples[1]
                reward = np.concatenate([reward, new_samples[2]], 0)
                done = np.concatenate([done, new_samples[3]], 0)

            sample_queue.put(((obs[:args.num_envs, -1]*255).byte(),
                               actions[:args.num_envs],
                               reward[:args.num_envs],
                               done[:args.num_envs]))

            total_episodes += np.sum(done[:args.num_envs])

        local_buf.append(obs,
                         actions, reward, done, policy_probs, values)

        if env_steps % args.jumps == 0 and env_steps > 0:
            # Send transitions from the local buffer to the replay buffer
            samples_to_buffer = training_worker.samples_to_buffer(*local_buf.stack())
            training_worker.buffer.append_samples(samples_to_buffer)
            local_buf.clear()

        if env_steps % args.training_interval == 0 and env_steps > 400:
            training_worker.step()  # TODO: Make this async, and add ability to take multiple steps here
            training_worker.log_results()

        if env_steps % args.log_interval == 0 and len(episode_rewards) > 0:
            print('Env Steps: {}, Mean Reward: {}, Median Reward: {}'.format(env_steps, np.mean(episode_rewards),
                                                                             np.median(episode_rewards)))
            wandb.log({'Mean Reward': np.mean(episode_rewards), 'Median Reward': np.median(episode_rewards),
                       'env_steps': env_steps})

        if env_steps % args.evaluation_interval == 0 and env_steps > 0:
            avg_reward = pizero.evaluate()
            print('Env steps: {}, Avg_Reward: {}'.format(env_steps, avg_reward))
            wandb.log({'env_steps': env_steps, 'avg_reward': avg_reward})

        env_steps += args.num_envs

    if args.profile:
        prof.dump_stats("profile.out")


if __name__ == '__main__':
    args = get_args()
    tags = []
    if len(args.name) == 0:
        run = wandb.init(project=args.wandb_proj,
                         entity="abs-world-models",
                         tags=tags,
                         config=vars(args))
    else:
        run = wandb.init(project=args.wandb_proj,
                         name=args.name,
                         entity="abs-world-models",
                         tags=tags, config=vars(args))

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
