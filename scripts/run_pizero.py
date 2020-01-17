from collections import deque

from src.mcts_memory import LocalBuffer
from src.model_trainer import TrainingWorker
from src.pizero import PiZero
from src.utils import get_args

import torch
import wandb
import numpy as np


def run_pizero(args):
    pizero = PiZero(args)
    env, mcts = pizero.env, pizero.mcts
    obs, env_steps = torch.from_numpy(env.reset()), 0
    training_worker = TrainingWorker(args, model=pizero.network)
    local_buf = LocalBuffer()
    eprets = np.zeros(args.num_envs, 'f')
    episode_rewards = deque(maxlen=10)
    history_buffer = [[]]*args.num_envs
    write_heads = list(range(args.num_envs))

    while env_steps < args.total_env_steps:
        # Run MCTS for the vectorized observation

        if len(history_buffer) > args.num_envs and args.reanalyze:
            new_samples = pizero.sample_for_reanalysis(history_buffer)
            obs = torch.cat([obs, new_samples[0]], 0)

        roots = mcts.batched_run(obs)

        actions, policy_probs, values = [], [], []
        for root in roots:
            # Select action for each obs
            action, p_logit = mcts.select_action(root)
            actions.append(action)
            policy_probs.append(p_logit.probs)
            values.append(root.value())
        actions = actions[:args.num_envs] # Cut out any reanalyzed actions.
        next_obs, reward, done, infos = env.step(actions)
        eprets += np.array(reward)
        for i in range(args.num_envs):
            if done[i]:
                episode_rewards.append(eprets[i])
                wandb.log({'Episode Reward': eprets[i], 'env_steps': env_steps})
                eprets[i] = 0
        next_obs = torch.from_numpy(next_obs)

        if args.reanalyze:
            # Still haven't concluded an episode
            if len(history_buffer) <= args.num_envs:
                # to preserve expectations for the buffer, just pad with
                # the current examples
                obs = torch.cat([obs]*5, 0)
                actions = actions * 5
                reward = np.concatenate([reward]*5, 0)
                done = np.concatenate([done]*5, 0)
                policy_probs = policy_probs * 5
                values = values * 5

            else:
                # Add the reanalyzed transitions to the real data.
                # Obs, policy_probs and values are already handled above.
                actions = actions + new_samples[1]
                reward = np.concatenate([reward, new_samples[2]], 0)
                done = np.concatenate([done, new_samples[3]], 0)

            for i in range(args.num_envs):
                history_buffer[write_heads[i]].append((obs[i, -1],
                                                       actions[i],
                                                       reward[i],
                                                       done[i],
                                                       ))

                # If this episode terminated, allocate a new slot.
                if done[i]:
                    history_buffer.append([])
                    write_heads[i] = len(history_buffer) - 1

        local_buf.append(obs,
                         torch.tensor(actions).float(),
                         torch.from_numpy(reward).float(),
                         torch.from_numpy(done).float(),
                         torch.stack(policy_probs).float(),
                         torch.stack(values).float().cpu())

        if env_steps % args.jumps == 0 and env_steps > 0:
            samples_to_buffer = training_worker.samples_to_buffer(*local_buf.stack())
            training_worker.buffer.append_samples(samples_to_buffer)
            local_buf.clear()

        if env_steps % args.training_interval == 0 and env_steps > 1000:
            training_worker.step()
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

        obs = next_obs
        env_steps += args.num_envs


if __name__ == '__main__':
    args = get_args()
    tags = []
    wandb.init(project=args.wandb_proj, entity="abs-world-models", tags=tags, config=vars(args))

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
    else:
        args.device = torch.device('cpu')

    run_pizero(args)
