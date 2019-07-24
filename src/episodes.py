from itertools import chain

import torch
import numpy as np
from collections import deque

from aari.envs import make_vec_envs


def get_random_agent_episodes(args, device, steps):
    envs = make_vec_envs(args, args.num_processes)
    obs = envs.reset()
    episode_rewards = deque(maxlen=10)
    print('-------Collecting samples----------')
    episodes = [[[]] for _ in range(args.num_processes)]  # (n_processes * n_episodes * episode_len)
    actions = [[[]] for _ in range(args.num_processes)]
    for step in range(steps // args.num_processes):
        # Take action using a random policy
        action = torch.tensor(
            np.array([np.random.randint(envs.action_space.n) for _ in range(args.num_processes)])) \
            .unsqueeze(dim=1).to(device)
        obs, reward, done, infos = envs.step(action)
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

            if done[i] != 1:
                episodes[i][-1].append((obs[i].clone(), action[i], reward[i].clone()))
            else:
                episodes[i].append([(obs[i].clone(), action[i].clone(), reward[i].clone())])

    # Convert to 2d list from 3d list
    episodes = list(chain.from_iterable(episodes))
    envs.close()

    inds = np.arange(len(episodes))
    rng = np.random.RandomState(seed=args.seed)
    rng.shuffle(inds)
    val_split_ind = int(0.9 * len(inds))
    tr_eps, val_eps = episodes[:val_split_ind], episodes[val_split_ind:]

    return tr_eps, val_eps
