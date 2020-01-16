from collections import deque

from src.mcts_memory import LocalBuffer
from src.model_trainer import TrainingWorker
from src.pizero import PiZero
from src.utils import get_argparser

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

    while env_steps < args.total_env_steps:
        # Run MCTS for the vectorized observation
        roots = mcts.batched_run(obs)
        actions, policy_probs, values = [], [], []
        for root in roots:
            # Select action for each obs
            action, p_logit = mcts.select_action(root)
            actions.append(action)
            policy_probs.append(p_logit.probs)
            values.append(root.value())
        next_obs, reward, done, infos = env.step(actions)
        eprets += np.array(reward)
        for i in range(len(done)):
            if done[i]:
                episode_rewards.append(eprets[i])
                wandb.log({'Episode Reward': eprets[i], 'env_steps': env_steps})
                eprets[i] = 0
        next_obs = torch.from_numpy(next_obs)

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
            wandb.log({'Mean Reward': np.mean(episode_rewards), 'Median Reward': np.median(episode_rewards),
                       'env_steps': env_steps})

        if env_steps % args.evaluation_interval == 0 and env_steps > 0:
            avg_reward = pizero.evaluate()
            print('Env steps: {}, Avg_Reward: {}'.format(env_steps, avg_reward))
            wandb.log({'env_steps': env_steps, 'avg_reward': avg_reward})

        obs = next_obs
        env_steps += args.num_envs


def reanalyze(args, buffer):
    pass


if __name__ == '__main__':
    args = get_argparser().parse_args()
    tags = []
    wandb.init(project=args.wandb_proj, entity="abs-world-models", tags=tags, config=vars(args))

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
    else:
        args.device = torch.device('cpu')

    run_pizero(args)
