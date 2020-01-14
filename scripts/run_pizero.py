from src.mcts_memory import LocalBuffer
from src.model_trainer import TrainingWorker
from src.pizero import PiZero
from src.utils import get_argparser

import torch
import wandb


def run_pizero(args):
    pizero = PiZero(args)
    env, mcts = pizero.env, pizero.mcts
    obs, env_steps = torch.from_numpy(env.reset()), 0
    training_worker = TrainingWorker(args, model=pizero.network)
    local_buf = LocalBuffer()

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
        next_obs, reward, done, _ = env.step(actions)
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

        if env_steps % args.training_interval == 0 and env_steps > 100:
            training_worker.step()
            training_worker.log_results()

        if env_steps % args.evaluation_interval == 0 and env_steps > 0:
            avg_reward = pizero.evaluate()
            print('Env steps: {}, Avg_Reward: {}'.format(env_steps, avg_reward))
            wandb.log({'env_steps': env_steps, 'avg_reward': avg_reward})

        obs = next_obs
        env_steps += 1


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
