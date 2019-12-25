from src.model_trainer import WorkerPolicy
from src.pizero import PiZero
from src.utils import get_argparser

import torch
import wandb


def run_pizero(args):
    pizero = PiZero(args)
    env, mcts = pizero.env, pizero.mcts
    obs, env_steps = env.reset(), 0
    training_worker = WorkerPolicy(args, mcts)

    while env_steps < args.total_env_steps:
        mcts.run(obs)
        action = mcts.select_action()
        next_obs, reward, done = env.step(action)
        training_worker.buffer.append(obs, action, next_obs, reward, not done)

        if env_steps % args.training_interval == 0:
            training_worker.train()
        obs = next_obs


if __name__ == '__main__':
    wandb.init()
    args = get_argparser().parse_args()
    tags = []
    wandb.init(project=args.wandb_proj, entity="abs-world-models", tags=tags, config=vars(args))

    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')

    run_pizero(args)