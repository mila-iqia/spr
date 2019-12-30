from src.model_trainer import TrainingWorker
from src.pizero import PiZero
from src.utils import get_argparser

import torch
import wandb


def run_pizero(args):
    pizero = PiZero(args)
    env, mcts = pizero.env, pizero.mcts
    obs, env_steps = env.reset(), 0
    training_worker = TrainingWorker(args, mcts)

    while env_steps < args.total_env_steps:
        root = mcts.run(obs)
        action, policy = mcts.select_action(root)
        next_obs, reward, done = env.step(action)
        training_worker.buffer.append(obs, action, reward, root.value(), policy, not done)

        # TODO: Train only after replay buffer reaches a certain capacity?
        if env_steps % args.training_interval == 0 and env_steps > 0:
            training_worker.train()
            training_worker.log_results()

        if env_steps % args.evaluation_interval == 0:
            avg_reward = pizero.evaluate()
            print('Env steps: {}, Avg_Reward: {}'.format(env_steps, avg_reward))
            wandb.log({'env_steps': env_steps, 'avg_reward': avg_reward})

        obs = next_obs
        env_steps += 1


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