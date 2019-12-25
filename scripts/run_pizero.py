from src.model_trainer import WorkerPolicy
from src.pizero import PiZero


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
    args = None  # TODO: create a separate argparser
    run_pizero(args)