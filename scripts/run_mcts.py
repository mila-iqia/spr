import wandb
import os

from rlpyt.utils.logging.context import logger_context
from src.rlpyt_atari_env import AtariEnv, AtariTrajInfo
from src.rlpyt_models import MinibatchRlEvalWandb
from src.sampler import SerialSampler, OneToOneSerialEvalCollector, SerialEvalCollector


def run_mcts(args=None):
    env = AtariEnv
    config = dict(
        agent=dict(),
        algo=dict(
            discount=args.discount,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            prioritized_replay=args.prioritized_replay,
            n_step_return=args.n_step,
            replay_size=int(1e6),
            target_update_interval=args.target_update_interval,
            target_update_tau=args.target_update_tau
        ),
        env=dict(
            game=args.game,
            episodic_lives=True,
        ),
        eval_env=dict(
            game=args.game,
            episodic_lives=False,
            horizon=int(27e3),
        ),
        model=dict(),
        optim=dict(),
        runner=dict(
            n_steps=args.n_steps,
            log_interval_steps=1e5,
        ),
        sampler=dict(
            batch_T=args.batch_t,
            batch_B=args.batch_b,
            max_decorrelation_steps=0,
            eval_n_envs=100,
            eval_max_steps=100*28000,
            eval_max_trajectories=100
        ),
    )

    sampler = SerialSampler(
        EnvCls=env,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=config["env"],
        eval_env_kwargs=config["eval_env"],
        batch_T=config['sampler']['batch_T'],
        batch_B=config['sampler']['batch_B'],
        max_decorrelation_steps=0,
        eval_CollectorCls=OneToOneSerialEvalCollector if args.fasteval else SerialEvalCollector,
        eval_n_envs=config["sampler"]["eval_n_envs"],
        eval_max_steps=config['sampler']['eval_max_steps'],
        eval_max_trajectories=config["sampler"]["eval_max_trajectories"],
    )

    # TODO: Implement new simpler algo, agnet and model classes
    algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"])
    agent = DQNSearchAgent(ModelCls=PizeroSearchCatDqnModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    wandb.config.update(config)
    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        affinity=dict(cuda_idx=0),
        log_interval_steps=args.n_steps//args.num_logs,
        seed=args.seed,
        final_eval_only=args.final_eval_only,
    )
    config = dict(game=args.game)
    name = "dqn_" + args.game
    log_dir = "example_1"
    with logger_context(log_dir, 0, name, config, snapshot_mode="last"):
        runner.train()

    # See if this actually makes runs release Beluga nodes after they're done training.
    quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='ms_pacman')
    parser.add_argument('--fasteval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--n-steps', type=int, default=100000)
    parser.add_argument('--target-update-interval', type=int, default=2000)
    parser.add_argument('--target-update-tau', type=float, default=1.)
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--beluga', action="store_true")

    # MCTS arguments
    parser.add_argument('--num-simulations', type=int, default=0)
    parser.add_argument('--eval-simulations', type=int, default=0)
    parser.add_argument('--c1', type=float, default=1.25, help='UCB c1 constant')

    args = parser.parse_args()

    if args.beluga:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(project='mcts', entity='abs-world-models', config=args, tags=[args.tag], dir=args.wandb_dir)
    wandb.config.update(vars(args))
    run_mcts(args)