from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.async_.collectors import DbGpuResetCollector
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuResetCollector
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.utils.launching.affinity import encode_affinity, make_affinity, quick_affinity_code, affinity_from_code
from rlpyt.utils.logging.context import logger_context
import wandb

from src.muzero import MuZeroAlgo, MuZeroAgent, MuZeroModel
from src.rlpyt_models import MinibatchRlEvalWandb, AsyncRlEvalWandb, PizeroCatDqnModel, PizeroSearchCatDqnModel, \
    SyncRlEvalWandb, SerialEvalCollectorFixed
from src.rlpyt_algos import PizeroCategoricalDQN, PizeroModelCategoricalDQN
from src.rlpyt_agents import DQNSearchAgent
from src.rlpyt_atari_env import AtariEnv


def build_and_train(game="ms_pacman", run_ID=0, args=None):
    affinity_dict = dict(
        n_cpu_core=args.n_gpu*4,
        n_gpu=args.n_gpu,
        n_socket=1,
        gpu_per_run=args.n_gpu,
    )
    samplerCls = GpuSampler
    collectorCls = GpuResetCollector
    runnerCls = SyncRlEvalWandb

    if args.async_sample:
        affinity_dict['async_sample'] = True
        affinity_dict['sample_gpu_per_run'] = 1
        affinity_dict['gpu_per_run'] = args.n_gpu - 1
        samplerCls = AsyncGpuSampler
        collectorCls = DbGpuResetCollector
        runnerCls = AsyncRlEvalWandb

    if args.n_gpu <= 1:
        runnerCls = MinibatchRlEvalWandb

    if args.n_gpu == 0:
        affinity = dict(cuda_idx=None)
        samplerCls = SerialSampler
    else:
        affinity = make_affinity(**affinity_dict)

    print(affinity)
    wandb.config.update(affinity_dict)

    config = configs['ernbw']
    config['env']['game'] = game
    config["env"]["stack_actions"] = args.stack_actions
    config["env"]["grayscale"] = args.grayscale
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["stack_actions"] = args.stack_actions
    config["eval_env"]["grayscale"] = args.grayscale
    config['env']['imagesize'] = args.imagesize
    config['eval_env']['imagesize'] = args.imagesize
    config["algo"]["batch_size"] = args.batch_size
    config['algo']['learning_rate'] = args.learning_rate
    config['algo']['replay_ratio'] = args.replay_ratio
    config['algo']['target_update_interval'] = args.target_update_interval
    config["algo"]["clip_grad_norm"] = args.max_grad_norm
    # config['optim']['eps'] = 0.00015
    config["sampler"]["eval_max_trajectories"] = 20
    config["sampler"]["eval_n_envs"] = config["sampler"]["eval_max_trajectories"]
    config["sampler"]["eval_max_steps"] = 625000  # int(125e3) / 4 * 50 (not actual max length, that's horizon)
    config['sampler']['batch_B'] = 4
    config["runner"]["log_interval_steps"] = 1e5
    wandb.config.update(config)
    sampler = samplerCls(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=collectorCls,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    args.discount = config["algo"]["discount"]
    config["model"] = {}
    config["model"]["imagesize"] = args.imagesize
    config["model"]["jumps"] = args.jumps
    config["model"]["dynamics_blocks"] = args.dynamics_blocks
    config["model"]["film"] = args.film
    config["model"]["stack_actions"] = args.stack_actions
    config["algo"]["reward_loss_weight"] = args.reward_lw
    config["algo"]["policy_loss_weight"] = args.policy_lw
    config["algo"]["value_loss_weight"] = args.value_lw
    algo = MuZeroAlgo(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"])  # Run with defaults.
    agent = MuZeroAgent(ModelCls=MuZeroModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    runner = runnerCls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        seed=args.seed,
        **config["runner"]
    )
    name = "dqn_" + game
    log_dir = "example"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='ms_pacman')
    parser.add_argument('--n-gpu', type=int, default=1)
    parser.add_argument('--async-sample', action="store_true")
    parser.add_argument('--stack-actions', type=int, default=0)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--grayscale', type=int, default=1)
    parser.add_argument('--imagesize', type=int, default=96)
    parser.add_argument('--jumps', type=int, default=5)
    parser.add_argument('--replay-ratio', type=int, default=2)
    parser.add_argument('--dynamics-blocks', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--norm-type', type=str, default='in', choices=["bn", "ln", "in", "none"], help='Normalization')
    parser.add_argument('--film', type=int, default=0)
    parser.add_argument('--nce', type=int, default=0)
    parser.add_argument('--noisy-nets', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--target-update-interval', type=float, default=2000, help='')
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    parser.add_argument('--reward-lw', type=float, default=1., help='Max Grad Norm')
    parser.add_argument('--policy-lw', type=float, default=1., help='Max Grad Norm')
    parser.add_argument('--value-lw', type=float, default=1., help='Max Grad Norm')
    # MCTS arguments
    parser.add_argument('--num-simulations', type=int, default=10)
    parser.add_argument('--eval-simulations', type=int, default=10)
    parser.add_argument('--latent-size', type=int, default=256)
    parser.add_argument('--search-epsilon', type=float, default=0.01, help='Epsilon for search e-greedy')
    parser.add_argument('--c1', type=float, default=1.25, help='UCB c1 constant')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.25, help='Root dirichlet alpha')
    parser.add_argument('--visit-temp', type=float, default=0.5, help='Visit counts softmax temperature for sampling actions')

    args = parser.parse_args()
    wandb.init(project='muzero', entity='abs-world-models', config=args, notes='')
    wandb.config.update(vars(args))
    build_and_train(
        game=args.game,
        args=args
    )

