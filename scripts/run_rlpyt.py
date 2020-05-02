
"""
Parallel sampler version of Atari DQN.  Increasing the number of parallel
environmnets (sampler batch_B) should improve the efficiency of the forward
pass for action sampling on the GPU.  Using a larger batch size in the algorithm
should improve the efficiency of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDqnAgent
from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.async_.collectors import DbGpuResetCollector, DbGpuWaitResetCollector
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuResetCollector
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.minibatch_rl import MinibatchRl as MinibatchRl
from rlpyt.utils.launching.affinity import encode_affinity, make_affinity, quick_affinity_code, affinity_from_code
from rlpyt.utils.logging.context import logger_context
import wandb
import psutil

from src.rlpyt_models import MinibatchRlEvalWandb, AsyncRlEvalWandb, PizeroCatDqnModel, PizeroSearchCatDqnModel, \
    SyncRlEvalWandb, SerialEvalCollectorFixed
from src.rlpyt_algos import PizeroCategoricalDQN, PizeroModelCategoricalDQN
from src.rlpyt_agents import DQNSearchAgent
from src.rlpyt_atari_env import AtariEnv


def debug_build_and_train(game="pong", run_ID=0, cuda_idx=0, model=False, detach_model=1, args=None):
    config = configs['ernbw']
    config['env']['game'] = game
    config["env"]["stack_actions"] = args.stack_actions
    config["env"]["grayscale"] = args.grayscale
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["stack_actions"] = args.stack_actions
    config["eval_env"]["grayscale"] = args.grayscale
    config['env']['imagesize'] = args.imagesize
    config['eval_env']['imagesize'] = args.imagesize
    config["model"]["dueling"] = True
    config["algo"]["min_steps_learn"] = 1000
    config["algo"]["n_step_return"] = 20
    config["algo"]["batch_size"] = 64
    config["algo"]["learning_rate"] = 0.0001
    config['algo']['replay_ratio'] = args.replay_ratio
    config['algo']['target_update_interval'] = 2000
    config['algo']['eps_steps'] = int(5e4)
    config["algo"]["clip_grad_norm"] = args.max_grad_norm
    config['algo']['pri_alpha'] = 0.5
    config['algo']['pri_beta_steps'] = int(10e4)
    # config['optim']['eps'] = 0.00015
    config["sampler"]["eval_max_trajectories"] = 20
    config["sampler"]["eval_n_envs"] = 20
    config["sampler"]["eval_max_steps"] = 625000  # int(125e3) / 4 * 50 (not actual max length, that's horizon)
    if args.noisy_nets:
        config['agent']['eps_init'] = 0.
        config['agent']['eps_final'] = 0.
        config['agent']['eps_eval'] = 0.
    wandb.config.update(config)
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=config["env"],
        eval_env_kwargs=dict(game=game),
        batch_T=1,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_CollectorCls=SerialEvalCollectorFixed,
        eval_n_envs=config["sampler"]["eval_n_envs"],
        eval_max_steps=config['sampler']['eval_max_steps'],
        eval_max_trajectories=config["sampler"]["eval_max_trajectories"],
    )
    args.discount = config["algo"]["discount"]
    if model:
        config["model"]["jumps"] = args.jumps
        config["model"]["detach_model"] = detach_model
        config["model"]["dynamics_blocks"] = args.dynamics_blocks
        config["model"]["film"] = args.film
        config["model"]["norm_type"] = args.norm_type
        config["model"]["nce"] = args.nce
        config["model"]["nce_type"] = args.nce_type
        config["model"]["encoder"] = args.encoder
        config["model"]["augmentation"] = args.augmentation
        config["model"]["aug_prob"] = args.aug_prob
        config["model"]["target_augmentation"] = args.target_augmentation
        config["model"]["eval_augmentation"] = args.eval_augmentation
        config["model"]["stack_actions"] = args.stack_actions
        algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"], detach_model=detach_model)  # Run with defaults.
        agent = DQNSearchAgent(ModelCls=PizeroSearchCatDqnModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    else:
        algo = PizeroCategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariCatDqnAgent(ModelCls=PizeroCatDqnModel, model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=10e4,
        log_interval_steps=1e4,
        affinity=dict(cuda_idx=cuda_idx),
        seed=69
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


def build_and_train(game="ms_pacman", run_ID=0, model=False, detach_model=1, args=None):
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

    if args.n_gpu == 1:
        runnerCls = MinibatchRlEvalWandb

    affinity = make_affinity(**affinity_dict)

    if args.beluga:
        affinity = convert_affinity(affinity, psutil.Process().cpu_affinity())

    print(affinity)
    wandb.config.update(affinity_dict)
    config = configs['ernbw']
    config['runner']['log_interval_steps'] = 1e5
    config['env']['game'] = game
    config["env"]["stack_actions"] = args.stack_actions
    config["env"]["grayscale"] = args.grayscale
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["stack_actions"] = args.stack_actions
    config["eval_env"]["grayscale"] = args.grayscale
    config["algo"]["n_step_return"] = 5
    config["algo"]["batch_size"] = 128
    config["algo"]["learning_rate"] = 6.25e-5
    config['algo']['replay_ratio'] = args.replay_ratio
    # config["sampler"]["max_decorrelation_steps"] = 0
    # config["algo"]["min_steps_learn"] = 2e4
    config['sampler']['batch_B'] = 32
    # config['sampler']['batch_T'] = 2
    config['sampler']['eval_n_envs'] = config["sampler"]["eval_max_trajectories"] = 8
    config["sampler"]["eval_max_steps"] = int(125e3)
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
    if model:
        config["model"]["jumps"] = args.jumps
        config["model"]["detach_model"] = detach_model
        config["model"]["dynamics_blocks"] = args.dynamics_blocks
        config["model"]["film"] = args.film
        config["model"]["nce"] = args.nce
        config["model"]["encoder"] = args.encoder
        config["model"]["nce_type"] = args.nce_type
        config["model"]["norm_type"] = args.norm_type
        config["model"]["augmentation"] = args.augmentation
        config["model"]["aug_prob"] = args.aug_prob
        config["model"]["target_augmentation"] = args.target_augmentation
        config["model"]["eval_augmentation"] = args.eval_augmentation
        config["model"]["stack_actions"] = args.stack_actions
        algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"], detach_model=detach_model)  # Run with defaults.
        agent = DQNSearchAgent(ModelCls=PizeroSearchCatDqnModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    else:
        algo = PizeroCategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariCatDqnAgent(ModelCls=PizeroCatDqnModel, model_kwargs=config["model"], **config["agent"])
    runner = runnerCls(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        seed=42,
        **config["runner"]
    )
    name = "dqn_" + game
    log_dir = "example"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


def convert_affinity(affinity, cpus):
    affinity.all_cpus = cpus[:len(affinity.all_cpus)]
    cpu_tracker = 0
    for optimizer in affinity.optimizer:
        cpus_to_alloc = len(optimizer["cpus"])
        optimizer["cpus"] = cpus[cpu_tracker:cpu_tracker+cpus_to_alloc]
        cpu_tracker = cpu_tracker + cpus_to_alloc
    for sampler in affinity.sampler:
        cpus_to_alloc = len(sampler["all_cpus"])
        sampler["all_cpus"] = cpus[cpu_tracker:cpu_tracker+cpus_to_alloc]
        sampler["master_cpus"] = cpus[cpu_tracker:cpu_tracker+cpus_to_alloc]
        new_workers_cpus = []
        for worker in sampler["workers_cpus"]:
            cpus_to_alloc = len(worker)
            worker = cpus[cpu_tracker:cpu_tracker+cpus_to_alloc]
            cpu_tracker = cpu_tracker + cpus_to_alloc
            new_workers_cpus.append(worker)

        sampler["workers_cpus"] = tuple(new_workers_cpus)

    return affinity


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='ms_pacman')
    parser.add_argument('--n-gpu', type=int, default=4)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--async-sample', action="store_true")
    parser.add_argument('--learn-model', action="store_true")
    parser.add_argument('--stack-actions', type=int, default=0)
    parser.add_argument('--grayscale', type=int, default=1)
    parser.add_argument('--imagesize', type=int, default=84)
    parser.add_argument('--beluga', action="store_true")
    parser.add_argument('--jumps', type=int, default=4)
    parser.add_argument('--replay-ratio', type=int, default=2)
    parser.add_argument('--dynamics-blocks', type=int, default=16)
    parser.add_argument('--norm-type', type=str, default='bn', choices=["bn", "ln", "in", "none"], help='Normalization')
    parser.add_argument('--encoder', type=str, default='repnet', choices=["repnet", "curl", "midsize"], help='Normalization')
    parser.add_argument('--aug-prob', type=float, default=0.9, help='Probability to apply augmentation')
    parser.add_argument('--film', type=int, default=0)
    parser.add_argument('--nce', type=int, default=0)
    parser.add_argument('--noisy-nets', type=int, default=0)
    parser.add_argument('--nce-type', type=str, default='stdim', choices=["stdim", "moco"], help='Style of NCE')
    parser.add_argument('--augmentation', type=str, default='none', choices=["none", "rrc", "affine", "crop"], help='Style of augmentation')
    parser.add_argument('--target-augmentation', type=int, default=0, help='Use augmentation on inputs to target networks')
    parser.add_argument('--eval-augmentation', type=int, default=0, help='Use augmentation on inputs at evaluation time')
    parser.add_argument('--detach-model', type=int, default=1)
    parser.add_argument('--debug_cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    # MCTS arguments
    parser.add_argument('--num-simulations', type=int, default=10)
    parser.add_argument('--eval-simulations', type=int, default=20)
    parser.add_argument('--latent-size', type=int, default=256)
    parser.add_argument('--virtual-threads', type=int, default=3)
    parser.add_argument('--q-dirichlet', type=int, default=0)
    parser.add_argument('--no-search-control', type=int, default=0, help='Do search to adjust replay ratio etc, but take actions based on original q values')
    parser.add_argument('--search-epsilon', type=float, default=0.01, help='Epsilon for search e-greedy')
    parser.add_argument('--virtual-loss-c', type=int, default=1.)
    parser.add_argument('--c1', type=float, default=1.25, help='UCB c1 constant')
    parser.add_argument('--dirichlet-alpha', type=float, default=0.25, help='Root dirichlet alpha')
    parser.add_argument('--visit-temp', type=float, default=0.5, help='Visit counts softmax temperature for sampling actions')

    args = parser.parse_args()
    wandb.init(project='rlpyt', entity='abs-world-models', config=args)
    wandb.config.update(vars(args))
    if args.debug:
        debug_build_and_train(game=args.game,
                              cuda_idx=args.debug_cuda_idx,
                              model=args.learn_model,
                              detach_model=args.detach_model,
                              args=args,
                              )
    else:
        build_and_train(
            game=args.game,
            model=args.learn_model,
            detach_model=args.detach_model,
            args=args,
        )

