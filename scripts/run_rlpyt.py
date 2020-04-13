
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
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.minibatch_rl import MinibatchRl as MinibatchRl
from rlpyt.utils.launching.affinity import encode_affinity, make_affinity, quick_affinity_code, affinity_from_code
from rlpyt.utils.logging.context import logger_context
import wandb
import psutil

from src.rlpyt_models import MinibatchRlEvalWandb, AsyncRlEvalWandb, PizeroCatDqnModel, PizeroSearchCatDqnModel
from src.rlpyt_algos import PizeroCategoricalDQN, PizeroModelCategoricalDQN
from src.rlpyt_agents import DQNSearchAgent


def debug_build_and_train(game="pong", run_ID=0, cuda_idx=0, model=False, detach_model=1, args=None):
    config = configs['ernbw']
    config['runner']['log_interval_steps'] = 1e5
    config['env']['game'] = game
    config["eval_env"]["game"] = config["env"]["game"]
    config["algo"]["n_step_return"] = 5
    config["algo"]["prioritized_replay"] = True
    config["algo"]["min_steps_learn"] = 1e3
    config["sampler"]["eval_max_trajectories"] = 5
    config["sampler"]["eval_n_envs"] = 5
    config["sampler"]["batch_B"] = 128
    wandb.config.update(config)
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=16,
        max_decorrelation_steps=0,
    )
    args.discount = config["algo"]["discount"]
    if model:
        config["model"]["jumps"] = args.jumps
        config["model"]["detach_model"] = detach_model
        algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"], detach_model=detach_model)  # Run with defaults.
        agent = DQNSearchAgent(ModelCls=PizeroSearchCatDqnModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    else:
        algo = PizeroCategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariCatDqnAgent(ModelCls=PizeroCatDqnModel, model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()


def build_and_train(game="ms_pacman", run_ID=0, model=False, detach_model=1, args=None):
    affinity_dict = dict(
        n_cpu_core=10,
        n_gpu=4,
        async_sample=True,
        n_socket=1,
        gpu_per_run=3,
        sample_gpu_per_run=1,
        # alternating=True
    )
    affinity = make_affinity(**affinity_dict)

    if args.beluga:
        affinity = convert_affinity(affinity, psutil.Process().cpu_affinity())

    print(affinity)
    wandb.config.update(affinity_dict)
    config = configs['ernbw']
    config['runner']['log_interval_steps'] = 1e5
    config['env']['game'] = game
    config["eval_env"]["game"] = config["env"]["game"]
    config["algo"]["n_step_return"] = 5
    config["algo"]["batch_size"] = 256
    config["algo"]["learning_rate"] = 1.25e-4
    # config["sampler"]["max_decorrelation_steps"] = 0
    # config["algo"]["min_steps_learn"] = 2e4
    config['sampler']['batch_B'] = 16
    # config['sampler']['batch_T'] = 2
    config['sampler']['eval_n_envs'] = 8
    config["sampler"]["eval_max_steps"] = int(125e3)
    config["sampler"]["eval_max_trajectories"] = 8
    wandb.config.update(config)
    sampler = AsyncGpuSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=DbGpuResetCollector,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    args.discount = config["algo"]["discount"]
    if model:
        config["model"]["jumps"] = args.jumps
        config["model"]["detach_model"] = detach_model
        algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"], detach_model=detach_model)  # Run with defaults.
        agent = DQNSearchAgent(ModelCls=PizeroSearchCatDqnModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    else:
        algo = PizeroCategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariCatDqnAgent(ModelCls=PizeroCatDqnModel, model_kwargs=config["model"], **config["agent"])
    runner = AsyncRlEvalWandb(
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
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--model', action="store_true")
    parser.add_argument('--beluga', action="store_true")
    parser.add_argument('--jumps', type=int, default=4)
    parser.add_argument('--detach-model', type=int, default=1)
    parser.add_argument('--debug_cuda_idx', help='gpu to use ', type=int, default=0)
    # MCTS arguments
    parser.add_argument('--num-simulations', type=int, default=10)
    parser.add_argument('--eval-simulations', type=int, default=25)
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
    wandb.init(project='rlpyt', entity='abs-world-models')
    if args.debug:
        debug_build_and_train(game=args.game,
                              cuda_idx=args.debug_cuda_idx,
                              model=args.model,
                              detach_model=args.detach_model,
                              args=args,
                              )
    else:
        build_and_train(
            game=args.game,
            model=args.model,
            detach_model=args.detach_model,
            args=args,
        )

