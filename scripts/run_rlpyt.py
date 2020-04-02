
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
    wandb.config.update(config)
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=dict(game=game),
        eval_env_kwargs=dict(game=game),
        batch_T=4,  # Four time-steps per sampler iteration.
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
    )
    if model:
        algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], **config["algo"], detach_model=detach_model)  # Run with defaults.
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

def build_and_train(game="pong", run_ID=0, model=False, detach_model=1, args=None):
    affinity = make_affinity(
        n_cpu_core=4,
        n_gpu=2,
        async_sample=True,
        n_socket=1,
        gpu_per_run=1,
        sample_gpu_per_run=1
    )
    config = configs['ernbw']
    config['runner']['log_interval_steps'] = 1e5
    config['env']['game'] = game
    config["eval_env"]["game"] = config["env"]["game"]
    config["algo"]["n_step_return"] = 5
    wandb.config.update(config)
    sampler = AsyncGpuSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=DbGpuWaitResetCollector,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    if model:
        algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], **config["algo"], detach_model=detach_model)  # Run with defaults.
        agent = DQNSearchAgent(ModelCls=PizeroSearchCatDqnModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    else:
        algo = PizeroCategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariCatDqnAgent(ModelCls=PizeroCatDqnModel, model_kwargs=config["model"], **config["agent"])
    runner = AsyncRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "dqn_" + game
    log_dir = "example"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--model', action="store_true")
    parser.add_argument('--detach_model', type=int, default=1)
    parser.add_argument('--debug_cuda_idx', help='gpu to use ', type=int, default=0)
    # MCTS arguments
    parser.add_argument('--num-simulations', type=int, default=10)
    parser.add_argument('--eval-simulations', type=int, default=25)
    parser.add_argument('--virtual-threads', type=int, default=3)
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
    wandb.config.update()

