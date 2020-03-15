
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
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.launching.affinity import encode_affinity, make_affinity, quick_affinity_code, affinity_from_code
from rlpyt.utils.logging.context import logger_context
import wandb

from src.rlpyt_models import MinibatchRlEvalWandb, AsyncRlEvalWandb


def build_and_train(game="pong", run_ID=0):
    affinity = make_affinity(
        n_cpu_core=12,
        n_gpu=4,
        async_sample=True,
        n_socket=1,
        gpu_per_run=3,
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
    algo = CategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
    agent = AtariCatDqnAgent(model_kwargs=config["model"], **config["agent"])
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
    args = parser.parse_args()
    wandb.init(project='rlpyt', entity='abs-world-models')
    build_and_train(
        game=args.game,
    )
    wandb.config.update()
