"""
Parallel sampler version of Atari DQN.  Increasing the number of parallel
environmnets (sampler batch_B) should improve the efficiency of the forward
pass for action sampling on the GPU.  Using a larger batch size in the algorithm
should improve the efficiency of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.utils.logging.context import logger_context

import hydra
from omegaconf import DictConfig, OmegaConf
import omegaconf
import wandb
import torch
import numpy as np

# torch.autograd.set_detect_anomaly(True)
from src.models import SPRCatDqnModel
from src.rlpyt_framework import OneToOneSerialEvalCollector, SerialSampler, MinibatchRlEvalWandb
from src.algos import SPRCategoricalDQN
from src.agents import SPRAgent
from src.rlpyt_atari_env import AtariEnv


@hydra.main(config_name="config")
def main(args: DictConfig):
    args = OmegaConf.merge(configs['ernbw'], args)
    config = OmegaConf.to_container(args, resolve=True)
    print(OmegaConf.to_yaml(args, resolve=True))

    success = False
    while not success:
        try:
            if args.public:
                wandb.init(anonymous="allow", config=config, tags=[args.tag])
            else:
                wandb.init(project=args.project,
                           entity=args.entity,
                           config=config,
                           tags=[args.tag],
                           dir="./"
                           )
            success = True
        except Exception as e:
            print(e)
            success = False

    print(args.env.seed)
    seed = int(args.env.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    algo = SPRCategoricalDQN(OptimCls=torch.optim.AdamW,
                             optim_kwargs=config["optim"],
                             jumps=args.model.jumps,
                             **config["algo"])  # Run with defaults.
    agent = SPRAgent(ModelCls=SPRCatDqnModel, model_kwargs=config["model"], **config["agent"])
    sampler = SerialSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,  # default traj info + GameScore
        env_kwargs=config["env"],
        eval_env_kwargs=config["eval_env"],
        batch_T=config['sampler']['batch_T'],
        batch_B=config['sampler']['batch_B'],
        max_decorrelation_steps=0,
        eval_CollectorCls=OneToOneSerialEvalCollector,
        eval_n_envs=config["sampler"]["eval_n_envs"],
        eval_max_steps=config['sampler']['eval_max_steps'],
        eval_max_trajectories=config["sampler"]["eval_max_trajectories"],
    )
    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        affinity=dict(cuda_idx=args.cuda_idx),
        log_interval_steps=args.n_steps // args.num_logs,
        seed=seed,
        final_eval_only=args.final_eval_only,
    )
    config = dict(game=args.env.game)
    name = "dqn_" + args.env.game
    log_dir = "logs"
    with logger_context(log_dir, args.run_id, name, config, snapshot_mode="last", override_prefix=True):
        runner.train()

    quit()


if __name__ == "__main__":
    main()
