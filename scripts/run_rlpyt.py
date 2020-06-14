
"""
Parallel sampler version of Atari DQN.  Increasing the number of parallel
environmnets (sampler batch_B) should improve the efficiency of the forward
pass for action sampling on the GPU.  Using a larger batch size in the algorithm
should improve the efficiency of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDqnAgent
# from rlpyt.algos.dqn.cat_dqn import CategoricalDQN
# from rlpyt.models.dqn.atari_catdqn_model import AtariCatDqnModel
from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs
from rlpyt.samplers.async_.collectors import DbGpuResetCollector, DbGpuWaitResetCollector
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuResetCollector, GpuEvalCollector
from rlpyt.envs.atari.atari_env import AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.minibatch_rl import MinibatchRl as MinibatchRl
from rlpyt.utils.launching.affinity import encode_affinity, make_affinity, quick_affinity_code, affinity_from_code
from rlpyt.utils.logging.context import logger_context
import wandb
import os
import psutil

from src.rlpyt_models import MinibatchRlEvalWandb, AsyncRlEvalWandb, PizeroCatDqnModel, PizeroSearchCatDqnModel, \
    SyncRlEvalWandb
from src.sampler import OneToOneSerialEvalCollector, OneToOneGpuEvalCollector, SerialEvalCollector
from src.rlpyt_algos import PizeroCategoricalDQN, PizeroModelCategoricalDQN
from src.rlpyt_agents import DQNSearchAgent
from src.rlpyt_atari_env import AtariEnv
from src.rlpyt_control_model import AtariCatDqnModel, CategoricalDQN

import torch
import numpy as np

def debug_build_and_train(game="pong", run_ID=0, cuda_idx=0, model=False, detach_model=1, args=None, control=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = configs['ernbw']
    config['env']['game'] = game
    config["env"]["stack_actions"] = args.stack_actions
    config["env"]["grayscale"] = args.grayscale
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["stack_actions"] = args.stack_actions
    config["eval_env"]["grayscale"] = args.grayscale
    config['env']['imagesize'] = args.imagesize
    config['eval_env']['imagesize'] = args.eval_imagesize
    config["model"]["dueling"] = bool(args.dueling)
    config["algo"]["min_steps_learn"] = 2000
    config["algo"]["n_step_return"] = args.n_step
    config["algo"]["batch_size"] = args.batch_size
    config["algo"]["learning_rate"] = 0.0001
    config['algo']['replay_ratio'] = args.replay_ratio
    config['algo']['target_update_interval'] = args.target_update_interval
    config['algo']['target_update_tau'] = args.target_update_tau
    config['algo']['eps_steps'] = args.eps_steps
    config["algo"]["clip_grad_norm"] = args.max_grad_norm
    config['algo']['pri_alpha'] = 0.5
    config['algo']['pri_beta_steps'] = int(10e4)
    config['optim']['eps'] = 0.00015
    config["sampler"]["eval_max_trajectories"] = 100
    config["sampler"]["eval_n_envs"] = 100
    config["sampler"]["eval_max_steps"] = 100*28000
    config['sampler']['batch_B'] = args.batch_b
    config['sampler']['batch_T'] = args.batch_t
    if args.noisy_nets:
        config['agent']['eps_init'] = 0.
        config['agent']['eps_final'] = 0.
        config['agent']['eps_eval'] = 0.001
    wandb.config.update(config)
    sampler = SerialSampler(
        EnvCls=AtariEnv,
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
    args.discount = config["algo"]["discount"]
    if model:
        config["model"]["imagesize"] = args.imagesize
        config["model"]["jumps"] = args.jumps
        config["model"]["detach_model"] = detach_model
        config["model"]["dynamics_blocks"] = args.dynamics_blocks
        config["model"]["film"] = args.film
        config["model"]["nce"] = args.nce
        config["model"]["encoder"] = args.encoder
        config["model"]["transition_model"] = args.transition_model
        config["model"]["padding"] = args.padding
        config["model"]["noisy_nets"] = args.noisy_nets
        config["model"]["momentum_encoder"] = args.momentum_encoder
        config["model"]["target_encoder_sn"] = args.target_encoder_sn
        config["model"]["shared_encoder"] = args.shared_encoder
        config["model"]["local_nce"] = args.local_nce
        config["model"]["global_nce"] = args.global_nce
        config["model"]["hard_neg_factor"] = args.hard_neg_factor
        config["model"]["distributional"] = args.distributional
        config["model"]["use_all_targets"] = args.use_all_targets
        config["model"]["grad_scale_factor"] = args.grad_scale_factor
        config["model"]["global_local_nce"] = args.global_local_nce
        config["model"]["buffered_nce"] = args.buffered_nce
        config["model"]["cosine_nce"] = args.cosine_nce
        config["model"]["norm_type"] = args.norm_type
        config["model"]["augmentation"] = args.augmentation
        config["model"]["frame_dropout"] = args.frame_dropout
        config["model"]["keep_last_frame"] = args.keep_last_frame
        config["model"]["time_contrastive"] = args.time_contrastive
        config["model"]["aug_prob"] = args.aug_prob
        config["model"]["target_augmentation"] = args.target_augmentation
        config["model"]["no_rl_augmentation"] = args.no_rl_augmentation
        config["model"]["eval_augmentation"] = args.eval_augmentation
        config["model"]["stack_actions"] = args.stack_actions
        config["model"]["classifier"] = args.classifier
        config["algo"]["model_rl_weight"] = args.model_rl_weight
        config["algo"]["reward_loss_weight"] = args.reward_loss_weight
        config["algo"]["model_nce_weight"] = args.model_nce_weight
        config["algo"]["nce_loss_weight"] = args.nce_loss_weight
        config["algo"]["nce_loss_weight_final"] = args.nce_loss_weight_final
        config["algo"]["nce_loss_decay_steps"] = args.nce_loss_decay_steps
        config["algo"]["amortization_loss_weight"] = args.amortization_loss_weight
        config["algo"]["amortization_decay_constant"] = args.amortization_decay_constant
        config["algo"]["time_contrastive"] = args.time_contrastive
        config["algo"]["distributional"] = args.distributional
        config["algo"]["prioritized_replay"] = args.prioritized_replay
        algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"], detach_model=detach_model)  # Run with defaults.
        agent = DQNSearchAgent(ModelCls=PizeroSearchCatDqnModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    elif control:
        algo = CategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariCatDqnAgent(ModelCls=AtariCatDqnModel, model_kwargs=config["model"], **config["agent"])
    else:
        algo = Dqn(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariDqnAgent(ModelCls=PizeroCatDqnModel, model_kwargs=config["model"], **config["agent"])

    runner = MinibatchRlEvalWandb(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=args.n_steps,
        affinity=dict(cuda_idx=cuda_idx),
        log_interval_steps=args.n_steps//10,
        seed=args.seed,
        final_eval_only=args.final_eval_only,
    )
    config = dict(game=game)
    name = "dqn_" + game
    log_dir = "example_1"
    with logger_context(log_dir, run_ID, name, config, snapshot_mode="last"):
        runner.train()

    # See if this actually makes runs release Beluga nodes after they're done training.
    quit()


def build_and_train(game="ms_pacman", run_ID=0, model=False,
                    detach_model=1, args=None, control=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    affinity_dict = dict(
        n_cpu_core=args.n_gpu*4,
        n_gpu=args.n_gpu,
        n_socket=1,
        gpu_per_run=args.n_gpu,
    )
    samplerCls = GpuSampler
    collectorCls = GpuResetCollector
    runnerCls = SyncRlEvalWandb
    eval_CollectorCls = GpuEvalCollector

    if args.async_sample:
        affinity_dict['async_sample'] = True
        affinity_dict['sample_gpu_per_run'] = 1
        affinity_dict['gpu_per_run'] = args.n_gpu - 1
        samplerCls = AsyncGpuSampler
        collectorCls = DbGpuResetCollector
        runnerCls = AsyncRlEvalWandb

    if args.n_gpu <= 1:
        runnerCls = MinibatchRlEvalWandb
        samplerCls = SerialSampler
        eval_CollectorCls = SerialEvalCollector

    if args.n_gpu == 0:
        affinity = dict(cuda_idx=None)
    else:
        affinity = make_affinity(**affinity_dict)

    if args.beluga:
        affinity = convert_affinity(affinity, psutil.Process().cpu_affinity())

    print(affinity)
    wandb.config.update(affinity_dict)

    config = configs['double']
    config['env']['game'] = game
    config["env"]["stack_actions"] = args.stack_actions
    config["env"]["grayscale"] = args.grayscale
    config['env']['imagesize'] = args.imagesize
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["stack_actions"] = args.stack_actions
    config["eval_env"]["grayscale"] = args.grayscale
    config['eval_env']['imagesize'] = args.eval_imagesize
    config["model"]["dueling"] = bool(args.dueling)
    config["algo"]["batch_size"] = args.batch_size
    config['algo']['replay_ratio'] = args.replay_ratio
    config['algo']['target_update_interval'] = args.target_update_interval
    config["algo"]["clip_grad_norm"] = args.max_grad_norm
    config["algo"]["n_step_return"] = args.n_step
    config['algo']['pri_alpha'] = 0.5
    config['algo']['pri_beta_steps'] = args.n_steps
    # config['optim']['eps'] = 0.00015
    config["sampler"]["eval_max_trajectories"] = 100
    config["sampler"]["eval_n_envs"] = config["sampler"]["eval_max_trajectories"]
    config["sampler"]["eval_max_steps"] = 100*28000
    config['sampler']['batch_B'] = args.batch_b
    config['sampler']['batch_T'] = args.batch_t
    config["runner"]["log_interval_steps"] = 1e6
    if args.noisy_nets:
        config['agent']['eps_init'] = 0.
        config['agent']['eps_final'] = 0.
        config['agent']['eps_eval'] = 0.001
    wandb.config.update(config)
    sampler = samplerCls(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=collectorCls,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["eval_env"],
        eval_CollectorCls=eval_CollectorCls,
        **config["sampler"]
    )
    args.discount = config["algo"]["discount"]
    if model:
        config["model"]["imagesize"] = args.imagesize
        config["model"]["jumps"] = args.jumps
        config["model"]["detach_model"] = detach_model
        config["model"]["dynamics_blocks"] = args.dynamics_blocks
        config["model"]["film"] = args.film
        config["model"]["nce"] = args.nce
        config["model"]["distributional"] = args.distributional
        config["model"]["encoder"] = args.encoder
        config["model"]["transition_model"] = args.transition_model
        config["model"]["norm_type"] = args.norm_type
        config["model"]["momentum_encoder"] = args.momentum_encoder
        config["model"]["target_encoder_sn"] = args.target_encoder_sn
        config["model"]["shared_encoder"] = args.shared_encoder
        config["model"]["local_nce"] = args.local_nce
        config["model"]["noisy_nets"] = args.noisy_nets
        config["model"]["global_nce"] = args.global_nce
        config["model"]["hard_neg_factor"] = args.hard_neg_factor
        config["model"]["use_all_targets"] = args.use_all_targets
        config["model"]["grad_scale_factor"] = args.grad_scale_factor
        config["model"]["global_local_nce"] = args.global_local_nce
        config["model"]["buffered_nce"] = args.buffered_nce
        config["model"]["cosine_nce"] = args.cosine_nce
        config["model"]["augmentation"] = args.augmentation
        config["model"]["no_rl_augmentation"] = args.no_rl_augmentation
        config["model"]["frame_dropout"] = args.frame_dropout
        config["model"]["keep_last_frame"] = args.keep_last_frame
        config["model"]["time_contrastive"] = args.time_contrastive
        config["model"]["aug_prob"] = args.aug_prob
        config["model"]["target_augmentation"] = args.target_augmentation
        config["model"]["eval_augmentation"] = args.eval_augmentation
        config["model"]["stack_actions"] = args.stack_actions
        config["model"]["classifier"] = args.classifier
        config["algo"]["model_rl_weight"] = args.model_rl_weight
        config["algo"]["reward_loss_weight"] = args.reward_loss_weight
        config["algo"]["model_nce_weight"] = args.model_nce_weight
        config["algo"]["nce_loss_weight"] = args.nce_loss_weight
        config["algo"]["nce_loss_weight_final"] = args.nce_loss_weight_final
        config["algo"]["nce_loss_decay_steps"] = args.nce_loss_decay_steps
        config["algo"]["amortization_loss_weight"] = args.amortization_loss_weight
        config["algo"]["amortization_decay_constant"] = args.amortization_decay_constant
        config["algo"]["time_contrastive"] = args.time_contrastive
        config["algo"]["distributional"] = args.distributional
        config["algo"]["dqn_hidden_size"] = args.dqn_hidden_size
        algo = PizeroModelCategoricalDQN(optim_kwargs=config["optim"], jumps=args.jumps, **config["algo"], detach_model=detach_model)  # Run with defaults.
        agent = DQNSearchAgent(ModelCls=PizeroSearchCatDqnModel, search_args=args, model_kwargs=config["model"], **config["agent"])
    elif control:
        algo = CategoricalDQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariCatDqnAgent(ModelCls=AtariCatDqnModel, model_kwargs=config["model"], **config["agent"])
    else:
        config["model"]["imagesize"] = args.imagesize
        algo = DQN(optim_kwargs=config["optim"], **config["algo"])  # Run with defaults.
        agent = AtariDqnAgent(ModelCls=PizeroCatDqnModel, model_kwargs=config["model"], **config["agent"])
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

    quit()


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
    parser.add_argument('--der', action="store_true")
    parser.add_argument('--control', action="store_true")
    parser.add_argument('--fasteval', type=int, default=1)
    parser.add_argument('--async-sample', action="store_true")
    parser.add_argument('--learn-model', action="store_true")
    parser.add_argument('--stack-actions', type=int, default=0)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--grayscale', type=int, default=1)
    parser.add_argument('--imagesize', type=int, default=100)
    parser.add_argument('--n-steps', type=int, default=int(50e6))
    parser.add_argument('--dqn-hidden-size', type=int, default=256)
    parser.add_argument('--target-update-interval', type=int, default=312)
    parser.add_argument('--target-update-tau', type=float, default=1.)
    parser.add_argument('--batch-b', type=int, default=16)
    parser.add_argument('--batch-t', type=int, default=2)
    parser.add_argument('--eval-imagesize', type=int, default=100)
    parser.add_argument('--beluga', action="store_true")
    parser.add_argument('--jumps', type=int, default=0)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--replay-ratio', type=int, default=2)
    parser.add_argument('--dynamics-blocks', type=int, default=2)
    parser.add_argument('--n-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--transition-model', type=str, default='standard', choices=["standard", "film", "effnet"], help='Type of transition model to use')
    parser.add_argument('--tag', type=str, default='', help='Tag for wandb run.')
    parser.add_argument('--norm-type', type=str, default='in', choices=["bn", "ln", "in", "none"], help='Normalization')
    parser.add_argument('--encoder', type=str, default='curl', choices=["repnet", "curl", "midsize", "nature", "effnet", "bignature", "deepnature"], help='Type of encoder to use')
    parser.add_argument('--padding', type=str, default='same', choices=["same", "valid"], help='Padding choice for Curl Encoder')
    parser.add_argument('--aug-prob', type=float, default=1., help='Probability to apply augmentation')
    parser.add_argument('--frame-dropout', type=float, default=0., help='Probability to dropout frame in framestack.')
    parser.add_argument('--keep-last-frame', type=int, default=1, help='Always keep last frame in frame dropout.')
    parser.add_argument('--film', type=int, default=0)
    parser.add_argument('--nce', type=int, default=0)
    parser.add_argument('--distributional', type=int, default=1)
    parser.add_argument('--prioritized-replay', type=int, default=1)
    parser.add_argument('--cosine-nce', type=int, default=0)
    parser.add_argument('--buffered-nce', type=int, default=0)
    parser.add_argument('--momentum-encoder', type=int, default=0)
    parser.add_argument('--target-encoder-sn', type=int, default=0)
    parser.add_argument('--shared-encoder', type=int, default=0)
    parser.add_argument('--local-nce', type=int, default=0)
    parser.add_argument('--global-nce', type=int, default=0)
    parser.add_argument('--global-local-nce', type=int, default=0)
    parser.add_argument('--use-all-targets', type=int, default=0, help="Also use different timesteps in the same trajectory as negative samples."
                                                                       " Only applies if jumps>0, buffered-nce 0")
    parser.add_argument('--hard-neg-factor', type=int, default=0, help="How many extra hard negatives to use for each example"
                                                                       " Only applies if buffered-nce 0")
    parser.add_argument('--noisy-nets', type=int, default=1)
    parser.add_argument('--grad-scale-factor', type=float, default=0.5, help="Amount by which to scale gradients for trans. model")
    parser.add_argument('--nce-type', type=str, default='custom', choices=["stdim", "moco", "curl", "custom"], help='Style of NCE')
    parser.add_argument('--classifier', type=str, default='bilinear', choices=["mlp", "bilinear", "q_l1"], help='Style of NCE classifier')
    parser.add_argument('--augmentation', type=str, default=['none'], nargs="+",
                        choices=["none", "rrc", "affine", "crop", "blur", "shift", "intensity"],
                        help='Style of augmentation')
    parser.add_argument('--no-rl-augmentation', type=int, default=0, help='Do a separate RL update without aug')
    parser.add_argument('--target-augmentation', type=int, default=0, help='Use augmentation on inputs to target networks')
    parser.add_argument('--eval-augmentation', type=int, default=0, help='Use augmentation on inputs at evaluation time')
    parser.add_argument('--reward-loss-weight', type=float, default=1.)
    parser.add_argument('--model-rl-weight', type=float, default=1.)
    parser.add_argument('--amortization-loss-weight', type=float, default=0.)
    parser.add_argument('--amortization-decay-constant', type=float, default=0.)
    parser.add_argument('--model-nce-weight', type=float, default=1.)
    parser.add_argument('--nce-loss-weight', type=float, default=1.)
    parser.add_argument('--nce-loss-weight-final', type=float, default=-1.)
    parser.add_argument('--nce-loss-decay-steps', type=float, default=50000)
    parser.add_argument('--eps-steps', type=int, default=50000)
    parser.add_argument('--detach-model', type=int, default=1)
    parser.add_argument('--final-eval-only', type=int, default=0)
    parser.add_argument('--time-contrastive', type=int, default=0)
    parser.add_argument('--debug_cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10., help='Max Grad Norm')
    # MCTS arguments
    parser.add_argument('--num-simulations', type=int, default=0)
    parser.add_argument('--eval-simulations', type=int, default=0)
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

    if args.nce_type == "stdim":
        args.local_nce = 1
        args.momentum_encoder = 0
        args.buffered_nce = 0
    elif args.nce_type == "moco":
        args.local_nce = 0
        args.momentum_encoder = 1
        args.buffered_nce = 1
    elif args.nce_type == "curl":
        args.local_nce = 0
        args.momentum_encoder = 1
        args.buffered_nce = 0

    wandb.init(project='resnet-exps', entity='abs-world-models', config=args, tags=[args.tag])
    wandb.config.update(vars(args))
    if args.der:
        debug_build_and_train(game=args.game,
                              cuda_idx=args.debug_cuda_idx,
                              model=args.learn_model,
                              detach_model=args.detach_model,
                              args=args,
                              control=args.control,
                              )
    else:
        build_and_train(
            game=args.game,
            model=args.learn_model,
            detach_model=args.detach_model,
            args=args,
            control=args.control,
        )

