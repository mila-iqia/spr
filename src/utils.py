from rlpyt.experiments.configs.atari.dqn.atari_dqn import configs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class dummy_context_mgr:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def set_config(args, game):
    # TODO: Use Hydra to manage configs
    config = configs['ernbw']
    config['env']['game'] = game
    config["env"]["grayscale"] = args.grayscale
    config["env"]["num_img_obs"] = args.framestack
    config["eval_env"]["game"] = config["env"]["game"]
    config["eval_env"]["grayscale"] = args.grayscale
    config["eval_env"]["num_img_obs"] = args.framestack
    config['env']['imagesize'] = args.imagesize
    config['eval_env']['imagesize'] = args.imagesize
    config['env']['seed'] = args.seed
    config['eval_env']['seed'] = args.seed
    config["model"]["dueling"] = bool(args.dueling)
    config["algo"]["min_steps_learn"] = args.min_steps_learn
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
    config["sampler"]["eval_max_steps"] = 100*28000  # 28k is just a safe ceiling
    config['sampler']['batch_B'] = args.batch_b
    config['sampler']['batch_T'] = args.batch_t

    config['agent']['eps_init'] = args.eps_init
    config['agent']['eps_final'] = args.eps_final
    config["model"]["noisy_nets_std"] = args.noisy_nets_std

    if args.noisy_nets:
        config['agent']['eps_eval'] = 0.001

    # New SPR Arguments
    config["model"]["imagesize"] = args.imagesize
    config["model"]["jumps"] = args.jumps
    config["model"]["dynamics_blocks"] = args.dynamics_blocks
    config["model"]["spr"] = args.spr
    config["model"]["noisy_nets"] = args.noisy_nets
    config["model"]["momentum_encoder"] = args.momentum_encoder
    config["model"]["shared_encoder"] = args.shared_encoder
    config["model"]["local_spr"] = args.local_spr
    config["model"]["global_spr"] = args.global_spr
    config["model"]["distributional"] = args.distributional
    config["model"]["renormalize"] = args.renormalize
    config["model"]["norm_type"] = args.norm_type
    config["model"]["augmentation"] = args.augmentation
    config["model"]["q_l1_type"] = args.q_l1_type
    config["model"]["dropout"] = args.dropout
    config["model"]["time_offset"] = args.time_offset
    config["model"]["aug_prob"] = args.aug_prob
    config["model"]["target_augmentation"] = args.target_augmentation
    config["model"]["eval_augmentation"] = args.eval_augmentation
    config["model"]["classifier"] = args.classifier
    config["model"]["final_classifier"] = args.final_classifier
    config['model']['momentum_tau'] = args.momentum_tau
    config["model"]["dqn_hidden_size"] = args.dqn_hidden_size
    config["model"]["model_rl"] = args.model_rl_weight
    config["model"]["residual_tm"] = args.residual_tm
    config["algo"]["model_rl_weight"] = args.model_rl_weight
    config["algo"]["reward_loss_weight"] = args.reward_loss_weight
    config["algo"]["model_spr_weight"] = args.model_spr_weight
    config["algo"]["t0_spr_loss_weight"] = args.t0_spr_loss_weight
    config["algo"]["time_offset"] = args.time_offset
    config["algo"]["distributional"] = args.distributional
    config["algo"]["delta_clip"] = args.delta_clip
    config["algo"]["prioritized_replay"] = args.prioritized_replay

    return config