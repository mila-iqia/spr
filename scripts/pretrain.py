from collections import deque
from itertools import chain

import torch
import numpy as np
import os
import wandb

from src.agent import Agent
from src.memory import ReplayMemory
from src.encoders import NatureCNN, ImpalaCNN
from src.envs import Env
from src.eval import test
from src.forward_model import ForwardModel
from src.stdim import InfoNCESpatioTemporalTrainer
from src.dqn_multi_step_stdim_with_actions import MultiStepActionInfoNCESpatioTemporalTrainer
from src.stdim_with_actions import ActionInfoNCESpatioTemporalTrainer
from src.utils import get_argparser, log, set_learning_rate, save_to_pil
from src.episodes import get_random_agent_episodes, get_labeled_episodes,\
    Transition, sample_real_transitions
from src.memory import blank_trans
from src.probe import ProbeTrainer
from atariari.benchmark.ram_annotations import atari_dict
import matplotlib.pyplot as plt


def pretrain(args):
    env = Env(args)
    env.train()
    dqn = Agent(args, env)

    # get initial exploration data
    if args.game.replace("_", "").lower() in atari_dict:
        train_transitions, train_labels,\
        val_transitions, val_labels, \
        test_transitions, test_labels = get_labeled_episodes(args)
    else:
        print("{} does not support probing; omitting.".format(args.game))
        train_transitions = get_random_agent_episodes(args)
        val_transitions = get_random_agent_episodes(args)

    encoder, encoder_trainer = init_encoder(args,
                                            train_transitions,
                                            num_actions=env.action_space(),
                                            agent=dqn)

    encoder_trainer.train(train_transitions,
                          val_transitions,
                          epochs=args.pretrain_epochs)

    if args.integrated_model:
        forward_model = encoder_trainer
    if not args.integrated_model:
        forward_model = train_model(args,
                                    encoder,
                                    train_transitions,
                                    env.action_space(),
                                    init_epochs=args.pretrain_epochs,
                                    val_eps=val_transitions)
        forward_model.args.epochs = args.epochs // 2
        encoder_trainer.epochs = args.epochs // 2

    visualize_temporal_prediction_accuracy(forward_model, val_transitions, args)

    if args.game.replace("_", "").lower() not in atari_dict:
        return

    probe = ProbeTrainer(encoder=encoder_trainer.encoder,
                         forward=encoder_trainer.prediction_module,
                         epochs=args.epochs,
                         method_name=args.method,
                         lr=args.probe_lr,
                         batch_size=args.batch_size,
                         patience=args.patience,
                         wandb=wandb,
                         # fully_supervised=(args.method == "supervised"),
                         save_dir=wandb.run.dir)

    probe.train(train_transitions, val_transitions,
                train_labels, val_labels)
    test_acc, test_f1score = probe.test(test_transitions, test_labels)

    wandb.log(test_acc)
    wandb.log(test_f1score)
    print(test_acc, test_f1score)

    with torch.no_grad():
        train_probe_loss, train_probe_acc, train_probe_f1 = probe.run_multistep(train_transitions, train_labels)
        val_probe_loss, val_probe_acc, val_probe_f1 = probe.run_multistep(val_transitions, val_labels)
        test_probe_loss, test_probe_acc, test_probe_f1 = probe.run_multistep(test_transitions, test_labels)

    plot_multistep_probing(wandb, train_probe_loss, train_probe_acc, train_probe_f1, "train")
    plot_multistep_probing(wandb, val_probe_loss, val_probe_acc, val_probe_f1, "val")
    plot_multistep_probing(wandb, test_probe_loss, test_probe_acc, test_probe_f1, "test")


def visualize_temporal_prediction_accuracy(model, transitions, args):
    with torch.no_grad():
        model.minimum_length = args.visualization_jumps
        model.maximum_length = args.visualization_jumps + 1
        model.dense_supervision = True

        model.encoder.eval(), model.classifier.eval()
        model.do_one_epoch(transitions, plots=True)
        model.encoder.train(), model.classifier.train()

        model.minimum_length = args.min_jump_length
        model.maximum_length = args.max_jump_length
        model.dense_supervision = args.dense_supervision


def init_encoder(args,
                 transitions,
                 num_actions,
                 agent=None):
    if args.integrated_model:
        trainer = MultiStepActionInfoNCESpatioTemporalTrainer
    else:
        trainer = InfoNCESpatioTemporalTrainer

    observation_shape = transitions[0].state.shape
    if args.encoder_type == "Nature":
        encoder = NatureCNN(observation_shape[0], args)
    elif args.encoder_type == "Impala":
        encoder = ImpalaCNN(observation_shape[0], args)
    encoder.to(args.device)
    torch.set_num_threads(1)

    config = {}
    config.update(vars(args))
    config['obs_space'] = observation_shape  # weird hack
    config['num_actions'] = num_actions  # weird hack
    if args.method == "infonce-stdim":
        if args.online_agent_training:
            trainer = trainer(encoder,
                              config,
                              device=args.device,
                              wandb=wandb,
                              agent=agent)
        else:
            trainer = trainer(encoder, config, device=args.device, wandb=wandb)
    else:
        assert False, "method {} has no trainer".format(args.method)
    return encoder, trainer


def train_model(args,
                encoder,
                real_transitions,
                num_actions,
                val_eps=None,
                init_epochs=None):
    forward_model = ForwardModel(args, encoder, num_actions)
    forward_model.train(real_transitions, init_epochs)
    if val_eps is not None:
        forward_model.sd_predictor.eval()
        forward_model.hidden.eval()
        forward_model.reward_predictor.eval()
        forward_model.train(val_eps, init_epochs)
        forward_model.sd_predictor.train()
        forward_model.hidden.train()
        forward_model.reward_predictor.train()
    return forward_model


def plot_multistep_probing(wandb, epoch_loss, accuracy, f1, prefix="train"):
    images = []
    labels = []

    dir = "./figs/{}/".format(wandb.run.name)
    try:
        os.makedirs(dir)
    except FileExistsError:
        # directory already exists
        pass

    # Iterate over keys, extracting data from list of dicts.
    for k, _ in epoch_loss[0].items():
        data = [d[k] for d in epoch_loss]
        plt.figure()
        plt.plot(np.arange(len(data)), data)
        plt.xlabel("Number of jumps")
        plt.ylabel(k)
        plt.tight_layout()
        plt.savefig(dir + "{}_{}_loss.png".format(prefix, k))
        image = save_to_pil()
        labels.append("{}_{}_loss".format(prefix, k))
        images.append(wandb.Image(image,
                                  caption="{} {} loss".format(prefix, k)))
        plt.close()

    # Iterate over keys, extracting data from list of dicts.
    for k, _ in accuracy[0].items():
        data = [d[k] for d in accuracy]
        plt.figure()
        plt.plot(np.arange(len(data)), data)
        plt.xlabel("Number of jumps")
        plt.ylabel(k)
        plt.tight_layout()
        plt.savefig(dir + "{}_{}_accuracy.png".format(prefix, k))
        image = save_to_pil()
        labels.append("{}_{}_acc".format(prefix, k))
        images.append(wandb.Image(image,
                                  caption="{} {} accuracy".format(prefix, k)))
        plt.close()

    # Iterate over keys, extracting data from list of dicts.
    for k, _ in f1[0].items():
        data = [d[k] for d in f1]
        plt.figure()
        plt.plot(np.arange(len(data)), data)
        plt.xlabel("Number of jumps")
        plt.ylabel(k)
        plt.tight_layout()
        plt.savefig(dir + "{}_{}_f1.png".format(prefix, k))
        image = save_to_pil()
        labels.append("{}_{}_f1".format(prefix, k))
        images.append(wandb.Image(image,
                                  caption="{} {} f1".format(prefix, k)))
        plt.close()

    log = {label: image for label, image in zip(labels, images)}
    wandb.log(log)


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    tags = ["Pretraining"]
    if len(args.name) > 0:
        wandb.init(project=args.wandb_proj, tags=tags, name=args.name, entity="abs-world-models")
    else:
        wandb.init(project=args.wandb_proj, tags=tags, entity="abs-world-models")
    wandb.config.update(vars(args))

    results_dir = os.path.join('results', args.id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')
    pretrain(args)
