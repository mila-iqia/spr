from collections import deque
from itertools import chain

import torch
import numpy as np
import scipy
import os
import wandb

from src.memory import ReplayMemory, blank_batch_trans
from src.agent import Agent
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
    dqn = Agent(args, env.action_space())

    # get initial exploration data
    if args.game.replace("_", "").lower() in atari_dict:
        train_transitions, train_labels,\
        val_transitions, val_labels, \
        test_transitions, test_labels = get_labeled_episodes(args)
        print("Using {} train transitions, {} validation transitions and {} test transitions.".format(len(train_transitions),
                                                                                                      len(val_transitions),
                                                                                                      len(test_transitions)))
    else:
        print("{} does not support probing; omitting.".format(args.game))
        train_transitions = get_random_agent_episodes(args)
        val_transitions = get_random_agent_episodes(args)

    train_memory = ReplayMemory(args, args.real_buffer_capacity, images=True,
                                priority_weight=args.model_priority_weight,
                                priority_exponent=args.model_priority_exponent)
    val_memory = ReplayMemory(args, args.real_buffer_capacity, images=True,
                              priority_weight=args.model_priority_weight,
                              priority_exponent=args.model_priority_exponent)
    for t in train_transitions:
        state, action, reward, terminal = t[1:]
        train_memory.append(state, action, reward, not terminal)
    for t in val_transitions:
        state, action, reward, terminal = t[1:]
        val_memory.append(state, action, reward, not terminal)

    encoder, encoder_trainer = init_encoder(args,
                                            train_transitions,
                                            num_actions=env.action_space(),
                                            agent=dqn)

    if len(args.load) > 0:
        print("Loading weights from {}".format(args.load))
        encoder_trainer.load_state_dict(torch.load(args.load))

    encoder_trainer.train(train_memory,
                          val_memory,
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

    assess_dones(encoder_trainer, val_transitions, "Val")
    assess_returns(encoder_trainer, val_transitions, "Val")
    assess_dones(encoder_trainer, train_transitions, "Train")
    assess_returns(encoder_trainer, train_transitions, "Train")
    visualize_temporal_prediction_accuracy(forward_model, val_memory, args)

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


def assess_dones(model, transitions, mode="Val"):
    with torch.no_grad():
        dir = "./figs/{}/".format(wandb.run.name)
        try:
            os.makedirs(dir)
        except FileExistsError:
            # directory already exists
            pass

        episodes = []
        current_ep = []
        for transition in transitions:
            current_ep.append(transition)
            if not transition.nonterminal:
                episodes.append(current_ep)
                current_ep = []

        pred_nonterminals = []
        times_to_termination = []
        for episode in episodes:
            state_deque = deque(maxlen=4)
            for i in range(4):
                state_deque.append(blank_batch_trans.state)
            for transition in episode:
                state_deque.append(transition.state)
                state = torch.stack(list(state_deque))
                state = state.float()/255.
                state = state.to(args.device)
                z = model.encoder(state).view(1, -1)
                action = torch.tensor(transition.action, device=args.device).long().unsqueeze(0)
                _, _, nonterminal = model.predict(z, action, mean_rew=True)

                pred_nonterminals.append(nonterminal.cpu().item())
                times_to_termination.append(len(episode) - transition.timestep)

        pred_dones = 1 - np.array(pred_nonterminals)
        times_to_termination = np.array(times_to_termination)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pred_dones, times_to_termination)

        print("{} r = {}, p = {}".format(mode, r_value, p_value))
        plt.figure()
        plt.scatter(pred_dones, times_to_termination, alpha=0.5)
        predictions = slope*pred_dones + intercept
        plt.plot(pred_dones, predictions, c="red")
        plt.xlabel("Predicted termination probability")
        plt.ylabel("Timesteps remaining until termination")
        plt.title(r"{} Termination r = {}, p = {}".format(mode, r_value, p_value))

        plt.savefig(dir + "{}_dones.png".format(mode))
        image = save_to_pil()
        dict = {"{}_dones".format(mode):
                    wandb.Image(image, caption="{} dones".format(mode)),
                "{}_dones_r".format(mode): r_value}
        wandb.log(dict)

        plt.close()


def assess_returns(model, transitions, mode="Val"):
    with torch.no_grad():
        dir = "./figs/{}/".format(wandb.run.name)
        try:
            os.makedirs(dir)
        except FileExistsError:
            # directory already exists
            pass

        episodes = []
        current_ep = []
        for transition in transitions:
            current_ep.append(transition)
            if not transition.nonterminal:
                episodes.append(current_ep)
                current_ep = []

        pred_rewards = []
        for episode in episodes:
            ep_rew = 0
            state_deque = deque(maxlen=4)
            for i in range(4):
                state_deque.append(blank_batch_trans.state)
            for transition in episode:
                state_deque.append(transition.state)
                state = torch.stack(list(state_deque))
                state = state.float()/255.
                state = state.to(args.device)
                z = model.encoder(state).view(1, -1)
                action = torch.tensor(transition.action, device=args.device).long().unsqueeze(0)
                _, reward, _ = model.predict(z, action, mean_rew=True)
                ep_rew += reward.item()

            pred_rewards.append(ep_rew/len(episode))

        true_rewards = [np.mean([t.reward for t in episode]) for episode in episodes]
        pred_rewards = np.array(pred_rewards)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pred_rewards, y=true_rewards)

        print("{} r = {}, p = {}".format(mode, r_value, p_value))
        plt.figure()
        plt.scatter(pred_rewards, true_rewards)
        predictions = slope*pred_rewards + intercept
        plt.plot(pred_rewards, predictions, c="red")
        plt.xlabel("Predicted mean reward")
        plt.ylabel("True mean reward")
        plt.title(r"{} Reward r = {}, p = {}".format(mode, r_value, p_value))

        plt.savefig(dir + "{}_rewards.png".format(mode))
        image = save_to_pil()
        dict = {"{}_rewards".format(mode):
                    wandb.Image(image, caption="{} returns".format(mode)),
                "{}_rewards_r".format(mode): r_value}
        wandb.log(dict)

        plt.close()



def visualize_temporal_prediction_accuracy(model, transitions, args):
    with torch.no_grad():
        model.minimum_length = args.visualization_jumps
        model.maximum_length = args.visualization_jumps + 1
        model.dense_supervision = True

        model.reset_trackers("val")
        model.encoder.eval(), model.classifier.eval()
        model.do_one_epoch(transitions, log=True, plots=True)
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
        plt.savefig(dir + "{}_{}.png".format(prefix, k))
        image = save_to_pil()
        labels.append("{}_{}".format(prefix, k))
        images.append(wandb.Image(image,
                                  caption="{} {}".format(prefix, k)))
        plt.close()

    # Iterate over keys, extracting data from list of dicts.
    for k, _ in f1[0].items():
        data = [d[k] for d in f1]
        plt.figure()
        plt.plot(np.arange(len(data)), data)
        plt.xlabel("Number of jumps")
        plt.ylabel(k)
        plt.tight_layout()
        plt.savefig(dir + "{}_{}.png".format(prefix, k))
        image = save_to_pil()
        labels.append("{}_{}".format(prefix, k))
        images.append(wandb.Image(image,
                                  caption="{} {}".format(prefix, k)))
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
