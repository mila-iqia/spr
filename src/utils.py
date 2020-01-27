import argparse
import copy
from datetime import datetime
import os
import subprocess

import atari_py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score as compute_f1_score
from collections import defaultdict
import wandb
import matplotlib.pyplot as plt
import io
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-env-steps', type=int, default=1000000,
                        help='Total number to env steps to train (default: 100000)')
    parser.add_argument('--num-envs', type=int, default=8, help='Number of parallel envs to run')
    parser.add_argument('--sync-envs', action='store_true')
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--target-update-interval', type=int, default=1000,
                        help="Number of gradient steps for each update to the "
                             "target network.  <=0 to disable target network.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to use')
    parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument('--framestack', type=int, default=4, metavar='T',
                        help='Number of consecutive frames stacked to form an observation')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                        help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--evaluation-episodes', type=int, default=5,
                        help='Number of episodes to average over when evaluating')

    # MCTS arguments
    parser.add_argument('--num-simulations', type=int, default=10)

    # PiZero arguments
    parser.add_argument('--training-interval', type=int, default=200,
                        help='Perform training after every {training-interval} env steps ')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to use during training')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='η', help='Learning rate')
    parser.add_argument('--optim', type=str, default='sgd', choices=["adam", "sgd"], help='Optimizer')
    parser.add_argument('--lr-decay-steps', type=float, default=350.e3, help='Learning rate decay time constant')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning rate decay scale')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--adam-eps', type=float, default=1e-4, help='Adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='Weight decay regularization constant')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden size of various MLPs')
    parser.add_argument('--dynamics-blocks', type=int, default=16, help='# of resblocks in dynamics model')
    parser.add_argument('--multistep', type=int, default=1, help='n-step for bootstrapping value targets')
    parser.add_argument('--priority-exponent', type=float, default=1., metavar='ω',
                        help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=1., metavar='β',
                        help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--jumps', type=int, default=5, help='')
    parser.add_argument('--value-loss-weight', type=float, default=0.1)
    parser.add_argument('--policy-loss-weight', type=float, default=1.)
    parser.add_argument('--reward-loss-weight', type=float, default=1.)
    parser.add_argument('--contrastive-loss-weight', type=float, default=1.)
    parser.add_argument('--film', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--no-nce', action='store_true')
    parser.add_argument('--reanalyze', action='store_true')
    parser.add_argument('--use-all-targets', action='store_true')
    parser.add_argument('--evaluation-interval', type=int, default=100000,
                        help='Evaluate after every {evaluation-interval} env steps')
    parser.add_argument('--log-interval', type=int, default=4000,
                        help='Evaluate after every {evaluation-interval} env steps')

    parser.add_argument('--wandb-proj', type=str, default='pizero')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--savedir', type=str, default='')

    args = parser.parse_args()
    args.max_episode_length = int(108e3)

    return args


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calculate_accuracy(preds, y):
    preds = preds >= 0.5
    labels = y >= 0.5
    acc = preds.eq(labels).sum().float() / labels.numel()
    return acc


def calculate_multiclass_f1_score(preds, labels):
    preds = torch.argmax(preds, dim=1).detach().numpy()
    labels = labels.numpy()
    f1score = compute_f1_score(labels, preds, average="weighted")
    return f1score


def calculate_multiclass_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    acc = float(torch.sum(torch.eq(labels, preds)).data) / labels.size(0)
    return acc


# Thanks Bjarten! (https://github.com/Bjarten/early-stopping-pytorch)
class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, wandb=None, name=""):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0.
        self.name = name
        self.wandb = wandb

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping for {self.name} counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'{self.name} has stopped')

        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def load_checkpoint(self, model):
        save_dir = self.wandb.run.dir
        model.load_state_dict(torch.load(save_dir + "/" + self.name + ".pt"))

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation accuracy increased for {self.name}  ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')

        save_dir = self.wandb.run.dir
        torch.save(model.state_dict(), save_dir + "/" + self.name + ".pt")
        self.val_acc_max = val_acc


# Simple ISO 8601 timestamped logger
def log(steps, avg_reward):
    s = 'T = ' + str(steps) + ' | Avg. reward: ' + str(avg_reward)
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)
    wandb.log({'avg_reward': avg_reward, 'total_steps': steps})


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fig2data(fig):
    """
    Borrowed from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def save_to_pil():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    im.load()
    return im
