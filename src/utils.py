import argparse
import copy
import os
import subprocess

import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score as compute_f1_score
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize
from collections import defaultdict

train_encoder_methods = ["infonce-stdim"]


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='MontezumaRevengeNoFrameskip-v4',
                        help='environment to train on (default: MontezumaRevengeNoFrameskip-v4)')
    parser.add_argument('--num-frame-stack', type=int, default=1,
                        help='Number of frames to stack for a state')
    parser.add_argument('--no-downsample', action='store_true', default=True,
                        help='Whether to use a linear classifier')
    parser.add_argument('--pretraining-steps', type=int, default=100000,
                        help='Number of steps to pretrain representations (default: 100000)')
    parser.add_argument('--num-processes', type=int, default=8,
                        help='Number of parallel environments to collect samples from (default: 8)')
    parser.add_argument('--method', type=str, default='infonce-stdim',
                        choices=train_encoder_methods,
                        help='Method to use for training representations (default: infonce-stdim)')

    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning Rate foe learning representations (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Mini-Batch Size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for  (default: 100)')
    parser.add_argument('--cuda-id', type=int, default=0,
                        help='CUDA device index')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed to use')
    parser.add_argument('--encoder-type', type=str, default="Nature", choices=["Impala", "Nature"],
                        help='Encoder type (Impala or Nature)')
    parser.add_argument('--feature-size', type=int, default=256,
                        help='Size of features')
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--color", action='store_true', default=False)
    parser.add_argument("--end-with-relu", action='store_true', default=False)
    parser.add_argument("--wandb-proj", type=str, default="awm")
    parser.add_argument("--num_rew_evals", type=int, default=10)

    parser.add_argument("--collect-mode", type=str, choices=["random_agent", "atari_zoo", "pretrained_ppo"],
                        default="random_agent")

    parser.add_argument('--num-runs', type=int, default=1)
    return parser


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


def save_model(model, envs, save_dir, model_name, use_cuda):
    save_path = os.path.join(save_dir)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    # A really ugly way to save a model to CPU
    save_model = model
    if use_cuda:
        save_model = copy.deepcopy(model).cpu()

    save_model = [save_model,
                  getattr(get_vec_normalize(envs), 'ob_rms', None)]

    torch.save(save_model, os.path.join(save_path, model_name + ".pt"))


def evaluate_policy(actor_critic, envs, args, eval_log_dir, device):
    eval_envs = make_vec_envs(
        args.env_name, args.seed + args.num_processes, args.num_processes,
        args.gamma, eval_log_dir, args.add_timestep, device, True)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                               actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(args.num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(action)

        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                        for done_ in done])
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".
          format(len(eval_episode_rewards),
                 np.mean(eval_episode_rewards)))
    eval_envs.close()
    return eval_episode_rewards


def visualize_activation_maps(encoder, input_obs_batch, wandb):
    scaled_images = input_obs_batch / 255.
    if encoder.__class__.__name__ == 'ImpalaCNN':
        fmap = F.relu(encoder.layer3(encoder.layer2(encoder.layer1(scaled_images)))).detach()
    elif encoder.__class__.__name__ == 'NatureCNN':
        fmap = F.relu(encoder.main[4](F.relu(encoder.main[2](F.relu(encoder.main[0](input_obs_batch)))))).detach()
    out_channels = fmap.shape[1]
    # upsample and add a dummy channel dimension
    fmap_upsampled = F.interpolate(fmap, size=input_obs_batch.shape[-2:], mode='bilinear').unsqueeze(dim=2)
    for i in range(input_obs_batch.shape[0]):
        fmap_grid = make_grid(fmap_upsampled[i], normalize=True)
        img_grid = make_grid([scaled_images[i]] * out_channels)
        plt.imshow(img_grid.cpu().numpy().transpose([1, 2, 0]))
        plt.imshow(fmap_grid.cpu().numpy().transpose([1, 2, 0]), cmap='jet', alpha=0.5)
        # plt.savefig('act_maps/' + 'file%02d.png' % i)
        wandb.log({'actmap': wandb.Image(plt, caption='Activation Map')})
    # generate_video()


def generate_video():
    os.chdir("act_maps")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])


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

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation accuracy increased for {self.name}  ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')

        save_dir = self.wandb.run.dir
        torch.save(model.state_dict(), save_dir + "/" + self.name + ".pt")
        self.val_acc_max = val_acc


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
