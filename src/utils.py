# Dependencies
import torch
from torchvision.utils import make_grid
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score as compute_f1_score

from collections import defaultdict

import argparse
import copy
import os
import subprocess

from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.utils import get_vec_normalize



def get_argparser():
    r"""Returns the argument parser with arguments given below.

    inputs:
    -------

    outputs:
    --------
    parser: ArgumentParser instant with arguments given below.
    """

    parser = argparse.ArgumentParser()

    # Basic arguments.
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

    # Training-specific arguments.
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
    parser.add_argument('--encoder-type', type=str, default="Nature", 
        choices=["Impala", "Nature"], help='Encoder type (Impala or Nature)')
    parser.add_argument('--feature-size', type=int, default=256, 
        help='Size of features')
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--end-with-relu", action='store_true', default=False)
    parser.add_argument("--wandb-proj", type=str, 
        default="curl-atari-neurips-scratch")
    parser.add_argument("--num_rew_evals", type=int, default=10)

    # rl-probe specific arguments
    parser.add_argument("--checkpoint-index", type=int, default=-1)

    # bert specific arguments
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--num_lin_projections", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--seq_len', type=int, default=5,
                            help='Sequence length.')
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--beta", type=float, default=1.0)

    # naff-specific arguments
    parser.add_argument("--naff_fc_size", type=int, default=2048,
                        help="fully connected layer width for naff")
    parser.add_argument("--pred_offset", type=int, default=1,
                        help="how many steps in future to predict")

    # CPC-specific arguments
    parser.add_argument('--sequence_length', type=int, default=100,
                    help='Sequence length.')
    parser.add_argument('--steps_start', type=int, default=0,
                    help='Number of immediate future steps to ignore.')
    parser.add_argument('--steps_end', type=int, default=99,
                    help='Number of future steps to predict.')
    parser.add_argument('--steps_step', type=int, default=4,
                    help='Skip every these many frames.')
    parser.add_argument('--gru_size', type=int, default=256,
                    help='Hidden size of the GRU layers.')
    parser.add_argument('--gru_layers', type=int, default=2,
                    help='Number of GRU layers.')
    parser.add_argument("--collect-mode", type=str, 
                    choices=["random_agent", "atari_zoo", "pretrained_ppo"],
                    default="random_agent")

    # probe arguments
    parser.add_argument("--weights-path", type=str, default="None")
    parser.add_argument("--train-encoder", action='store_true', default=True)
    parser.add_argument('--probe-lr', type=float, default=5e-2)
    parser.add_argument("--probe-collect-mode", type=str, 
        choices=["random_agent", "atari_zoo", "pretrained_ppo"],
        default="random_agent")
    parser.add_argument('--zoo-algos', nargs='+', default=["a2c"])
    parser.add_argument('--zoo-tags', nargs='+', default=["10HR"])
    parser.add_argument('--num-runs', type=int, default=1)
    
    return parser


def set_seeds(seed):
    r"""Sets numpy, torch seeds for reproducibility. Sets backend options.
    
    inputs:
    -------
    seed: Integer value to be set as seed.

    outputs:
    --------
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def calculate_accuracy(preds, y):
    r"""Computes accuracy for torch arrays for predictions preds and ground
    truth labels y for the binary classification setting. Predictions
    and ground truths must be [0, 1]-ranged and are quantized to nearest 
    integer label to get the desired label.

    inputs:
    -------
    preds: Torch array of predictions in range [0, 1]. SHAPE: any.

    y: Torch array of ground truth values in range [0, 1]. SHAPE: preds.shape

    outputs:
    --------
    acc: Torch scalar giving accuracy as a fraction in [0, 1].
    """

    preds = preds >= 0.5
    labels = y >= 0.5
    acc = preds.eq(labels).sum().float() / labels.numel()

    return acc


def calculate_multiclass_f1_score(preds, labels):
    r"""Computes the F1-score for torch arrays for predictions preds and
    ground truth labels. F1-score is harmonic mean of precision, recall.
    The method is intended for multi-class classification. It just requires
    that preds are available as logits for classes so that argmax-ing the
    logits along dim=1 will produce the predicted label.

    inputs:
    -------
    preds: Torch array of prediction. SHAPE: [<batch_size>, <feat_dim>]. 
        The actual predictions are obtained by taking argmax along dim=1.

    labels: Ground truth labels representing the true classes as integers.
        SHAPE: [<batch_size>, ].

    outputs:
    --------
    f1score: The F1-score for the predictions and labels as np.floatX scalar.
    """

    preds = torch.argmax(preds, dim=1).detach().numpy()
    labels = labels.numpy()
    f1score = compute_f1_score(labels, preds, average="weighted")

    return f1score


def calculate_multiclass_accuracy(preds, labels):
    r"""Computes accuracy of torch arrays for predictions preds and ground
    truth labels. The method is intended for multi-class classification. It
    just requires that preds are available as logits for classes and the
    actual predictions can be obtained by argmax-ing along dim=1.

    inputs:
    -------
    preds: Torch array of prediction. SHAPE: [<batch_size>, <feat_dim>]. 
        The actual predictions are obtained by taking argmax along dim=1.

    labels: Ground truth labels representing the true classes as integers.
        SHAPE: [<batch_size>, ].

    outputs:
    --------
    acc: Float giving the accuracy as a fraction in [0, 1].
    """

    preds = torch.argmax(preds, dim=1)
    acc = float(torch.sum(torch.eq(labels, preds)).data) / labels.size(0)

    return acc


def save_model(model, envs, save_dir, model_name, use_cuda):
    r"""Saves the model parameters along with the environment envs's attribute
    value for ob_rms. If that attribute is missing, None is saved for it. The 
    current saving mechanism consists of creating a deepcopy of the model and 
    porting it to cpu. Then, the ob_rms attribute is added to the model.
    An alternative implementation can be via state_dict.

    inputs:
    -------
    model: The torch model which needs to be saved.

    envs: The environments. Their ob_rms attribute will also be stored.

    save_dir: The directory in which to save the model.

    model_name: The name of the model save file.

    use_cuda: Whether cuda is being used. If so, the model's copy to be saved
        needs to be ported to cpu. Otherwise, it is saved as is.

    outputs:
    --------
    """
    
    save_path = os.path.join(save_dir)
    try:
        os.makedirs(save_path)
    except OSError:
        pass

    # A really ugly way to save a model to CPU
    save_model = model
    # If model is on gpu, the copy to be saved needs to be ported to cpu.
    if use_cuda:
        save_model = copy.deepcopy(model).cpu()

    # Add attribute ob_rms from envs.
    save_model = [save_model,
                  getattr(get_vec_normalize(envs), 'ob_rms', None)]

    torch.save(save_model, os.path.join(save_path, model_name + ".pt"))


def evaluate_policy(actor_critic, envs, args, eval_log_dir, device):
    r"""Evaluates a policy contained in actor_critic on environments envs.

    inputs:
    -------
    actor_critic: The object holding the policy in terms of method .act(...).

    envs: The environments on which to evaluate the policy.

    args: Arguments for setting configs of eval_envs.

    eval_log_dir: The directory for log of evaluations.

    device: The device on which to create eval_envs and carry out evaluation.

    outputs:
    --------
    eval_episode_rewards: The list of evaluation episode rewards as floats.
    """

    # Create evaluation environments.
    eval_envs = make_vec_envs(
        args.env_name, args.seed + args.num_processes, args.num_processes,
        args.gamma, eval_log_dir, args.add_timestep, device, True)

    # Get the eval_envs vectorized.
    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

    # Initiate evaluation episode rewards list.
    eval_episode_rewards = []

    # Reset the eval_envs.
    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                               actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(args.num_processes, 1, device=device)

    # Take 10 actions and get 10 rewards.
    while len(eval_episode_rewards) < 10:

        # Act via the policy. Get the action.
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

        # Observe reward and next obsevations based on the action.
        obs, reward, done, infos = eval_envs.step(action)

        # If done, then mask out the rewards. Otherwise, keep as is.
        eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                        for done_ in done])
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    # Display mean reward stats for the number of episodes.
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".
          format(len(eval_episode_rewards),
                 np.mean(eval_episode_rewards)))

    eval_envs.close()

    return eval_episode_rewards


def visualize_activation_maps(encoder, input_obs_batch, wandb):
    r"""Visualizes the activation maps of an encoder architecture. Intended to
    work for the two defined encoder architectures of 'ImpalaCNN' and 
    'NatureCNN' only. Expects that the inputs are batches of images with pixel
    values in the range of [0-255] and torch's standard shape format. The plots
    plt generated are stored to Weights and Biases for nice visualization.

    inputs:
    -------
    encoder: The standard encoder network, expected to be either 'ImpalaCNN'
        or 'NatureCNN', as a torch model. 
    input_obs_batch: The input images for which to visualize activations.
        SHAPE: [<batch_size>, <channels=3>, <width>, <height>]. The pixel
        values are inteded to be [0-255] ranged.
    wandb: Weights and Biases handle for logging visualization data.

    outputs:
    --------
    """

    # Scale images to 0-1.
    scaled_images = input_obs_batch / 255.

    # Define the feature map for the encoder.
    if encoder.__class__.__name__ == 'ImpalaCNN':
        fmap = F.relu(encoder.layer3(encoder.layer2(encoder.layer1(scaled_images)))).detach()
    elif encoder.__class__.__name__ == 'NatureCNN':
        fmap = F.relu(encoder.main[4](F.relu(encoder.main[2](F.relu(encoder.main[0](input_obs_batch)))))).detach()

    # Get feature channel count.
    out_channels = fmap.shape[1]

    # Upsample and add a dummy channel dimension.
    fmap_upsampled = F.interpolate(fmap, size=input_obs_batch.shape[-2:],\
                        mode='bilinear').unsqueeze(dim=2)

    # Process each image of the batch.
    for i in range(input_obs_batch.shape[0]):
        fmap_grid = make_grid(fmap_upsampled[i], normalize=True)
        img_grid = make_grid([scaled_images[i]] * out_channels)
        plt.imshow(img_grid.cpu().numpy().transpose([1, 2, 0]))
        plt.imshow(fmap_grid.cpu().numpy().transpose([1, 2, 0]), cmap='jet', alpha=0.5)
        # plt.savefig('act_maps/' + 'file%02d.png' % i)
        wandb.log({'actmap': wandb.Image(plt, caption='Activation Map')})
    
    # generate_video()


def generate_video():
    r"""Creates a video from given .png files with configs given below.

    inputs:
    -------

    outputs:
    --------
    """

    os.chdir("act_maps")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', 
        '-pix_fmt', 'yuv420p', 'video_name.mp4'
    ])


class appendabledict(defaultdict):
    r"""Extends the defaultdict class to enable subslicing and append."""

    def __init__(self, type_=list, *args, **kwargs):
        r"""The constructor.

        inputs:
        -------
        type_=list: The type to be passed to initialize the super.

        outputs:
        --------
        """ 
        self.type_ = type_
        super().__init__(type_, *args, **kwargs)
        # def map_(self, func):
        #   for k, v in self.items():
        #       self.__setitem__(k, func(v))

    def subslice(self, slice_):
        r"""Indexes every value in the dict according to a specified slice.

        Parameters
        ----------
        slice : int or slice type
            An indexing slice , e.g., ``slice(2, 20, 2)`` or ``2``.


        Returns
        -------
        sliced_dict : dict (not appendabledict type!)
            A dictionary with each value from this object's dictionary, but the value is sliced according to slice_
            e.g. if this dictionary has {a:[1,2,3,4], b:[5,6,7,8]}, then self.subslice(2) returns {a:3,b:7}
                 self.subslice(slice(1,3)) returns {a:[2,3], b:[6,7]}

        """
        sliced_dict = {}
        for k, v in self.items():
            sliced_dict[k] = v[slice_]
        return sliced_dict

    def append_update(self, other_dict):
        r"""Appends current dict's values with values from other_dict.

        Parameters
        ----------
        other_dict : dict
            A dictionary that you want to append to this dictionary


        Returns
        -------
        Nothing. The side effect is this dict's values change

        """
        for k, v in other_dict.items():
            self.__getitem__(k).append(v)


# Thanks Bjarten! (https://github.com/Bjarten/early-stopping-pytorch)
class EarlyStopping(object):
    r"""Early stops the training if validation loss doesn't improve 
    after a given patience. Enables storage of better generalizing models
    while saving unnecessary compute after such model is achieved and before
    the patience exhausts!"""

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


def bucket_coord(coord, num_buckets, min_coord=0, max_coord=255, stride=1):
    r"""Converts a coordinate value into a bucketed coordinate value, with
    minimum value taken by min_coord, maximum value by max_coord and the
    number of buckets being num_buckets. The min_coord and max_coord are 
    intended to be integers. The stride is how much a variable is incremented
    by (usually 1). 

    inputs:
    -------
    coord: The coordinate to be bucketed.

    num_buckets: The number of buckets that are desired.

    min_coord=0: The minimum coordinate of the entire coord range.

    max_coord=255: The maximum coordinate of the entire coord range.

    stride=1: The increment in the variable.

    outputs:
    --------
    bucketed_coord: The bucketed coordinate value, which is the bucket 
        number to which the coordinate would belong as an integer. 
    """

    try:
        assert (coord <= max_coord and coord >= min_coord)
    except:
        print("coord: %i, max: %i, min: %i, num_buckets: %i" % (coord, max_coord, min_coord, num_buckets))
        assert False, coord
    
    coord_range = (max_coord - min_coord) + 1
    
    # thresh is how many units in raw_coord space correspond to one bucket
    if coord_range < num_buckets:  
        # we never want to upsample from the original coord
        thresh = stride
    else:
        thresh = coord_range / num_buckets

    bucketed_coord = np.floor((coord - min_coord) / thresh)

    return bucketed_coord


def bucket_discrete(coord, possible_values):
    r"""Converts a coordinate into one of the discrete possible_values.
    The possible_values are mapped to their index as the discrete quantizing
    value. The coord is then hashed as per this mapping.

    inputs:
    -------
    coord: The coordinate to be converted to discrete value.

    possible_values: The discrete values list a coordinate can take.

    outputs:
    --------
    discretized_value (implicit): The discrete value to which coord is mapped.
    """
    
    inds = range(len(possible_values))
    hash_table = dict(zip(possible_values, inds))

    return hash_table[coord]


class Cutout(object):
    r"""Implements randomly masking out one or more patches from an image."""

    def __init__(self, n_holes, length):
        r"""The constructor.

        inputs:
        -------
        n_holes: Number of patches to cut out of each image as an integer.

        length: The length (in pixels) of each square patch.
        """

        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        r"""Implements the call on the instance of this class. The input image
        is masked with self.n_holes many square patches of size self.length.
        
        inputs:
        -------
        img: Torch tensor image of size (C, H, W).

        outputs:
        --------
        img: Image with n_holes of dimension length x length cut out of it.
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