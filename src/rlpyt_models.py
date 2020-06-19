import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import spectral_norm

from rlpyt.models.dqn.atari_catdqn_model import DistributionalHeadModel
from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel
from rlpyt.models.utils import scale_grad, update_state_dict
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.sync_rl import SyncRlMixin, SyncWorkerEval
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.model_trainer import ValueNetwork, TransitionModel, \
    NetworkOutput, from_categorical, ScaleGradient, init, \
    ResidualBlock, renormalize, FiLMTransitionModel, init_normalization
from src.buffered_nce import BufferedNCE, LocBufferedNCE, BlockNCE
from src.rlpyt_effnet import RLEffNet, EffnetTransitionModel
from src.utils import count_parameters, dummy_context_mgr
import numpy as np
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d
from kornia.geometry.transform import Resize
from rlpyt.utils.logging import logger
import copy
import wandb
import time

atari_human_scores = dict(
    alien=7127.7, amidar=1719.5, assault=742.0, asterix=8503.3,
    bank_heist=753.1, battle_zone=37187.5, boxing=12.1,
    breakout=30.5, chopper_command=7387.8, crazy_climber=35829.4,
    demon_attack=1971.0, freeway=29.6, frostbite=4334.7,
    gopher=2412.5, hero=30826.4, jamesbond=302.8, kangaroo=3035.0,
    krull=2665.5, kung_fu_master=22736.3, ms_pacman=6951.6, pong=14.6,
    private_eye=69571.3, qbert=13455.0, road_runner=7845.0,
    seaquest=42054.7, up_n_down=11693.2
)

atari_der_scores = dict(
    alien=739.9, amidar=188.6, assault=431.2, asterix=470.8,
    bank_heist=51.0, battle_zone=10124.6, boxing=0.2,
    breakout=1.9, chopper_command=861.8, crazy_climber=16185.3,
    demon_attack=508, freeway=27.9, frostbite=866.8,
    gopher=349.5, hero=6857.0, jamesbond=301.6,
    kangaroo=779.3, krull=2851.5, kung_fu_master=14346.1,
    ms_pacman=1204.1, pong=-19.3, private_eye=97.8, qbert=1152.9,
    road_runner=9600.0, seaquest=354.1, up_n_down=2877.4,
)

atari_nature_scores = dict(
    alien=3069, amidar=739.5, assault=3359,
    asterix=6012, bank_heist=429.7, battle_zone=26300.,
    boxing=71.8, breakout=401.2, chopper_command=6687.,
    crazy_climber=114103, demon_attack=9711., freeway=30.3,
    frostbite=328.3, gopher=8520., hero=19950., jamesbond=576.7,
    kangaroo=6740., krull=3805., kung_fu_master=23270.,
    ms_pacman=2311., pong=18.9, private_eye=1788.,
    qbert=10596., road_runner=18257., seaquest=5286., up_n_down=8456.
)

atari_random_scores = dict(
    alien=227.8, amidar=5.8, assault=222.4,
    asterix=210.0, bank_heist=14.2, battle_zone=2360.0,
    boxing=0.1, breakout=1.7, chopper_command=811.0,
    crazy_climber=10780.5, demon_attack=152.1, freeway=0.0,
    frostbite=65.2, gopher=257.6, hero=1027.0, jamesbond=29.0,
    kangaroo=52.0, krull=1598.0, kung_fu_master=258.5,
    ms_pacman=307.3, pong=-20.7, private_eye=24.9,
    qbert=163.9, road_runner=11.5, seaquest=68.4, up_n_down=533.4
)

def channel_dropout(images, drop_prob, keep_last=True):
    mask = torch.rand(images.shape[:2], device=images.device)
    mask = (mask > drop_prob).float()
    if keep_last:
        mask[:, -1] = 1.
    else:
        present_frames = mask.sum(-1)
        correction = torch.clamp(1 - present_frames, 0., 1.)
        mask[:, -1] = mask[:, -1] + correction
    images = images * mask[:, :, None, None]
    return images


def maybe_update_summary(key, value):
    if key not in wandb.run.summary:
        wandb.run.summary[key] = value
    else:
        wandb.run.summary[key] = max(value, wandb.run.summary[key])

class AsyncRlEvalWandb(AsyncRlEval):
    def log_diagnostics(self, itr, sampler_itr, throttle_time):
        cum_steps = sampler_itr * self.sampler.batch_size
        self.wandb_info = {'cum_steps': cum_steps}
        super().log_diagnostics(itr, sampler_itr, throttle_time)
        wandb.log(self.wandb_info)

    def _log_infos(self, traj_infos=None):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    values = [info[k] for info in traj_infos]
                    logger.record_tabular_misc_stat(k,
                                                    values)
                    self.wandb_info[k + "Average"] = np.average(values)
                    self.wandb_info[k + "Std"] = np.std(values)
                    self.wandb_info[k + "Min"] = np.min(values)
                    self.wandb_info[k + "Max"] = np.max(values)
                    self.wandb_info[k + "Median"] = np.median(values)
                    if k == 'GameScore':
                        game = self.sampler.env_kwargs['game']
                        random_score = atari_random_scores[game]
                        der_score = atari_der_scores[game]
                        nature_score = atari_nature_scores[game]
                        human_score = atari_human_scores[game]
                        normalized_score = (np.average(values) - random_score) / (human_score - random_score)
                        der_normalized_score = (np.average(values) - random_score) / (der_score - random_score)
                        nature_normalized_score = (np.average(values) - random_score) / (nature_score - random_score)
                        self.wandb_info[k + "Normalized"] = normalized_score
                        self.wandb_info[k + "DERNormalized"] = der_normalized_score
                        self.wandb_info[k + "NatureNormalized"] = nature_normalized_score

                        maybe_update_summary(k+"Best", np.average(values))
                        maybe_update_summary(k+"NormalizedBest", normalized_score)
                        maybe_update_summary(k+"DERNormalizedBest", der_normalized_score)
                        maybe_update_summary(k+"NatureNormalizedBest", nature_normalized_score)


        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
                wandb.run.summary[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class MinibatchRlEvalWandb(MinibatchRlEval):

    def __init__(self, final_eval_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_eval_only = final_eval_only

    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size
        self.wandb_info = {'cum_steps': cum_steps}
        super().log_diagnostics(itr, eval_traj_infos, eval_time)
        wandb.log(self.wandb_info)

    def _log_infos(self, traj_infos=None):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    values = [info[k] for info in traj_infos]
                    logger.record_tabular_misc_stat(k,
                                                    values)

                    wandb.run.summary[k] = np.average(values)
                    self.wandb_info[k + "Average"] = np.average(values)
                    self.wandb_info[k + "Std"] = np.std(values)
                    self.wandb_info[k + "Min"] = np.min(values)
                    self.wandb_info[k + "Max"] = np.max(values)
                    self.wandb_info[k + "Median"] = np.median(values)
                    if k == 'GameScore':
                        game = self.sampler.env_kwargs['game']
                        random_score = atari_random_scores[game]
                        der_score = atari_der_scores[game]
                        nature_score = atari_nature_scores[game]
                        human_score = atari_human_scores[game]
                        normalized_score = (np.average(values) - random_score) / (human_score - random_score)
                        der_normalized_score = (np.average(values) - random_score) / (der_score - random_score)
                        nature_normalized_score = (np.average(values) - random_score) / (nature_score - random_score)
                        self.wandb_info[k + "Normalized"] = normalized_score
                        self.wandb_info[k + "DERNormalized"] = der_normalized_score
                        self.wandb_info[k + "NatureNormalized"] = nature_normalized_score

                        maybe_update_summary(k+"Best", np.average(values))
                        maybe_update_summary(k+"NormalizedBest", normalized_score)
                        maybe_update_summary(k+"DERNormalizedBest", der_normalized_score)
                        maybe_update_summary(k+"NatureNormalizedBest", nature_normalized_score)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
                wandb.run.summary[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)

    def evaluate_agent(self, itr):
        """
        Record offline evaluation of agent performance, by ``sampler.evaluate_agent()``.
        """
        if itr > 0:
            self.pbar.stop()

        if self.final_eval_only:
            eval = itr == 0 or itr >= self.n_itr - 1
        else:
            eval = itr == 0 or itr >= self.min_itr_learn - 1
        if eval:
            logger.log("Evaluating agent...")
            self.agent.eval_mode(itr)  # Might be agent in sampler.
            eval_time = -time.time()
            traj_infos = self.sampler.evaluate_agent(itr)
            eval_time += time.time()
        else:
            traj_infos = []
            eval_time = 0.0
        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time

    def train(self):
        """
        Performs startup, evaluates the initial agent, then loops by
        alternating between ``sampler.obtain_samples()`` and
        ``algo.optimize_agent()``.  Pauses to evaluate the agent at the
        specified log interval.
        """
        n_itr = self.startup()
        self.n_itr = n_itr
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
        self.shutdown()



class SyncRlEvalWandb(SyncRlMixin, MinibatchRlEvalWandb):
    """
    Multi-process RL with offline agent performance evaluation.  Only the
    master process runs agent evaluation.
    """

    @property
    def WorkerCls(self):
        return SyncWorkerEval

    def log_diagnostics(self, *args, **kwargs):
        super().log_diagnostics(*args, **kwargs)
        self.par.barrier.wait()


class PizeroCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms=51,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            framestack=4,
            grayscale=True,
            actions=False,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        f, c, h, w = image_shape
        self.conv = RepNet(f*c)
        # conv_out_size = self.conv.conv_out_size(h, w)
        # self.dyamics_network = TransitionModel(conv_out_size, num_actions)
        # self.reward_network = ValueNetwork(conv_out_size)
        if dueling:
            self.head = PizeroDistributionalDuelingHeadModel(256, output_size, hidden_size=1, pixels=36)
        else:
            self.head = PizeroDistributionalHeadModel(256, output_size, hidden_size=1, pixels=36)

    def forward(self, observation, prev_action, prev_reward):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        while len(observation.shape) <= 4:
            observation = observation.unsqueeze(0)
        observation = observation.flatten(-4, -3)
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        p = self.head(conv_out)
        p = F.softmax(p, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p.squeeze()


class PizeroSearchCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms=51,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            framestack=4,
            grayscale=True,
            actions=False,
            jumps=0,
            detach_model=True,
            nce=False,
            augmentation="none",
            target_augmentation=0,
            eval_augmentation=0,
            stack_actions=False,
            dynamics_blocks=16,
            film=False,
            norm_type="bn",
            encoder="repnet",
            noisy_nets=0,
            aug_prob=0.8,
            classifier="mlp",
            imagesize=84,
            time_contrastive=0,
            local_nce=False,
            global_nce=False,
            global_local_nce=False,
            buffered_nce=False,
            momentum_encoder=False,
            shared_encoder=False,
            padding='same',
            frame_dropout=0,
            keep_last_frame=True,
            cosine_nce=False,
            no_rl_augmentation=False,
            transition_model="standard",
            target_encoder_sn=False,
            use_all_targets=False,
            grad_scale_factor=0.5,
            hard_neg_factor=0,
            distributional=1,
            byol=0,
            dqn_hidden_size=256,
            byol_tau=0.01,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()

        self.noisy = noisy_nets
        self.time_contrastive = time_contrastive
        self.aug_prob = aug_prob
        self.classifier_type = classifier

        self.distributional = distributional
        n_atoms = 1 if not self.distributional else n_atoms
        self.dqn_hidden_size = dqn_hidden_size

        self.transforms = []
        self.eval_transforms = []

        self.uses_augmentation = False
        self.no_rl_augmentation = no_rl_augmentation
        for aug in augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                # Crashes if aug-prob not 1: use CenterCrop((84, 84)) or Resize((84, 84)) in that case.
                eval_transformation = CenterCrop((84, 84))
                self.uses_augmentation = True
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
                self.uses_augmentation = True
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((84, 84)))
                eval_transformation = nn.Identity()
            elif aug == "intensity":
                transformation = Intensity(scale=0.1)
                eval_transformation = nn.Identity()
            elif aug == "none":
                transformation = eval_transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        frame_dropout_fn = nn.Identity() if frame_dropout <= 0 else \
            lambda x: channel_dropout(x, frame_dropout, keep_last_frame)
        self.uses_augmentation = self.uses_augmentation or frame_dropout > 0
        self.transforms.append(frame_dropout_fn)
        self.eval_transforms.append(nn.Identity())

        self.dueling = dueling
        f, c, h, w = image_shape
        if encoder == "repnet":
            self.conv = RepNet(f*c, norm_type=norm_type)
        elif encoder == "curl":
            self.conv = CurlEncoder(f*c, norm_type=norm_type, padding=padding)
        elif encoder == "midsize":
            self.conv = SmallEncoder(256, f*c, norm_type=norm_type)
        elif encoder == "nature":
            self.conv = Conv2dModel(
                in_channels=f*c,
                channels=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                paddings=[0, 0, 0],
                use_maxpool=False,
            )
        elif encoder == "deepnature":
            self.conv = Conv2dModel(
                in_channels=f*c,
                channels=[32, 64, 64, 64, 64],
                kernel_sizes=[8, 4, 3, 3, 3],
                strides=[4, 2, 1, 1, 1],
                paddings=[0, 0, 0, 1, 1],
                use_maxpool=False,
            )
        elif encoder == "bignature":
            self.conv = Conv2dModel(
                in_channels=f*c,
                channels=[32, 64, 128, 128],
                kernel_sizes=[8, 4, 3, 3],
                strides=[4, 2, 1, 1],
                paddings=[0, 0, 0, 1],
                use_maxpool=False,
            )
        elif encoder == "impala":
            self.conv = ImpalaCNN(f*c, depths=[16, 32, 64],
                                  norm_type=norm_type)
        elif encoder == "effnet":
            self.conv = RLEffNet(imagesize,
                                 in_channels=f*c,
                                 norm_type=norm_type,)

        fake_input = torch.zeros(1, f*c, imagesize, imagesize)
        fake_output = self.conv(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        self.jumps = jumps
        self.grad_scale_factor = grad_scale_factor
        self.detach_model = detach_model
        self.use_nce = nce
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.stack_actions = stack_actions
        self.num_actions = output_size
        self.hard_neg_factor = hard_neg_factor

        if encoder in ["repnet", "midsize"]:
            if dueling:
                self.head = PizeroDistributionalDuelingHeadModel(self.hidden_size, output_size,
                                                                 pixels=self.pixels,
                                                                 norm_type=norm_type,
                                                                 noisy=self.noisy,
                                                                 n_atoms=n_atoms)
            else:
                self.head = PizeroDistributionalHeadModel(self.hidden_size, output_size,
                                                          pixels=self.pixels,
                                                          norm_type=norm_type,
                                                          noisy=self.noisy,
                                                          n_atoms=n_atoms)
        else:
            if dueling:
                self.head = DQNDistributionalDuelingHeadModel(self.hidden_size,
                                                              output_size,
                                                              hidden_size=self.dqn_hidden_size,
                                                              pixels=self.pixels,
                                                              noisy=self.noisy,
                                                              n_atoms=n_atoms)
            else:
                self.head = DQNDistributionalHeadModel(self.hidden_size,
                                                       output_size,
                                                       hidden_size=self.dqn_hidden_size,
                                                       pixels=self.pixels,
                                                       noisy=self.noisy,
                                                       n_atoms=n_atoms)

        if transition_model == "film":
            dynamics_model = FiLMTransitionModel
        elif transition_model == "effnet":
            dynamics_model = EffnetTransitionModel
        else:
            dynamics_model = TransitionModel

        if self.jumps > 0 or not self.detach_model:
            self.dynamics_model = dynamics_model(channels=self.hidden_size,
                                                 num_actions=output_size,
                                                 pixels=self.pixels,
                                                 hidden_size=self.hidden_size,
                                                 limit=1,
                                                 blocks=dynamics_blocks,
                                                 norm_type=norm_type,
                                                 renormalize=encoder=="repnet")
        else:
            self.dynamics_model = nn.Identity()

        if self.use_nce:
            self.local_nce = local_nce
            self.global_nce = global_nce
            self.global_local_nce = global_local_nce
            self.momentum_encoder = momentum_encoder
            self.momentum_tau = byol_tau
            self.buffered_nce = buffered_nce
            self.shared_encoder = shared_encoder
            assert self.local_nce or self.global_nce or self.global_local_nce
            assert not (self.shared_encoder and self.momentum_encoder)
            assert not (target_encoder_sn and self.momentum_encoder)

            assert hard_neg_factor == 0 or buffered_nce == 0
            assert hard_neg_factor == 0 or self.use_nce
            assert hard_neg_factor == 0 or self.jumps == 0 or use_all_targets

            # in case someone tries something silly like --local-nce 2
            self.num_nces = int(bool(self.local_nce)) + \
                            int(bool(self.global_nce)) +\
                            int(bool(self.global_local_nce))

            if self.local_nce:
                self.local_final_classifier = nn.Identity()
                if self.classifier_type == "mlp":
                    self.local_classifier = nn.Sequential(nn.Linear(self.hidden_size,
                                                                    self.hidden_size),
                                                          nn.BatchNorm1d(self.hidden_size),
                                                          nn.ReLU(),
                                                          nn.Linear(self.hidden_size,
                                                                    self.hidden_size))
                    self.local_final_classifier = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                                                nn.BatchNorm1d(self.hidden_size),
                                                                nn.ReLU(),
                                                                nn.Linear(self.hidden_size,
                                                                    self.hidden_size))
                elif self.classifier_type == "bilinear":
                    self.local_classifier = nn.Linear(self.hidden_size, self.hidden_size)
                elif self.classifier_type == "none":
                    self.local_classifier = nn.Identity()

                self.local_target_classifier = self.local_classifier
                local_buffer_size = self.hidden_size
                self.local_classifier.apply(weights_init)
            else:
                self.local_classifier = self.local_target_classifier = nn.Identity()
            if self.global_nce:
                self.global_final_classifier = nn.Identity()
                if self.classifier_type == "mlp":
                    self.global_classifier = nn.Sequential(
                                                nn.Flatten(-3, -1),
                                                nn.Linear(self.pixels*self.hidden_size, 512),
                                                nn.BatchNorm1d(512),
                                                nn.ReLU(),
                                                nn.Linear(512, 256)
                                                )
                    self.global_final_classifier = nn.Sequential(
                        nn.Linear(256, 512),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        nn.Linear(512, 256)
                    )
                    self.global_target_classifier = self.global_classifier
                    global_buffer_size = 256
                elif self.classifier_type == "q_l1":
                    self.global_classifier = nn.Sequential(*self.head.network[:2])
                    self.global_final_classifier = nn.Linear(256, 256)
                    self.global_target_classifier = self.global_classifier
                    global_buffer_size = 256
                elif self.classifier_type == "bilinear":
                    self.global_classifier = nn.Sequential(nn.Flatten(-3, -1),
                                                           nn.Linear(self.hidden_size*self.pixels,
                                                                     self.hidden_size*self.pixels))
                    self.global_target_classifier = nn.Flatten(-3, -1)
                elif self.classifier_type == "none":
                    self.global_classifier = nn.Flatten(-3, -1)
                    self.global_target_classifier = nn.Flatten(-3, -1)

                    global_buffer_size = self.hidden_size*self.pixels
                self.global_classifier.apply(weights_init)
            else:
                self.global_classifier = self.global_target_classifier = nn.Identity()
            if self.global_local_nce:
                if self.classifier_type == "mlp":
                    self.global_local_classifier = MLPHead(self.hidden_size,
                                                           output_size=self.hidden_size,
                                                           hidden_size=-1,
                                                           pixels=self.pixels,
                                                           noisy=0,
                                                           )
                elif self.classifier_type == "q_l1":
                    self.global_local_classifier = nn.Sequential(*self.head.network[:2],
                                                                 nn.Linear(256, self.hidden_size))
                    global_local_buffer_size = self.hidden_size
                else:
                    # Bilinear's size is different, since only comparing to one
                    # patch at a time.
                    self.global_local_classifier = nn.Sequential(nn.Flatten(-3, -1),
                                                                 nn.Linear(self.hidden_size*self.pixels,
                                                                           self.hidden_size))

                self.global_local_target_classifier = self.global_local_classifier
            else:
                self.global_local_classifier = self.global_local_target_classifier = nn.Identity()

            if self.momentum_encoder:
                self.nce_target_encoder = copy.deepcopy(self.conv)
                self.global_target_classifier = copy.deepcopy(self.global_target_classifier)
                self.local_target_classifier = copy.deepcopy(self.local_target_classifier)
                self.global_local_target_classifier = copy.deepcopy(self.global_local_target_classifier)
                for param in (list(self.nce_target_encoder.parameters())
                            + list(self.global_target_classifier.parameters())
                            + list(self.local_target_classifier.parameters())
                            + list(self.global_local_target_classifier.parameters())):
                    param.requires_grad = False

            elif not self.shared_encoder:
                # Use a separate target encoder on the last frame only.
                self.global_target_classifier = copy.deepcopy(self.global_target_classifier)
                self.local_target_classifier = copy.deepcopy(self.local_target_classifier)
                self.global_local_target_classifier = copy.deepcopy(self.global_local_target_classifier)
                if self.stack_actions:
                    input_size = c - 1
                else:
                    input_size = c
                if encoder == "curl":
                    self.nce_target_encoder = CurlEncoder(input_size,
                                                          norm_type=norm_type,
                                                          padding=padding)
                elif encoder == "nature" or encoder == "deepnature":
                    self.nce_target_encoder = Conv2dModel(in_channels=input_size,
                                                          channels=[32, 64, 64],
                                                          kernel_sizes=[8, 4, 3],
                                                          strides=[4, 2, 1],
                                                          paddings=[0, 0, 0],
                                                          use_maxpool=False,
                                                      )
                elif encoder == "bignature":
                    self.nce_target_encoder = Conv2dModel(in_channels=input_size,
                                                          channels=[32, 64, 128, 128],
                                                          kernel_sizes=[8, 4, 3, 1],
                                                          strides=[4, 2, 1, 1],
                                                          paddings=[0, 0, 0, 0],
                                                          use_maxpool=False,
                                                         )
                elif encoder == "effnet":
                    self.nce_target_encoder = RLEffNet(imagesize,
                                                       in_channels=input_size,
                                                       norm_type=norm_type,)
                else:
                    self.nce_target_encoder = SmallEncoder(self.hidden_size, input_size,
                                                           norm_type=norm_type)

            elif self.shared_encoder:
                self.nce_target_encoder = self.conv

            if target_encoder_sn:
                for module in (list(self.nce_target_encoder.modules())
                             + list(self.global_classifier.modules())
                             + list(self.global_target_classifier.modules())
                             + list(self.global_local_classifier.modules())
                             + list(self.global_local_target_classifier.modules())
                             + list(self.local_classifier.modules())
                             + list(self.local_target_classifier.modules())
                              ):
                    if hasattr(module, "weight"):
                        spectral_norm(module)

            if self.buffered_nce:
                if self.local_nce or self.global_local_nce:
                    self.local_nce = LocBufferedNCE(local_buffer_size, 2**10,
                                                    buffer_names=["main"],
                                                    n_locs=self.pixels)
                if self.global_nce:
                    self.global_nce = BufferedNCE(global_buffer_size, 2**10,
                                                  buffer_names=["main"])
                if self.global_local_nce:
                    self.global_local_nce = LocBufferedNCE(global_local_buffer_size,
                                                           2**10,
                                                           buffer_names=["main"],
                                                           n_locs=self.pixels)
            else:
                # This style of NCE can also be used for global-local with expand()
                self.nce = BlockNCE(normalize=cosine_nce,
                                    use_self_targets=use_all_targets
                                                     or self.jumps == 0,
                                    byol=byol)

        if self.detach_model:
            self.target_head = copy.deepcopy(self.head)
            for param in self.target_head.parameters():
                param.requires_grad = False

        print("Initialized model with {} parameters".format(count_parameters(self)))

    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)

    def do_global_nce(self, latents, target_latents, observation):
        global_latents = self.global_classifier(latents)
        global_latents = self.global_final_classifier(global_latents)
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            global_targets = self.global_target_classifier(target_latents)
        if not self.buffered_nce:
            # Need (locs, times, batch_size, rkhs) for the non-buffered nce
            # to mask out negatives from the same trajectory if desired.
            global_targets = global_targets.view(-1, observation.shape[1],
                                                 self.jumps+1, global_targets.shape[-1]).transpose(1, 2)
            global_latents = global_latents.view(-1, observation.shape[1]*(self.hard_neg_factor + 1),
                                                 self.jumps+1, global_latents.shape[-1]).transpose(1, 2)
            global_nce_loss, global_nce_accs = self.nce.forward(global_latents, global_targets)
        else:
            global_nce_loss, global_nce_accs = self.global_nce.forward(global_latents, global_targets)
            self.global_nce.update_buffer(global_targets)

        return global_nce_loss, global_nce_accs

    def do_local_nce(self, latents, target_latents, observation):
        local_latents = latents.flatten(-2, -1).permute(2, 0, 1)
        local_latents = self.local_classifier(local_latents)
        local_latents = self.local_final_classifier(local_latents)
        local_target_latents = target_latents.flatten(-2, -1).permute(2, 0, 1)
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            local_targets = self.local_target_classifier(local_target_latents)

        if not self.buffered_nce:
            if self.local_nce or self.global_local_nce:
                local_latents = local_latents.view(-1,
                                                   observation.shape[1]*(self.hard_neg_factor + 1),
                                                   self.jumps+1,
                                                   local_latents.shape[-1]).transpose(1, 2)
                local_targets = local_targets.view(-1,
                                                   observation.shape[1],
                                                   self.jumps+1,
                                                   local_targets.shape[-1]).transpose(1, 2)
                local_nce_loss, local_nce_accs = self.nce.forward(local_latents, local_targets)
        else:
            local_nce_loss, local_nce_accs = self.local_nce.forward(local_latents, local_targets)

        return local_nce_loss, local_nce_accs

    def do_global_local_nce(self, latents, target_latents, observation):
        local_latents = latents.flatten(-2, -1).permute(2, 0, 1)
        local_target_latents = target_latents.flatten(-2, -1).permute(2, 0, 1)
        global_latents = self.global_local_classifier(latents)

        if not self.buffered_nce:
            with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
                global_targets = self.global_local_target_classifier(target_latents)
            global_latents = global_latents.view(-1,
                                                 observation.shape[1]*(1+self.hard_neg_factor),
                                                 self.jumps + 1,
                                                 global_latents.shape[-1]).transpose(1, 2)
            global_targets = global_targets.view(-1,
                                                 observation.shape[1],
                                                 self.jumps + 1,
                                                 global_targets.shape[-1]).transpose(1, 2)
            local_latents = local_latents.view(local_latents.shape[0],
                                               observation.shape[1]*(1+self.hard_neg_factor),
                                               self.jumps+1,
                                               local_latents.shape[-1],
                                               ).transpose(1, 2)
            local_target_latents = local_target_latents.view(local_latents.shape[0],
                                                             observation.shape[1],
                                                             self.jumps+1,
                                                             local_latents.shape[-1],
                                                             ).transpose(1, 2)

            global_targets = global_targets.expand(local_latents.shape[0], -1, -1, -1)
            global_latents = global_latents.expand(local_latents.shape[0], -1, -1, -1)
            gl_nce_loss_1, gl_nce_accs_1 = self.nce.forward(local_latents, global_targets)
            gl_nce_loss_2, gl_nce_accs_2 = self.nce.forward(global_latents, local_target_latents)
            gl_nce_loss = 0.5 * (gl_nce_loss_1 + gl_nce_loss_2)
            gl_nce_accs = 0.5 * (gl_nce_accs_1 + gl_nce_accs_2)
        else:
            global_latents = global_latents.view(-1,
                                                 observation.shape[1]*(self.jumps + 1),
                                                 global_latents.shape[-1])
            global_latents = global_latents.expand(local_latents.shape[0], -1, -1)
            gl_nce_loss, gl_nce_accs = self.global_local_nce.forward(global_latents, local_target_latents)
            self.global_local_nce.update_buffer(local_target_latents)
        return gl_nce_loss, gl_nce_accs

    def add_hard_negatives(self, latents, actions, pred_qs=None):
        if self.hard_neg_factor <= 0:
            return latents, actions
        else:
            if pred_qs is None:
                hard_negs = torch.randint(0, self.num_actions, (self.hard_neg_factor*actions.shape[0],), dtype=torch.long, device=actions.device)
                neg_actions = actions[hard_negs]
            else:
                with torch.no_grad():
                    censored_qs = pred_qs.copy()
                    censored_qs[torch.arange(0, actions.shape[0], dtype=torch.long, device=actions.device), actions] = -10000.
                    best_actions = torch.argsort(censored_qs, -1, descending=True)
                    neg_actions = best_actions[:, :self.hard_neg_factor].transpose(0, 1).flatten()

            latents = torch.cat((self.hard_neg_factor+1)*[latents], 0)
            actions = torch.cat([actions, neg_actions], 0)
            return latents, actions

    def add_observation_negatives(self, observation):
        base_observation = observation[0]
        if self.hard_neg_factor <= 0:
            return base_observation

        hard_negs = torch.randint(1, observation.shape[0],
                                  (self.hard_neg_factor*observation.shape[1],),
                                  dtype=torch.long, device=observation.device)
        range = torch.arange(0, observation.shape[1], 1, dtype=torch.long, device=observation.device)
        range = range.repeat(self.hard_neg_factor)
        hard_negs = observation[hard_negs, range]
        return torch.cat([base_observation, hard_negs], 0)

    def do_nce(self, pred_latents, observation):
        pred_latents = torch.stack(pred_latents, 1)
        latents = pred_latents[:observation.shape[1]].flatten(0, 1)  # batch*jumps, *
        neg_latents = pred_latents[observation.shape[1]:].flatten(0, 1)
        latents = torch.cat([latents, neg_latents], 0)
        target_images = observation[self.time_contrastive:self.jumps + self.time_contrastive+1].transpose(0, 1).flatten(2, 3)
        target_images = self.transform(target_images, True)

        if not self.momentum_encoder and not self.shared_encoder:
            target_images = target_images[..., -1:, :, :]
        with torch.no_grad() if self.momentum_encoder else dummy_context_mgr():
            target_latents = self.nce_target_encoder(target_images.flatten(0, 1))

        if self.global_local_nce:
            gl_nce_loss, gl_nce_accs = self.do_global_local_nce(latents, target_latents, observation)
        else:
            gl_nce_loss = gl_nce_accs = 0
        if self.local_nce:
            local_nce_loss, local_nce_accs = self.do_local_nce(latents, target_latents, observation)
        else:
            local_nce_loss = local_nce_accs = 0
        if self.global_nce:
            global_nce_loss, global_nce_accs = self.do_global_nce(latents, target_latents, observation)
        else:
            global_nce_loss = global_nce_accs = 0

        nce_loss = (global_nce_loss + local_nce_loss + gl_nce_loss)/self.num_nces
        nce_accs = (global_nce_accs + local_nce_accs + gl_nce_accs)/self.num_nces

        nce_loss = nce_loss.view(-1, observation.shape[1]) # split to batch, jumps

        if self.momentum_encoder:
            update_state_dict(self.nce_target_encoder,
                              self.conv.state_dict(),
                              self.momentum_tau)
            if self.classifier_type != "bilinear":
                # q_l1 is also bilinear for local
                if self.local_nce and self.classifier_type != "q_l1":
                    update_state_dict(self.local_target_classifier,
                                      self.local_classifier.state_dict(),
                                      self.momentum_tau)
                if self.global_nce:
                    update_state_dict(self.global_target_classifier,
                                      self.global_classifier.state_dict(),
                                      self.momentum_tau)
                if self.global_local_nce:
                    update_state_dict(self.global_local_target_classifier,
                                      self.global_local_classifier.state_dict(),
                                      self.momentum_tau)
        return nce_loss, nce_accs

    def apply_transforms(self, transforms, eval_transforms, image):
        if eval_transforms is None:
            for transform in transforms:
                image = transform(image)
        else:
            for transform, eval_transform in zip(transforms, eval_transforms):
                image = maybe_transform(image, transform,
                                        eval_transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False):
        images = images.float()/255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        if augment:
            processed_images = self.apply_transforms(self.transforms,
                                                     self.eval_transforms,
                                                     flat_images)
        else:
            processed_images = self.apply_transforms(self.eval_transforms,
                                                     None,
                                                     flat_images)
        processed_images = processed_images.view(*images.shape[:-3],
                                                 *processed_images.shape[1:])
        return processed_images

    def stem_parameters(self):
        return list(self.conv.parameters()) + list(self.head.parameters())

    def stem_forward(self, img, prev_action, prev_reward):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        return conv_out

    def head_forward(self,
                     conv_out,
                     prev_action,
                     prev_reward,
                     target=False,
                     logits=False):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        if target:
            p = self.target_head(conv_out)
        else:
            p = self.head(conv_out)

        if self.distributional:
            if logits:
                p = F.log_softmax(p, dim=-1)
            else:
                p = F.softmax(p, dim=-1)
        else:
            p = p.squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    def forward(self, observation, prev_action, prev_reward, train=False):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        if train:
            log_pred_ps = []
            pred_reward = []
            pred_latents = []
            pred_nce_latents = []
            input_obs = self.add_observation_negatives(observation).flatten(1, 2)
            input_obs = self.transform(input_obs, not self.no_rl_augmentation)
            latent = self.stem_forward(input_obs,
                                       prev_action[0],
                                       prev_reward[0])
            pred_latents.append(latent[:observation.shape[1]])
            pred_nce_latents.append(latent)
            if self.jumps > 0 or not self.detach_model:
                if self.detach_model:
                    self.target_head.load_state_dict(self.head.state_dict())
                    latent = latent.detach()
                pred_rew = self.dynamics_model.reward_predictor(pred_latents[0])
                pred_reward.append(F.log_softmax(pred_rew, -1))

                for j in range(1, self.jumps + 1):
                    latent, action = self.add_hard_negatives(latent[:observation.shape[1]], prev_action[j])
                    latent, pred_rew = self.step(latent, action)
                    pred_rew = pred_rew[:observation.shape[1]]
                    latent = ScaleGradient.apply(latent, self.grad_scale_factor)
                    pred_latents.append(latent[:observation.shape[1]])
                    pred_nce_latents.append(latent)
                    pred_reward.append(F.log_softmax(pred_rew, -1))

            for i in range(len(pred_latents)):
                log_pred_ps.append(self.head_forward(pred_latents[i],
                                                     prev_action[i],
                                                     prev_reward[i],
                                                     target=self.detach_model and i > 0,
                                                     logits=True))

            if self.use_nce:
                if self.no_rl_augmentation and self.uses_augmentation:
                    pred_nce_latents = []
                    input_obs = self.transform(input_obs, True)
                    nce_latent = self.stem_forward(input_obs,
                                                   prev_action[0],
                                                   prev_reward[0])
                    pred_nce_latents.append(nce_latent)
                    for j in range(1, self.jumps + 1):
                        nce_latent, _ = self.dynamics_model(nce_latent, prev_action[j])
                        nce_latent = ScaleGradient.apply(nce_latent, self.grad_scale_factor)
                        pred_nce_latents.append(nce_latent)
                nce_loss, nce_accs = self.do_nce(pred_nce_latents, observation)
            else:
                nce_loss = torch.zeros((self.jumps + 1, observation.shape[1]), device=latent.device)
                nce_accs = torch.zeros(1,)

            return log_pred_ps,\
                   pred_reward,\
                   nce_loss,\
                   nce_accs

        else:
            observation = observation.flatten(-4, -3)
            stacked_observation = observation.unsqueeze(1).repeat(1, max(1, self.target_augmentation), 1, 1, 1)
            stacked_observation = stacked_observation.view(-1, *observation.shape[1:])

            img = self.transform(stacked_observation, self.target_augmentation)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            p = self.head(conv_out)

            p = p.view(observation.shape[0],
                       max(1, self.target_augmentation),
                       *p.shape[1:]).mean(1)

            if self.distributional:
                p = F.softmax(p, dim=-1)
            else:
                p = p.squeeze(-1)

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)

            return p

    def initial_inference(self, obs, actions=None, logits=False):
        if len(obs.shape) == 5:
            obs = obs.flatten(1, 2)
        obs = self.transform(obs, self.eval_augmentation)
        hidden_state = self.conv(obs)
        value_logits = self.head(hidden_state)

        # reward_logits = self.dynamics_model.reward_predictor(hidden_state)

        if logits:
            return hidden_state, value_logits

        if self.distributional:
            value = from_categorical(value_logits, logits=True, limit=10) #TODO Make these configurable
        else:
            value = value_logits.squeeze(-1)
        return hidden_state, value

    def inference(self, state, action):
        next_state, reward_logits, value_logits = self.step(state, action)
        value_logits = self.head(next_state)
        if self.distributional:
            value = from_categorical(value_logits, logits=True, limit=10) #TODO Make these configurable
        else:
            value = value_logits.squeeze(-1)
        reward = from_categorical(reward_logits, logits=True, limit=1)

        return next_state, reward, value

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        return next_state, reward_logits


class MLPHead(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=-1,
                 pixels=30,
                 noisy=0):
        super().__init__()
        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        self.noisy = noisy
        if hidden_size <= 0:
            hidden_size = input_channels*pixels
        self.linears = [linear(input_channels*pixels, hidden_size),
                        linear(hidden_size, output_size)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        if not noisy:
            self.network.apply(weights_init)
        self._output_size = output_size

    def forward(self, input):
        return self.network(input)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class DQNDistributionalHeadModel(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=256,
                 pixels=30,
                 n_atoms=51,
                 noisy=0):
        super().__init__()
        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        self.linears = [linear(input_channels*pixels, hidden_size),
                        linear(hidden_size, output_size * n_atoms)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        if not noisy:
            self.network.apply(weights_init)
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.network(input).view(-1, self._output_size, self._n_atoms)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class DQNDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 pixels=30,
                 n_atoms=51,
                 hidden_size=256,
                 grad_scale=2 ** (-1 / 2),
                 noisy=0):
        super().__init__()
        linear = NoisyLinear if noisy else nn.Linear
        self.linears = [linear(pixels * input_channels, hidden_size),
                        linear(hidden_size, output_size * n_atoms),
                        linear(pixels * input_channels, hidden_size),
                        linear(hidden_size, n_atoms)
                        ]
        self.advantage_layers = [nn.Flatten(-3, -1),
                                 self.linears[0],
                                 nn.ReLU(),
                                 self.linears[1]]
        self.value_layers = [nn.Flatten(-3, -1),
                             self.linears[2],
                             nn.ReLU(),
                             self.linears[3]]
        self.advantage_hidden = nn.Sequential(*self.advantage_layers[:3])
        self.advantage_out = self.advantage_layers[3]
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value = nn.Sequential(*self.value_layers)
        self.network = self.advantage_hidden
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class PizeroDistributionalHeadModel(torch.nn.Module):
    """An MLP head which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=128,
                 pixels=30,
                 n_atoms=51,
                 norm_type="bn",
                 noisy=False):
        super().__init__()
        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        self.hidden_size = hidden_size
        self.linears = [linear(pixels*hidden_size, 256),
                        linear(256, output_size * n_atoms)]
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.network(input).view(-1, self._output_size, self._n_atoms)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class PizeroDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 hidden_size=128,
                 pixels=30,
                 n_atoms=51,
                 grad_scale=2 ** (-1 / 2),
                 norm_type="bn",
                 noisy=False):
        super().__init__()
        self.hidden_size = hidden_size
        linear = NoisyLinear if noisy else nn.Linear
        self.linears = [linear(pixels*hidden_size, 256),
                        linear(256, n_atoms),
                        linear(pixels * hidden_size,
                               output_size * n_atoms, bias=False)
                        ]
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type),
                  nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.advantage_hidden = nn.Sequential(
            nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
            nn.ReLU(),
            init_normalization(hidden_size, norm_type),
            nn.Flatten(-3, -1))
        self.advantage_out = self.linears[2]
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value = nn.Sequential(*layers)
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()


def weights_init(m):
    if isinstance(m, Conv2dSame):
        torch.nn.init.kaiming_uniform_(m.layer.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.layer.bias)
    elif isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.bias)


class CurlEncoder(nn.Module):
    def __init__(self,
                 input_channels,
                 norm_type="bn", padding='same'):
        super().__init__()
        self.input_channels = input_channels
        if padding == 'same':
            self.main = nn.Sequential(
                Conv2dSame(self.input_channels, 32, 5, stride=5),  # 20x20
                nn.ReLU(),
                Conv2dSame(32, 64, 5, stride=5),  #4x4
                nn.ReLU())
        elif padding == 'valid':
            self.main = nn.Sequential(
                nn.Conv2d(self.input_channels, 32, 5, stride=5, padding=0),  # 20x20
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, stride=5, padding=0),  #4x4
                nn.ReLU())
        self.main.apply(weights_init)
        self.train()

    def forward(self, inputs):
        fmaps = self.main(inputs)
        return fmaps


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if not self.bias:
            self.bias_mu.fill_(0)
            self.bias_sigma.fill_(0)
        else:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Self.training alone isn't a good-enough check, since we may need to
        # activate .eval() during sampling even when we want to use noise
        # (due to batchnorm, dropout, or similar).
        # The extra "sampling" flag serves to override this behavior and causes
        # noise to be used even when .eval() has been called.
        if self.training or self.sampling:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class SmallEncoder(nn.Module):
    def __init__(self,
                 feature_size=256,
                 input_channels=4,
                 norm_type="bn"):
        super().__init__()
        self.feature_size = feature_size
        self.input_channels = input_channels
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(self.input_channels, 32, 8, stride=2, padding=4)),  # 50x50
            nn.ReLU(),
            init_normalization(32, norm_type),
            init_(nn.Conv2d(32, 64, 4, stride=2, padding=1)),  # 25x25
            nn.ReLU(),
            init_normalization(64, norm_type),
            init_(nn.Conv2d(64, 128, 4, stride=2, padding=1)),  # 12 x 12
            nn.ReLU(),
            init_normalization(128, norm_type),
            init_(nn.Conv2d(128, self.feature_size, 4, stride=2, padding=1)),  # 6 x 6
            nn.ReLU(),
            init_(nn.Conv2d(self.feature_size, self.feature_size,
                            1, stride=1, padding=0)),
            nn.ReLU())
        self.train()

    def forward(self, inputs):
        fmaps = self.main(inputs)
        return fmaps


class RepNet(nn.Module):
    def __init__(self, channels=3, norm_type="bn"):
        super().__init__()
        self.input_channels = channels
        layers = nn.ModuleList()
        hidden_channels = 128
        layers.append(nn.Conv2d(self.input_channels, hidden_channels,
                                kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(init_normalization(hidden_channels, norm_type))
        for _ in range(2):
            layers.append(ResidualBlock(hidden_channels,
                                        hidden_channels,
                                        norm_type))
        layers.append(nn.Conv2d(hidden_channels, hidden_channels * 2,
                                kernel_size=3, stride=2, padding=1))
        hidden_channels = hidden_channels * 2
        layers.append(nn.ReLU())
        layers.append(init_normalization(hidden_channels, norm_type))
        for _ in range(3):
            layers.append(ResidualBlock(hidden_channels,
                                        hidden_channels,
                                        norm_type))
        layers.append(nn.AvgPool2d(2))
        for _ in range(3):
            layers.append(ResidualBlock(hidden_channels,
                                        hidden_channels,
                                        norm_type))
        layers.append(nn.AvgPool2d(2))
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        if x.shape[-3] < self.input_channels:
            # We need to consolidate the framestack.
            x = x.flatten(-4, -3)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        latent = self.network(x)
        return renormalize(latent, 1)


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(Conv2dSame, self).__init__()
        self.F = kernel_size
        self.S = stride
        self.D = dilation
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation=dilation)

    def forward(self, x_in):
        N, C, H, W = x_in.shape
        H2 = int(np.ceil(H / self.S))
        W2 = int(np.ceil(W / self.S))
        Pr = (H2 - 1) * self.S + (self.F - 1) * self.D + 1 - H
        Pc = (W2 - 1) * self.S + (self.F - 1) * self.D + 1 - W
        x_pad = nn.ZeroPad2d((Pr//2, Pr - Pr//2, Pc//2, Pc - Pc//2))(x_in)
        x_out = self.layer(x_pad)
        return x_out


class ImpalaCNN(nn.Module):
    def __init__(self, input_channels,
                 depths=(16, 32, 32, 32),
                 norm_type="bn"):
        super(ImpalaCNN, self).__init__()
        self.depths = [input_channels] + depths
        self.layers = []
        self.norm_type = norm_type
        for i in range(len(depths)):
            self.layers.append(self._make_layer(self.depths[i],
                                                self.depths[i+1]))
        self.layers = nn.Sequential(*self.layers)
        self.train()

    def _make_layer(self, in_channels, depth):
        return nn.Sequential(
            Conv2dSame(in_channels, depth, 3),
            nn.MaxPool2d(3, stride=2),
            nn.ReLU(),
            ResidualBlock(depth, depth, norm_type=self.norm_type),
            nn.ReLU(),
            ResidualBlock(depth, depth, norm_type=self.norm_type)
        )

    @property
    def local_layer_depth(self):
        return self.depths[-2]

    def forward(self, inputs, fmaps=False):
        return self.layers(inputs)


def maybe_transform(image, transform, alt_transform, p=0.8):
    processed_images = transform(image)
    if p >= 1:
        return processed_images
    else:
        base_images = alt_transform(image)
        mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                          device=processed_images.device)
        mask = (mask < p).float()
        processed_images = mask * processed_images + (1 - mask) * base_images
        return processed_images

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

