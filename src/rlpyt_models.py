import torch
import torch.nn.functional as F
import torch.nn as nn

from rlpyt.models.dqn.atari_catdqn_model import DistributionalHeadModel
from rlpyt.models.dqn.dueling import DistributionalDuelingHeadModel
from rlpyt.models.utils import scale_grad, update_state_dict
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.runners.sync_rl import SyncRlMixin, SyncWorkerEval
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from src.model_trainer import ValueNetwork, TransitionModel, \
    NetworkOutput, from_categorical, ScaleGradient, init, \
    ResidualBlock, renormalize, FiLMTransitionModel, init_normalization
from src.buffered_nce import BufferedNCE, LocBufferedNCE, BlockNCE
import numpy as np
from kornia.augmentation import RandomAffine,\
    RandomCrop,\
    CenterCrop, \
    RandomResizedCrop
from kornia.filters import GaussianBlur2d
from rlpyt.utils.logging import logger
import copy
import wandb
import time


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

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class MinibatchRlEvalWandb(MinibatchRlEval):
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
                    self.wandb_info[k + "Average"] = np.average(values)
                    self.wandb_info[k + "Std"] = np.std(values)
                    self.wandb_info[k + "Min"] = np.min(values)
                    self.wandb_info[k + "Max"] = np.max(values)
                    self.wandb_info[k + "Median"] = np.median(values)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


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


class SerialEvalCollectorFixed(SerialEvalCollector):
    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
                                     len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        done_idxs = set()
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                                   agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    if b not in done_idxs:
                        completed_traj_infos.append(traj_infos[b].terminate(o))
                    done_idxs.add(b)
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (self.max_trajectories is not None and
                    len(completed_traj_infos) >= self.max_trajectories):
                logger.log("Evaluation reached max num trajectories "
                           f"({self.max_trajectories}).")
                break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                       f"({self.max_T}).")
        return completed_traj_infos


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
            self.head = PizeroDistributionalDuelingHeadModel(256, output_size, pixels=36)
        else:
            self.head = PizeroDistributionalHeadModel(256, output_size, pixels=36)

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
            buffered_nce=False,
            momentum_encoder=False,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()

        self.noisy = noisy_nets
        self.time_contrastive = time_contrastive
        self.aug_prob = aug_prob
        self.classifier_type = classifier

        self.transforms = []
        self.eval_transforms = []
        for aug in augmentation:
            assert aug in ["affine", "crop", "rrc", "blur", "none"]
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
                eval_transformation = nn.Identity()
            elif aug == "crop":
                transformation = RandomCrop((84, 84))
                eval_transformation = CenterCrop((84, 84))
                imagesize = 84
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
                eval_transformation = nn.Identity()
            elif aug == "blur":
                transformation = GaussianBlur2d((5, 5), (1.5, 1.5))
                eval_transformation = nn.Identity()
            else:
                transformation = eval_transformation = nn.Identity()
            self.transforms.append(transformation)
            self.eval_transforms.append(eval_transformation)

        self.dueling = dueling
        f, c, h, w = image_shape
        assert encoder in ["repnet", "curl", "midsize"]
        if encoder == "repnet":
            self.conv = RepNet(f*c, norm_type=norm_type)
            self.pixels = int(np.floor(imagesize/16.))**2
            self.hidden_size = 256
        if encoder == "curl":
            self.conv = CurlEncoder(f*c, norm_type=norm_type)
            self.pixels = int(np.ceil(imagesize/25.))**2
            self.hidden_size = 64
        if encoder == "midsize":
            self.conv = SmallEncoder(256, f*c, norm_type=norm_type)
            self.pixels = int(np.floor(imagesize/16.))**2
            self.hidden_size = 256
        self.jumps = jumps
        self.detach_model = detach_model
        self.use_nce = nce
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.stack_actions = stack_actions

        if encoder in ["repnet", "midsize"]:
            if dueling:
                self.head = PizeroDistributionalDuelingHeadModel(self.hidden_size, output_size,
                                                                 pixels=self.pixels,
                                                                 norm_type=norm_type,
                                                                 noisy=self.noisy)
            else:
                self.head = PizeroDistributionalHeadModel(self.hidden_size, output_size,
                                                          pixels=self.pixels,
                                                          norm_type=norm_type,
                                                          noisy=self.noisy)
        else:
            if dueling:
                self.head = DQNDistributionalDuelingHeadModel(self.hidden_size,
                                                              output_size,
                                                              hidden_size=256,
                                                              pixels=self.pixels,
                                                              noisy=self.noisy)
            else:
                self.head = DQNDistributionalHeadModel(self.hidden_size,
                                                       output_size,
                                                       hidden_size=256,
                                                       pixels=self.pixels,
                                                       noisy=self.noisy)

        if film:
            dynamics_model = FiLMTransitionModel
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
            self.momentum_encoder = momentum_encoder
            self.buffered_nce = buffered_nce

            if self.local_nce:
                if classifier == "mlp":
                    self.classifier = nn.Sequential(nn.Linear(self.hidden_size,
                                                              self.hidden_size),
                                                    nn.ReLU(),
                                                    nn.Linear(self.hidden_size,
                                                              self.hidden_size))
                else:
                    self.classifier = nn.Linear(self.hidden_size, self.hidden_size)

                buffer_size = self.hidden_size
            else:
                if classifier == "mlp":
                    self.classifier = MLPHead(self.hidden_size,
                                              output_size=self.hidden_size,
                                              hidden_size=-1,
                                              pixels=self.pixels,
                                              noisy=0,
                                              )
                    buffer_size = self.hidden_size
                else:
                    self.classifier = nn.Sequential(nn.Flatten(-3, -1),
                                                    nn.Linear(self.hidden_size*self.pixels,
                                                              self.hidden_size*self.pixels))
                    buffer_size = self.hidden_size*self.pixels

            if classifier == "mlp":
                self.target_classifier = copy.deepcopy(self.classifier)
            else:
                self.target_classifier = nn.Identity() if self.local_nce else nn.Flatten(-3, -1)

            if self.momentum_encoder:
                self.nce_target_encoder = copy.deepcopy(self.conv)
                for param in self.target_classifier.parameters():
                    param.requires_grad = False
                for param in self.nce_target_encoder.parameters():
                    param.requires_grad = False
            else:
                if self.stack_actions:
                    input_size = c - 1
                else:
                    input_size = c
                if encoder == "curl":
                    self.nce_target_encoder = CurlEncoder(input_size, norm_type=norm_type)
                else:
                    self.nce_target_encoder = SmallEncoder(self.hidden_size, input_size,
                                                           norm_type=norm_type)
            if self.buffered_nce:
                if self.local_nce:
                    self.nce = LocBufferedNCE(buffer_size, 2**12, buffer_names=["main"], n_locs=self.pixels)
                else:
                    self.nce = BufferedNCE(buffer_size, 2**12, buffer_names=["main"])
            else:
                self.nce = BlockNCE(normalize=False)

        if self.detach_model:
            self.target_head = copy.deepcopy(self.head)
            for param in self.target_head.parameters():
                param.requires_grad = False

    def do_nce(self, pred_latents, observation):
        stacked_latents = torch.stack(pred_latents, 1).flatten(0, 1)
        if self.local_nce:
            stacked_latents = stacked_latents.flatten(-2, -1).permute(2, 0, 1)
        stacked_latents = self.classifier(stacked_latents)
        target_images = observation[self.time_contrastive:self.jumps + self.time_contrastive+1].transpose(0, 1).flatten(2, 3)
        target_images = self.transform(target_images, True)

        if self.momentum_encoder:
            with torch.no_grad():
                target_latents = self.nce_target_encoder(target_images.flatten(0, 1))
                if self.local_nce:
                    target_latents = target_latents.flatten(-2, -1).permute(2, 0, 1)
                target_latents = self.target_classifier(target_latents)
                update_state_dict(self.nce_target_encoder, self.conv.state_dict(), 0.001)
                if self.classifier_type != "bilinear":
                    update_state_dict(self.target_classifier, self.classifier.state_dict(), 0.001)
        else:
            target_images = target_images[..., -1, :, :]
            if len(target_images.shape) == 4:
                target_images = target_images.unsqueeze(2)
            target_latents = self.nce_target_encoder(target_images.flatten(0, 1))
            if self.local_nce:
                target_latents = target_latents.view(observation.shape[1], -1,
                                                     *target_latents.shape[1:])
                target_latents = target_latents.flatten(3, 4).permute(3, 1, 0, 2).flatten(1, 2)
            target_latents = self.target_classifier(target_latents)

        if not self.buffered_nce:
            # Need (locs, times, batch_size, rkhs) for the non-buffered nce
            # to mask out negatives from the same trajectory if desired.
            target_latents = target_latents.view(-1, observation.shape[1],
                                                 self.jumps+1, target_latents.shape[-1]).transpose(1, 2)
            stacked_latents = stacked_latents.view(-1, observation.shape[1],
                                                   self.jumps+1, stacked_latents.shape[-1]).transpose(1, 2)
        nce_loss, nce_accs = self.nce.forward(stacked_latents, target_latents)

        nce_loss = nce_loss.view(observation.shape[1], -1)
        if self.jumps > 0:
            nce_model_loss = nce_loss[:, 1:].mean(1)
            nce_loss = nce_loss[:, 0]
        else:
            nce_loss = nce_loss[:, 0]
            nce_model_loss = torch.zeros_like(nce_loss)
        nce_accs = nce_accs.mean().item()

        return nce_loss, nce_model_loss, nce_accs

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

    def head_forward(self, conv_out, prev_action, prev_reward, target=False):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)
        if target:
            p = self.target_head(conv_out)
        else:
            p = self.head(conv_out)
        p = F.softmax(p, dim=-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    def forward(self, observation, prev_action, prev_reward, train=False):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        if train:
            pred_ps = []
            pred_reward = []
            pred_latents = []
            input_obs = observation[0].flatten(1, 2)
            input_obs = self.transform(input_obs, True)

            latent = self.stem_forward(input_obs,
                                       prev_action[0],
                                       prev_reward[0])
            pred_ps.append(self.head_forward(latent,
                                             prev_action[0],
                                             prev_reward[0]),)
            pred_latents.append(latent)

            if self.jumps > 0 or not self.detach_model:
                if self.detach_model:
                    self.target_head.load_state_dict(self.head.state_dict())
                    latent = latent.detach()
                pred_rew = self.dynamics_model.reward_predictor(latent)
                pred_reward.append(F.log_softmax(pred_rew, -1))

                for j in range(1, self.jumps + 1):
                    latent, pred_rew, _, _ = self.step(latent, prev_action[j])
                    latent = ScaleGradient.apply(latent, 0.5)
                    pred_latents.append(latent)
                    pred_reward.append(pred_rew)
                    pred_ps.append(self.head_forward(latent,
                                                     prev_action[j],
                                                     prev_reward[j],
                                                     target=self.detach_model))

            if self.use_nce:
                nce_loss, nce_model_loss, nce_accs = self.do_nce(pred_latents, observation)
            else:
                nce_loss = nce_model_loss = nce_accs = torch.tensor(0.)

            return pred_ps,\
                   pred_reward,\
                   nce_loss, nce_model_loss, nce_accs

        else:
            observation = observation.flatten(-4, -3)
            stacked_observation = observation.unsqueeze(1).repeat(1, max(1, self.target_augmentation), 1, 1, 1)
            stacked_observation = stacked_observation.view(-1, *observation.shape[1:])

            img = self.transform(stacked_observation, self.target_augmentation)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            p = self.head(conv_out)
            p = F.softmax(p, dim=-1)

            p = p.view(observation.shape[0],
                       max(1, self.target_augmentation),
                       *p.shape[1:]).mean(1)

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)

            return p

    def initial_inference(self, obs, actions=None, logits=False):
        if len(obs.shape) == 5:
            obs = obs.flatten(1, 2)
        obs = self.transform(obs, self.eval_augmentation)
        hidden_state = self.conv(obs)
        policy_logits = None
        value_logits = self.head(hidden_state)

        # reward_logits = self.dynamics_model.reward_predictor(hidden_state)

        if logits:
            return hidden_state, None, policy_logits, value_logits

        value = from_categorical(value_logits, logits=True, limit=10) #TODO Make these configurable
        # reward = from_categorical(reward_logits, logits=True, limit=1)
        return hidden_state, None, policy_logits, value

    def inference(self, state, action):
        next_state, reward_logits, \
        policy_logits, value_logits = self.step(state, action)
        value = from_categorical(value_logits, logits=True, limit=10) #TODO Make these configurable
        reward = from_categorical(reward_logits, logits=True, limit=1)

        return NetworkOutput(next_state, reward, policy_logits, value)

    def step(self, state, action):
        next_state, reward_logits = self.dynamics_model(state, action)
        policy_logits = None
        value_logits = self.head(next_state)
        return next_state, reward_logits, policy_logits, value_logits


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
        if hidden_size <= 0:
            hidden_size = input_channels*pixels
        self.linears = [linear(input_channels*pixels, hidden_size),
                        linear(hidden_size, output_size)]
        layers = [nn.Flatten(-3, -1),
                  self.linears[0],
                  nn.ReLU(),
                  self.linears[1]]
        self.network = nn.Sequential(*layers)
        self._output_size = output_size

    def forward(self, input):
        return self.network(input)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()


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
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        return self.network(input).view(-1, self._output_size, self._n_atoms)

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()


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
                        linear(hidden_size, output_size * n_atoms, bias=False),
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
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms))
        self.value = nn.Sequential(*self.value_layers)
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
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms))
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


class CurlEncoder(nn.Module):
    def __init__(self,
                 input_channels,
                 norm_type="bn"):
        super().__init__()
        self.input_channels = input_channels
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.main = nn.Sequential(
            Conv2dSame(self.input_channels, 32, 5, stride=5),  # 20x20
            nn.ReLU(),
            Conv2dSame(32, 64, 5, stride=5),  #4x4
            nn.ReLU())
        self.train()

    def forward(self, inputs):
        fmaps = self.main(inputs)
        return fmaps


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class SmallEncoder(nn.Module):
    def __init__(self,
                 feature_size,
                 input_channels,
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


def maybe_transform(image, transform, alt_transform, p=0.8):
    base_images = alt_transform(image)
    processed_images = transform(image)
    mask = torch.rand((processed_images.shape[0], 1, 1, 1),
                      device=processed_images.device)
    mask = (mask < p).float()
    processed_images = mask * processed_images + (1 - mask) * base_images
    return processed_images
