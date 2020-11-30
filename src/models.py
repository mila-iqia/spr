import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.categorical import Categorical

from rlpyt.models.utils import update_state_dict
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, select_at_indexes
from src.utils import count_parameters, minimal_c51_loss, \
    average_targets, timeit, safe_log, c51_backup, clamp_probs, \
    from_categorical, scalar_backup, minimal_scalar_loss, get_augmentation, \
    maybe_transform
from src.networks import Conv2dModel, ImpalaCNN, RewardPredictor, DonePredictor,\
    TransitionModel, DQNDistributionalDuelingHeadModel, CosineEmbeddingNetwork,\
    init_normalization, Residual, InvertedResidual
import numpy as np
import copy
import wandb


class SPRCatDqnModel(torch.nn.Module):
    """2D conlutional network feeding into MLP with ``n_atoms`` outputs
    per action, representing a discrete probability distribution of Q-values."""

    def __init__(
            self,
            image_shape,
            output_size,
            n_atoms,
            dueling,
            jumps,
            spr,
            augmentation,
            target_augmentation,
            eval_augmentation,
            dynamics_blocks,
            block_drop_prob,
            norm_type,
            use_ws,
            noisy_nets,
            aug_prob,
            projections,
            imagesize,
            distributional,
            dqn_hidden_size,
            tm_hidden_size,
            momentum_tau,
            renormalize,
            selection_temperature,
            search_policy,
            q_l1_type,
            rollout_depth,
            dropout,
            predictor,
            model_rl,
            rollout_rl,
            noisy_nets_std,
            residual_tm,
            encoder,
            target_entropy=-1.,
            V_min=-10.,
            V_max=10.,
            discount=0.99,
            target_depth=5,
            eval_depth=0,
            target_runs=1,
            eval_runs=1,
            lambda_from_state=1,
            rollout_rl_type="offset",
            batch_thresh=0.1,
            lambda_type="exponential",
            separate_spr_pass=False,
            eval_quantiles=32,
            train_quantiles=8,
            resblock="inverted",
            expand_ratio=2,
            use_dones=True,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            framestack=4,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()

        self.rollout_rl_backup = rollout_rl_type == "backup"

        self.noisy = noisy_nets
        self.aug_prob = aug_prob
        self.rollout_depth = rollout_depth
        self.search_policy = search_policy
        self.selection_temp = nn.parameter.Parameter(torch.Tensor([1.])*selection_temperature)
        self.target_entropy = target_entropy

        self.V_min = V_min
        self.V_max = V_max
        self.discount = discount
        self.target_depth = target_depth
        self.eval_depth = eval_depth
        self.target_runs = target_runs
        self.eval_runs = eval_runs
        self.debug = False
        self.batch_thresh = batch_thresh
        self.separate_spr_pass = separate_spr_pass

        self.transforms = get_augmentation(augmentation, imagesize)
        self.target_transforms = get_augmentation(target_augmentation, imagesize)

        self.distributional = distributional and distributional != "quantile"
        self.quantile = distributional == "quantile"
        self.train_quantiles = train_quantiles
        self.eval_quantiles = eval_quantiles
        n_atoms = 1 if not self.distributional else n_atoms
        self.n_atoms = n_atoms
        self.dqn_hidden_size = dqn_hidden_size

        if resblock == "inverted":
            resblock = InvertedResidual
        else:
            resblock = Residual

        self.dueling = dueling
        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])
        if encoder == "impala":
            self.conv = ImpalaCNN(in_channels,
                                  depths=[32, 64, 64],
                                  strides=[3, 2, 2],
                                  norm_type=norm_type,
                                  use_ws=use_ws,
                                  resblock=resblock,
                                  expand_ratio=expand_ratio,
                                  drop_prob=block_drop_prob)

        elif encoder.lower() == "normednature":
            self.conv = Conv2dModel(
                in_channels=in_channels,
                channels=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                paddings=[0, 0, 0],
                use_maxpool=False,
                dropout=dropout,
                norm_type=norm_type,
                use_ws=use_ws,
            )
        else:
            self.conv = Conv2dModel(
                in_channels=in_channels,
                channels=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                paddings=[0, 0, 0],
                use_maxpool=False,
                dropout=dropout,
            )

        fake_input = torch.zeros(1, f*c, imagesize, imagesize)
        fake_output = self.conv(fake_input)
        self.latent_shape = fake_output.shape[1:]
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        if self.quantile:
            self.quantile_conditioning = CosineEmbeddingNetwork(embedding_dim=self.pixels*self.hidden_size)

        self.lambda_predictor = LambdaPredictor(state_size=(*fake_output.shape[1:],),
                                                rl_loss_function=minimal_c51_loss
                                                if self.distributional
                                                else minimal_scalar_loss,
                                                use_state=lambda_from_state,
                                                prediction_type=lambda_type,
                                                encoder=self.encode_images,
                                                rollout_depth=self.rollout_depth)
        self.jumps = jumps
        self.model_rl = model_rl
        self.rollout_rl = rollout_rl
        self.use_spr = spr
        self.target_augmentation = target_augmentation
        self.eval_augmentation = eval_augmentation
        self.num_actions = output_size

        if dueling:
            self.head = DQNDistributionalDuelingHeadModel(self.hidden_size,
                                                          output_size,
                                                          hidden_size=self.dqn_hidden_size,
                                                          pixels=self.pixels,
                                                          noisy=self.noisy,
                                                          n_atoms=n_atoms,
                                                          std_init=noisy_nets_std)
        else:
            raise NotImplementedError

        self.renormalize = init_normalization(self.hidden_size, renormalize)

        if self.jumps > 0:
            self.dynamics_model = TransitionModel(channels=self.hidden_size,
                                                  num_actions=output_size,
                                                  hidden_size=self.hidden_size
                                                  if tm_hidden_size <= 0
                                                  else tm_hidden_size,
                                                  blocks=dynamics_blocks,
                                                  norm_type=norm_type,
                                                  resblock=resblock,
                                                  expand_ratio=expand_ratio,
                                                  drop_prob=block_drop_prob,
                                                  renormalize=self.renormalize,
                                                  residual=residual_tm)
        else:
            self.dynamics_model = nn.Identity()

        self.reward_predictor = RewardPredictor(self.hidden_size,
                                                pixels=self.pixels,
                                                limit=1,
                                                norm_type=norm_type)

        self.use_dones = use_dones
        if use_dones:
            self.done_predictor = DonePredictor(self.hidden_size,
                                                num_actions=output_size,
                                                pixels=self.pixels,
                                                norm_type=norm_type)
        else:
            self.done_predictor = lambda x, y: torch.zeros(x.shape[0]).to(x)

        self.projections = nn.ModuleList()
        projection_dims = []
        self.momentum_tau = momentum_tau

        # Initialize projections even if SPR off -- may want to log diversity
        if isinstance(projections, str):
            projections = projections.split("_")
        for projection in projections:
            projection_encoders = []
            projection_dim = 0
            if "mlp" in projection:
                projection_encoders.append(nn.Sequential(
                                            nn.Flatten(-3, -1),
                                            nn.Linear(self.pixels*self.hidden_size, 512),
                                            nn.BatchNorm1d(512),
                                            nn.ReLU(),
                                            nn.Linear(512, 256)
                                            ))
                projection_dim += 256
            elif "value" in projection:
                projection_encoders.append(self.head.value_net[:2])
                projection_dim += self.dqn_hidden_size
            elif "advantage" in projection:
                projection_encoders.append(self.head.advantage_net[:2])
                projection_dim += self.dqn_hidden_size
            elif "q" in projection:
                projection_encoders.append(
                    nn.Sequential(self.head, nn.Flatten(-2, -1)),)
                projection_dim += (n_atoms*output_size)
            elif "reward" in projection:
                projection_encoders.append(self.reward_predictor.network[:5])
                projection_dim += 256
            elif "done" == projection:
                projection_encoders.append(
                    self.done_predictor.network[:5],
                )
                projection_dim += 256
            elif "none" in projection:
                self.projections_encoders.append(nn.Identity)
                projection_dim += (self.hidden_size*self.pixels)
            self.projections.append(Projection(projection_encoders,
                                               noisy="noisy" in q_l1_type,
                                               relu="relu" in q_l1_type,
                                               needs_action="done"==projection,
                                               num_actions=output_size))
            projection_dims.append(projection_dim)

            if self.use_spr:
                self.predictors = nn.ModuleList()
                for dim in projection_dims:
                    if predictor == "mlp":
                        self.predictors.append(nn.Sequential(
                            nn.Linear(dim, dim*2),
                            nn.BatchNorm1d(dim*2),
                            nn.ReLU(),
                            nn.Linear(dim*2, dim)
                        ))
                    elif predictor == "linear":
                        self.predictors.append(nn.Sequential(
                            nn.Linear(dim, dim),
                        ))
                    elif predictor == "none":
                        self.predictors.append(nn.Identity())

                self.target_encoder = copy.deepcopy(self.conv)
                self.target_projections = copy.deepcopy(self.projections)
                for param in (list(self.target_encoder.parameters()) +
                              list(self.target_projections.parameters())):
                    param.requires_grad = False

        print("Initialized model with {} parameters".format(count_parameters(self)))

    def calibration_loss(self, latents, actions, all_pred_ps):
        actions = actions.transpose(0, 1)
        with torch.no_grad():
            true_qs = self.head_forward(latents, noise_override=False, logits=False)
            if self.distributional:
                true_qs = from_categorical(true_qs, limit=10, logits=False)
            else:
                true_qs = true_qs.mean(-1)

        probs = F.log_softmax(true_qs/self.selection_temp, -1)

        with torch.no_grad():
            if self.search_policy == "on_policy":
                if self.distributional:
                    all_pred_ps = from_categorical(all_pred_ps, limit=10, logits=False)
                else:
                    all_pred_ps = all_pred_ps.mean(-1)
                target_actions = all_pred_ps.argmax(-1)
            elif self.search_policy == "off_policy":
                target_actions = actions.flatten(0, 1)
            elif self.search_policy == "target_entropy":
                mean_entropy = Categorical(probs=probs).entropy().mean()
                if self.target_entropy > 0:
                    if abs(mean_entropy - self.target_entropy) < 0.05:
                        pass
                    elif mean_entropy > self.target_entropy:
                        self.selection_temp = max(self.selection_temp * 0.99, 0.01)
                    else:
                        self.selection_temp = self.selection_temp * 1.010101
            elif self.search_policy == "fixed_t":
                return torch.tensor([0]).to(latents)

        loss = F.nll_loss(probs.flatten(0, 1),
                          target_actions,
                          reduction="none")

        return loss.view(*actions.shape).mean(1)

    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)

    @torch.no_grad()
    def calculate_diversity(self, latents, actions, observation):
        global_latents = self.projections[0](latents, actions)
        # shape is jumps, bs, dim
        global_latents = F.normalize(global_latents, p=2., dim=-1, eps=1e-3)
        cos_sim = torch.matmul(global_latents, global_latents.transpose(0, 1))
        mask = 1 - (torch.eye(cos_sim.shape[0], device=cos_sim.device, dtype=torch.float))

        cos_sim = cos_sim*mask
        offset = cos_sim.shape[-1]/(cos_sim.shape[-1] - 1)
        cos_sim = cos_sim.mean()*offset
        return cos_sim

    def encode_images(self, images, augment=True, target=False):
        images = self.transform(images, augment)
        latents = self.stem_forward(images.flatten(-4, -3), target=target)
        latents = self.renormalize(latents)
        return latents.view(images.shape[0],
                            images.shape[1],
                            *latents.shape[1:])

    @torch.no_grad()
    def action_selection(self,
                         q_values,
                         temperature=1.,
                         num_to_select=1,
                         adjust_temp=True,
                         ):

        if self.distributional:
            q_values = from_categorical(q_values, limit=10, logits=False)
        else:
            q_values = q_values.mean(-1)

        if temperature == 0:
            _, actions = torch.topk(q_values, num_to_select, -1)
            actions = actions.flatten(0, 1)
            probs = torch.zeros_like(q_values).unsqueeze(0).expand(num_to_select, *q_values.shape).flatten(0, 1)
            probs[torch.arange(actions.shape[0], device=actions.device,
                               dtype=torch.long), actions] = 1.
        else:
            probs = clamp_probs(F.softmax(q_values/temperature, -1))
            dist = Categorical(probs=probs)
            try:
                actions = dist.sample((num_to_select,))
            except Exception as e:
                # Occasionally we get bizarre crashes here in the first
                # iteration.  Probably nonrecoverable, so just print the values
                # to see what's going wrong, then raise the exception again.
                print(q_values)
                print(probs)
                raise e
            actions = actions.T.flatten(0, 1)
        return actions, probs

    def spr_loss(self, f_x1s, f_x2s):
        f_x1 = F.normalize(f_x1s.float(), p=2., dim=-1, eps=1e-3)
        f_x2 = F.normalize(f_x2s.float(), p=2., dim=-1, eps=1e-3)
        loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
        return loss

    def do_spr_loss(self, spatial_latents, spatial_targets, actions, observation):
        spr_loss = 0
        for predictor, projection, target_projection in zip(
            self.predictors, self.projections, self.target_projections
        ):
            preds = projection(spatial_latents, actions)
            preds = predictor(preds)
            with torch.no_grad():
                targets = target_projection(spatial_targets, actions)

            targets = targets.view(-1, observation.shape[1],
                                                 self.jumps+1, targets.shape[-1]).transpose(1, 2)
            preds = preds.view(-1, observation.shape[1],
                                 self.jumps+1,
                                 preds.shape[-1]).transpose(1, 2)

            spr_loss = spr_loss + self.spr_loss(preds, targets).view(-1, observation.shape[1]) # split to batch, jumps

        spr_loss = spr_loss / len(self.predictors)
        update_state_dict(self.target_encoder,
                          self.conv.state_dict(),
                          self.momentum_tau)
        update_state_dict(self.target_projections,
                          self.projections.state_dict(),
                          self.momentum_tau)
        return spr_loss

    def apply_transforms(self, transforms, image):
        for transform in transforms:
            image = maybe_transform(image, transform, p=self.aug_prob)
        return image

    @torch.no_grad()
    def transform(self, images, augment=False, target=False):
        images = images.float()/255. if images.dtype == torch.uint8 else images
        if augment:
            flat_images = images.reshape(-1, *images.shape[-3:])
            transforms = self.transforms if not target else self.target_transforms
            processed_images = self.apply_transforms(transforms,
                                                     flat_images)
            processed_images = processed_images.view(*images.shape[:-3],
                                                     *processed_images.shape[1:])
            return processed_images
        else:
            return images

    def stem_parameters(self):
        return list(self.conv.parameters()) + list(self.head.parameters())

    def model_parameters(self):
        params = list(self.dynamics_model.parameters()) +\
                 list(self.reward_predictor.parameters())
        if self.use_dones:
            params = params + list(self.done_predictor.parameters())

        return params

    def stem_forward(self, img, prev_action=None, prev_reward=None, target=False):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        if target:
            encoder = self.target_encoder
        else:
            encoder = self.conv

        conv_out = encoder(img.view(T * B, *img_shape))  # Fold if T dimension.
        conv_out = self.renormalize(conv_out)
        return conv_out

    def sample_taus(self, batch_size, device="cuda"):
        if self.training:
            num_quantiles = self.train_quantiles
        else:
            num_quantiles = self.eval_quantiles
        self.taus = torch.rand(batch_size,
                               num_quantiles,
                               dtype=torch.float32,
                               device=device)

    def head_forward(self,
                     conv_out,
                     prev_action=None,
                     prev_reward=None,
                     logits=False,
                     **kwargs,):
        lead_dim, T, B, img_shape = infer_leading_dims(conv_out, 3)

        if self.quantile:
            conv_out = self.quantile_conditioning(self.taus, conv_out)

        p = self.head(conv_out, **kwargs)
        if self.distributional:
            if logits:
                p = F.log_softmax(p, dim=-1)
            else:
                p = F.softmax(p, dim=-1)
        elif self.quantile:
            p = p.squeeze(-1)
            p = p.view(T*B, self.taus.shape[1], p.shape[-1])
            p = p.permute(0, 2, 1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = restore_leading_dims(p, lead_dim, T, B)
        return p

    @torch.no_grad()
    def adaptive_depth(self, weights, cutoff=0.9):
        cum_weights = torch.cumsum(weights, -1)
        valid = (cum_weights > cutoff).float().mean(0) > self.batch_thresh
        depth = torch.min(torch.nonzero(valid.float(), as_tuple=False))

        return depth.item()

    def backup(self, *args):
        if not self.distributional:
            return scalar_backup(*args, discount=self.discount)
        else:
            return c51_backup(*args,
                              V_max=self.V_max,
                              V_min=self.V_min,
                              discount=self.discount)

    def model_rollout(self,
                      input,
                      depth,
                      override_actions=None,
                      encode=False,
                      augment=False,
                      backup=True,
                      adaptive_depth=False,
                      runs=1):
        """
        :param input: Spatial latents at t=0 or image to encode at t=0
        :param depth: How many states to roll out over.
        :param action_selection_temp: Temperature to use for boltzmann action
                selection.  Uses argmax if temp=0.
        :param actions: override action selection with a fixed set of actions
        :param encoder: Whether or not to use the encoder
        :param augment: Whether or not to augment, if using the encoder.
        :return: latents, actions, rewards, dones, values and lambas
        """
        if self.debug:
            import ipdb; ipdb.set_trace()
        if encode:
            latent = self.encode_images(input, augment=augment, target=False)[0]
        else:
            latent = input
        with torch.no_grad():
            lambdas = self.lambda_predictor(latent)
            if adaptive_depth:
                depth = self.adaptive_depth(lambdas)
        pred_latents = []
        pred_rewards = []
        pred_dones = []
        pred_values = []
        all_pred_values = []
        actions = override_actions if override_actions is not None else []

        value_q_t = self.head_forward(latent, logits=False)
        all_pred_values.append(value_q_t)

        if override_actions is None:
            selection_q_t = self.head_forward(latent, logits=False, noise_override=False)
            action, probs = self.action_selection(selection_q_t,
                                                  temperature=self.selection_temp,
                                                  num_to_select=runs)
            actions.append(action)

        if runs > 1:
            assert override_actions is None
            latent = latent.unsqueeze(1).expand(-1, runs, *([-1]*len(latent.shape[1:])))
            latent = latent.flatten(0, 1)
            value_q_t = value_q_t.unsqueeze(1).expand(-1, runs, *([-1]*len(value_q_t.shape[1:])))
            value_q_t = value_q_t.flatten(0, 1)
            lambdas = lambdas.unsqueeze(1).expand(-1, runs, *([-1]*len(lambdas.shape[1:])))
            lambdas = lambdas.flatten(0, 1)

        # Don't take expectation until inside the rollout (i.e., e-sarsa)
        value_q_t = select_at_indexes(actions[0], value_q_t)

        pred_values.append(value_q_t)
        pred_latents.append(latent)
        if depth > 0:
            for i in range(depth):
                action = actions[i]
                pred_dones.append(self.done_predictor(latent, action))
                latent = self.dynamics_model(latent, action)
                reward = self.reward_predictor(latent)
                value_q_t = self.head_forward(latent, logits=False)

                pred_rewards.append(reward)
                pred_latents.append(latent)
                all_pred_values.append(value_q_t)

                # Check to see if the next action is already known,
                # before choosing a new one
                if len(actions) == i+1:
                    selection_q_t = self.head_forward(latent, logits=False, noise_override=False)
                    action, probs = self.action_selection(selection_q_t, temperature=self.selection_temp)
                    actions.append(action)
                    value_q_t = (value_q_t*probs.unsqueeze(-1)).sum(1)
                else:
                    value_q_t = select_at_indexes(actions[i+1], value_q_t)

                pred_values.append(value_q_t)

            # Consolidate results
            pred_dones = torch.stack(pred_dones, 0)
            pred_rewards = F.log_softmax(torch.stack(pred_rewards, 0), -1)
            pred_values = torch.stack(pred_values, 1)

            if backup:
                rewards = from_categorical(torch.exp(pred_rewards), limit=1, logits=False)
                discounts = torch.cumprod(self.discount*(1 - pred_dones), 0)
                returns = torch.cumsum(rewards*discounts, 0)
                nonterminal = torch.cumprod(1 - pred_dones, 0)
                backup_values = [(pred_values[:, 0])] + \
                                [self.backup(i,
                                 returns[i-1],
                                 nonterminal[i-1],
                                 (pred_values[:, i]),) for i in range(1, depth+1)]
                pred_values = torch.stack(backup_values, 1)

        else:
            pred_values = torch.stack(pred_values, 1)
            pred_dones = None
            pred_rewards = None

        if override_actions is None:
            actions = torch.stack(actions, 1)
        all_pred_values = torch.stack(all_pred_values, 1).flatten(0, 1)
        return pred_latents, pred_values, all_pred_values, actions, \
               pred_dones, pred_rewards, lambdas

    def merge_values(self, original_qs, rollout_qs, actions):
        counts = torch.zeros_like(original_qs)[..., 0]

        counts = counts.scatter_add_(1, actions, torch.ones_like(actions).float())
        values = torch.zeros_like(original_qs)
        values = values.scatter_add_(1, actions.unsqueeze(-1).expand(-1, -1, values.shape[-1]), rollout_qs)

        mask = (counts > 0).float().unsqueeze(-1)
        values = original_qs*(1 - mask) + values*mask
        values = values/counts.clamp(1, None).unsqueeze(-1)

        return values

    def forward(self,
                observation,
                prev_action,
                prev_reward,
                train=False,
                eval=False,
                force_no_rollout=False,
                tau_override=None,
                seed_actions=None):
        """Returns the probability masses ``num_atoms x num_actions`` for the Q-values
        for each state/observation, using softmax output nonlinearity.

        For convenience reasons with DistributedDataParallel the forward method
        has been split into two cases, one for training and one for eval.
        """
        if eval:
            self.eval()
        else:
            self.train()
        while len(observation.shape) <= 5:
            observation.unsqueeze_(0)
        if tau_override is None:
            self.sample_taus(observation.shape[1], device=observation.device)
        else:
            self.taus = tau_override
        if train:
            # First, encode
            input_obs = observation[0].flatten(1, 2)
            input_obs = self.transform(input_obs, augment=True)
            latent = self.stem_forward(input_obs,
                                       prev_action[0],
                                       prev_reward[0])

            pred_latents, pred_ps, all_pred_ps, _, pred_dones, pred_reward, lambdas = \
                self.model_rollout(latent, self.jumps,
                                   override_actions=prev_action[1:],
                                   backup=self.rollout_rl_backup,
                                   adaptive_depth=False)

            extra_pred_done = self.done_predictor(pred_latents[-1],
                                                  prev_action[self.jumps+1]).unsqueeze(0)
            extra_pred_reward = self.reward_predictor(latent).unsqueeze(0)
            extra_pred_reward = F.log_softmax(extra_pred_reward, -1)
            pred_reward = extra_pred_reward if pred_reward is None else torch.cat([extra_pred_reward, pred_reward], 0)
            pred_dones = extra_pred_done if pred_dones is None else torch.cat([pred_dones, extra_pred_done], 0)

            if self.separate_spr_pass:
                pred_latents = [latent]
                if self.jumps > 0:
                    for j in range(1, self.jumps + 1):
                        latent = self.dynamics_model(latent, prev_action[j], blocks=False)
                        pred_latents.append(latent)

            diversity = self.calculate_diversity(pred_latents[0], prev_action[1], observation)
            pred_latents = torch.stack(pred_latents, 1)

            # Now calculate the SPR losses
            if self.use_spr:
                with torch.no_grad():
                    target_images = observation[:self.jumps+1].transpose(0, 1)
                    target_latents = self.encode_images(target_images, True, True)
                spr_loss = self.do_spr_loss(pred_latents.flatten(0, 1),
                                            target_latents.flatten(0, 1),
                                            prev_action[1:self.jumps+2].transpose(0, 1).flatten(),
                                            observation)
            else:
                spr_loss = torch.zeros((self.jumps + 1, observation.shape[1]), device=latent.device)

            calibration_loss = self.calibration_loss(pred_latents, prev_action[1:self.jumps+2], all_pred_ps)

            return pred_ps,\
                   pred_reward,\
                   pred_dones,\
                   spr_loss, \
                   diversity, \
                   lambdas, \
                   pred_latents ,\
                   calibration_loss

        else:
            aug = self.target_augmentation if not eval else self.eval_augmentation
            observation = observation[0].flatten(-4, -3)

            img = self.transform(observation, aug, target=not eval)

            # Infer (presence of) leading dimensions: [T,B], [B], or [].
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
            conv_out = self.renormalize(conv_out)

            p = self.head_forward(conv_out)

            depth = self.eval_depth if eval else self.target_depth
            runs = self.eval_runs if eval else self.target_runs
            if not force_no_rollout and depth > 0 and runs > 0:
                _, model_ps, _, actions, _, _, lambdas =\
                    self.model_rollout(conv_out,
                                       depth,
                                       override_actions=seed_actions,
                                       backup=True,
                                       adaptive_depth=True,
                                       runs=runs)
                if actions.shape[-1] > 1:
                    model_ps = average_targets(model_ps, lambdas)
                    model_ps = model_ps.view(p.shape[0], runs, *model_ps.shape[1:])
                    actions = actions.view(p.shape[0], runs, *actions.shape[1:])

                    p = self.merge_values(p, model_ps, actions[..., 0])

            if not self.distributional and eval:
                p = p.mean(-1)

            # Restore leading dimensions: [T,B], [B], or [], as input.
            p = restore_leading_dims(p, lead_dim, T, B)

            return p

    def select_action(self, obs):
        value = self.forward(obs, None, None, train=False, eval=True)

        if self.distributional:
            value = from_categorical(value, logits=False, limit=10)

        return value


class Projection(nn.Module):
    def __init__(self,
                 encoders,
                 noisy=None,
                 relu=False,
                 needs_action=False,
                 num_actions=18):
        """
        :param heads: Encoder heads, wrapped in nn.Sequential
        :param type:
        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.needs_action = needs_action
        self.relu = relu
        self.noisy = noisy
        self.num_actions = num_actions

    def forward(self, x, action):
        representations = []
        for encoder in self.encoders:
            if self.noisy is not None:
                [setattr(module, "noise_override", self.noisy) for module in encoder.modules()]
            if self.needs_action:
                batch_range = torch.arange(action.shape[0], device=action.device)
                action_onehot = torch.zeros(action.shape[0],
                                            self.num_actions,
                                            x.shape[-2],
                                            x.shape[-1],
                                            device=action.device)
                action_onehot[batch_range, action, :, :] = 1
                stacked_image = torch.cat([x, action_onehot], 1)
                representation = encoder(stacked_image)
            else:
                representation = encoder(x)
            if self.noisy is not None:
                [setattr(module, "noise_override", None) for module in encoder.modules()]
            if self.relu:
                representation = F.relu(representation)
            representations.append(representation)

        return torch.cat(representations, -1)


class LambdaPredictor(nn.Module):
    def __init__(self,
                 state_size=(64, 7, 7),
                 rl_loss_function=minimal_c51_loss,
                 batch_size=32,
                 lambda_eps=0.001,
                 use_state=True,
                 prediction_type="softmax",
                 needs_tau=False,
                 rollout_depth=5,
                 encoder=None,
                 ):
        super().__init__()
        pixels = state_size[-2]*state_size[-1]
        self.rollout_depth = rollout_depth
        self.prediction_type = prediction_type
        self.encoder = encoder
        output_size = rollout_depth + 1 if prediction_type == "softmax" else 1
        if use_state:
            self.network = nn.Sequential(nn.Dropout(0.5),
                                         nn.Conv2d(state_size[-3], 4, kernel_size=1),
                                         nn.Flatten(-3, -1),
                                         init_normalization(4*pixels, type="ln", one_d=True),
                                         nn.ReLU(),
                                         nn.Dropout(0.5),
                                         nn.Linear(4*pixels, output_size, bias=False))
        self.lambda_base = nn.Parameter(torch.zeros((1, output_size)), requires_grad=True)

        lr = 1e-3 if use_state else 1e-2
        weight_decay = 1e-3 if use_state else 1e-2
        self.optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.rl_loss_function = rl_loss_function
        self.needs_tau = needs_tau
        self.batch_size = batch_size
        self.eps = lambda_eps
        self.use_state = use_state
        self.iter = 0

    def predict(self, x):
        lambda_base = self.lambda_base.expand(x.shape[0], -1)
        if self.use_state:
            lambdas = self.network(x) + lambda_base
        else:
            lambdas = lambda_base

        if self.prediction_type == "softmax":
            return F.softmax(lambdas, -1)
        else:
            lambdas = torch.sigmoid(lambdas)*(1 - 2*self.eps) + self.eps
            weights = lambdas.expand(-1, self.rollout_depth+1,)
            weights = weights.cumprod(1)
            final_weights = lambdas.pow(self.rollout_depth+1) / (1 - lambdas)

            weights[:, -1] = final_weights[:, 0]
            weights = weights / torch.sum(weights, -1, keepdim=True)
            return weights

    @torch.no_grad()
    def forward(self, x):
        return self.predict(x)

    # @timeit
    def optimize(self, buffer):
        self.iter += 1
        samples = buffer.sample(self.batch_size)
        states, predictions, targets = samples[:3]

        if self.encoder is not None and self.use_state:
            states = self.encoder(states.unsqueeze(0))[0]
        lambdas = self.predict(states)
        predictions = average_targets(predictions, lambdas)

        if self.needs_tau:
            loss, _ = self.rl_loss_function(predictions, targets, samples[3])
        else:
            loss, _ = self.rl_loss_function(predictions, targets)

        if self.iter % 100 == 0:
            print(lambdas[0])

        self.optim.zero_grad()
        loss = loss.mean()
        loss.backward()
        self.optim.step()

        depth = torch.arange(lambdas.shape[-1], dtype=lambdas.dtype, device=lambdas.device)
        depth = (lambdas.detach()*(depth.unsqueeze(0))).sum(-1)

        return loss.item(), depth
