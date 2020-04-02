import torch
from rlpyt.agents.dqn.catdqn_agent import CatDqnAgent
from rlpyt.models.dqn.atari_catdqn_model import AtariCatDqnModel
from rlpyt.agents.dqn.atari.mixin import AtariMixin
from rlpyt.agents.dqn.atari.atari_catdqn_agent import AtariCatDQNAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
AgentInputs = namedarraytuple("AgentInputs",
    ["observation", "prev_action", "prev_reward"])
AgentInfo = namedarraytuple("AgentInfo", "q")
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])
from src.vectorized_mcts import VectorizedMCTS


class DQNSearchAgent(AtariCatDQNAgent):
    """Agent for Categorical DQN algorithm with search."""

    def __init__(self, args=None, eval=False, **kwargs):
        """Standard init, and set the number of probability atoms (bins)."""
        super().__init__(**kwargs)
        self.search_args = args
        self.eval = eval

    def initialize(self, env_spaces, share_memory=False,
                   global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory, global_B, env_ranks)
        # Overwrite distribution.
        self.search = VectorizedMCTS(self.search_args,
                                     env_spaces.action.n,
                                     self.model,
                                     eval=self.eval)

    def to_device(self, cuda_idx=None):
        """Moves the model to the specified cuda device, if not ``None``.  If
        sharing memory, instantiates a new model to preserve the shared (CPU)
        model.  Agents with additional model components (beyond
        ``self.model``) for action-selection or for use during training should
        extend this method to move those to the device, as well.

        Typically called in the runner during startup.
        """
        super().to_device(cuda_idx)
        self.search.to_device(cuda_idx)

    def eval_mode(self, itr):
        """Extend method to set epsilon for evaluation, using 1 for
        pre-training eval."""
        super().eval_mode(itr)
        logger.log(f"Agent at itr {itr}, eval eps "
            f"{self.eps_eval if itr > 0 else 1.}")
        self.search.set_eval(itr)


    def sample_mode(self, itr):
        """Extend method to set epsilon for sampling (including annealing)."""
        super().sample_mode(itr)
        self.search.set_train(itr)
        # itr_min = self._eps_itr_min_max[0]  # Shared memory for CpuSampler
        # itr_max = self._eps_itr_min_max[1]
        # if itr <= itr_max:
        #     prog = min(1, max(0, itr - itr_min) / (itr_max - itr_min))
        #     self.eps_sample = prog * self.eps_final + (1 - prog) * self.eps_init
        #     if itr % (itr_max // 10) == 0 or itr == itr_max:
        #         logger.log(f"Agent at itr {itr}, sample eps {self.eps_sample}"
        #             f" (min itr: {itr_min}, max_itr: {itr_max})")
        # self.distribution.set_epsilon(self.eps_sample)


    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Compute the discrete distribution for the Q-value for each
        action for each state/observation (no grad)."""
        # prev_action = self.distribution.to_onehot(prev_action)
        # model_inputs = buffer_to((observation, prev_action, prev_reward),
        #     device=self.device)
        action, p, value, initial_value = self.search.run(observation.to(self.device))
        p = p.cpu()
        action = action.cpu()
        agent_info = AgentInfo(p=p)  # Only change from DQN: q -> p.
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)


class VectorizedMCTS:
    def __init__(self, args, n_actions, network, eval=False):
        self.num_actions = n_actions
        self.network = network
        self.args = args
        self.pb_c_base = 19652
        self.pb_c_init = args.c1
        self.root_exploration_fraction = 0.25
        self.root_dirichlet_alpha = args.dirichlet_alpha
        self.visit_temp = args.visit_temp
        self.device = args.device
        self.n_runs = args.n_runs
        self.n_sims = args.n_sims
        self.id_null = self.n_sims + 1
        self.warmup_sims = min(self.n_sims // 3 + 1, n_actions)
        self.virtual_threads = args.virtual_threads
        self.vl_c = args.virtual_loss_c
        self.env_steps = 0
        self.cpu_search = args.cpu_search
        self.search_device = "cpu" if self.cpu_search else self.device
        self.eval = eval

    def initialize_on_device(self, device):
        # Initialize search tensors on the current device.
        # These are overwritten rather than reinitalized.
        # Store tensors to have [N_RUNS, N_SIMS] leading dimensions.
        self.q = torch.zeros((self.n_runs, self.n_sims + 2, self.num_actions),
                             device=device,
                             pin_memory=self.cpu_search)
        self.prior = torch.zeros((self.n_runs, self.n_sims + 2, self.num_actions), device=device)
        self.visit_count = torch.zeros((self.n_runs, self.n_sims + 2, self.num_actions), device=device)
        self.virtual_loss = torch.zeros((self.n_runs, self.n_sims + 2, self.num_actions), device=device)
        self.reward = torch.zeros((self.n_runs, self.n_sims + 2, self.num_actions), device=device)
        self.hidden_state = torch.zeros((self.n_runs, self.n_sims + 2, self.args.hidden_size, 6, 6), device=self.device)
        self.min_q, self.max_q = torch.zeros((self.n_runs,), device=device).fill_(MAXIMUM_FLOAT_VALUE), \
                                 torch.zeros((self.n_runs,), device=device).fill_(MINIMUM_FLOAT_VALUE)
        self.init_min_q, self.init_max_q = torch.zeros((self.n_runs,), device=device).fill_(MAXIMUM_FLOAT_VALUE), \
                                           torch.zeros((self.n_runs,), device=device).fill_(MINIMUM_FLOAT_VALUE)
        self.search_depths = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=device)
        self.dummy_ones = torch.ones_like(self.visit_count, device=device)
        self.dummy_zeros = torch.zeros_like(self.visit_count, device=device)

        # Initialize pointers defining the tree structure.
        self.id_children = torch.zeros((self.n_runs, self.n_sims + 2, self.num_actions),
                                       dtype=torch.int64, device=device)
        self.id_parent = torch.zeros((self.n_runs, self.n_sims + 2),
                                     dtype=torch.int64, device=device)

        # Pointers used during the search.
        self.id_current = torch.zeros((self.n_runs, 1), dtype=torch.int64, device=device,
                                      pin_memory=self.cpu_search)
        self.id_final = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=device,
                                    pin_memory=self.cpu_search)

        # Tensors defining the actions taken during the search.
        self.actions_final = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=device,
                                         pin_memory=self.cpu_search)
        self.search_actions = torch.zeros((self.n_runs, self.n_sims + 2),
                                          dtype=torch.int64, device=device,
                                          pin_memory=self.cpu_search)

        # A helper tensor used in indexing.
        self.batch_range = torch.arange(self.n_runs, device=device,
                                        pin_memory=self.cpu_search)

    def to_device(self, cuda_idx):
        self.device = torch.device("cuda", index=cuda_idx)
        self.initialize_on_device(self.device)

    def value_score(self, sim_id):
        """normalized_q(s,a)."""
        # if (sim_id - 1) % self.virtual_threads == 0:
        #     self.virtual_loss.fill_(0)
        # if sim_id <= 2:
        #     return -self.virtual_loss
        valid_indices = torch.where(self.visit_count > 0., self.dummy_ones, self.dummy_zeros)
        if sim_id <= self.warmup_sims:
            return -self.visit_count
        values = self.q - (valid_indices * self.min_q[:, None, None])
        values /= (self.max_q - self.min_q)[:, None, None]
        values = valid_indices * values
        return values

    def reset_tensors(self):
        """Reset all relevant tensors."""
        self.id_children.fill_(self.id_null)
        self.id_parent.fill_(self.id_null)
        self.visit_count.fill_(0)
        self.q.fill_(0)
        self.search_actions.fill_(0)
        self.min_q.fill_(MAXIMUM_FLOAT_VALUE)
        self.max_q.fill_(MINIMUM_FLOAT_VALUE)

    @torch.no_grad()
    def run(self, obs):
        self.reset_tensors()
        obs = obs.to(self.device).float() / 255.

        hidden_state, reward, policy_logits, initial_value = self.network.initial_inference(obs)
        self.hidden_state[:, 0, :] = hidden_state
        self.prior[:, 0] = F.softmax(policy_logits, dim=-1).to(self.search_device)
        self.add_exploration_noise()

        for sim_id in range(1, self.n_sims+1):
            # Pre-compute action to select at each node in case it is visited in this sim
            actions = self.ucb_select_child(sim_id)
            self.id_current.fill_(0)
            self.search_depths.fill_(0)

            # Because the tree has exactly sim_id nodes, we are guaranteed
            # to take at most sim_id transitions (including expansion).
            for depth in range(sim_id):
                # Select the tensor of children of the current node
                current_children = self.id_children.gather(1, self.id_current.unsqueeze(-1).expand(-1, -1, self.num_actions))

                # Select the children corresponding to the current actions
                current_actions = actions.gather(1, self.id_current.clamp_max(sim_id-1))
                id_next = current_children.squeeze().gather(-1, current_actions)
                self.search_actions[:, depth] = current_actions.squeeze()

                # Create a mask for live runs that will be true on the
                # exact step that a run terminates
                # A run terminates when its next state is unexpanded (null)
                # However, terminated runs also have this condition, so we
                # check that the current state is not yet null.
                done_mask = (id_next == self.id_null)
                live_mask = (self.id_current != self.id_null)
                final_mask = live_mask * done_mask

                # Note the final node id and action of terminated runs
                # to use in expansion.
                self.id_final[final_mask] = self.id_current[final_mask]
                self.actions_final[final_mask] = current_actions[final_mask]

                # If not done, increment search depths by one.
                self.search_depths[~done_mask] += 1

                self.id_current = id_next

                if torch.all(done_mask):
                    break

            input_state = self.hidden_state.gather(1, self.id_final[:, :, None, None, None].expand(-1, -1, 256, 6, 6).to(self.device)).squeeze()
            hidden_state, reward, policy_logits, value = self.network.inference(
                input_state, self.actions_final.to(self.device))

            # The new node is stored at entry sim_id
            self.hidden_state[:, sim_id, :] = hidden_state
            self.reward[self.batch_range, sim_id, self.actions_final.squeeze()] = reward.to(self.search_device)
            self.prior[:, sim_id] = F.softmax(policy_logits, dim=-1).to(self.search_device)

            # Store the pointers from parent to new node and back.
            self.id_children[self.batch_range, self.id_final.squeeze(), self.actions_final.squeeze()] = sim_id
            self.id_parent[:, sim_id] = self.id_final.squeeze()

            # The backup starts from the new node
            self.id_final.fill_(sim_id)
            self.backup(self.id_final, sim_id, value.to(self.search_device))

        # Get action, policy and value from the root after the search has finished
        action, policy = self.select_action()
        if self.args.no_search_value_targets:
            value = initial_value
        else:
            value = torch.sum(self.visit_count[:, 0] * self.q[:, 0], dim=-1)/torch.sum(self.visit_count[:, 0], dim=-1)

        if self.args.no_search_value_targets:
            value = initial_value
        return action, policy, value, initial_value

    def add_exploration_noise(self):
        concentrations = torch.tensor([self.root_dirichlet_alpha] * self.num_actions, device=self.search_device)
        noise = torch.distributions.dirichlet.Dirichlet(concentrations).sample((self.n_runs,))
        frac = self.root_exploration_fraction
        self.prior[:, 0, :] = (self.prior[:, 0, :] * (1-frac)) + (noise * frac)

    def backup(self, id_final, depth, value_final):
        returns = value_final
        id_current = id_final

        # Same number of steps as when we expanded
        for d in range(depth, 0, -1):

            # Determine the parent of the current node
            parent_id = self.id_parent.gather(1, id_current)
            actions = self.search_actions.gather(1, self.search_depths)

            # A backup has terminated if the parent id is null
            not_done_mask = (parent_id != self.id_null).float()

            # Get the rewards observed when transitioning to the current node
            reward = self.reward[self.batch_range, parent_id.squeeze(), actions.squeeze()]
            # Calculate the return as observed by the new parent.
            returns = returns*self.args.discount + reward

            # Update q and count at the parent for the actions taken then
            self.visit_count[self.batch_range, parent_id.squeeze(), actions.squeeze()] += not_done_mask.squeeze()
            # self.virtual_loss[self.batch_range, parent_id.squeeze(), actions.squeeze()] += (self.vl_c * not_done_mask.squeeze())
            values = ((self.q[self.batch_range, parent_id.squeeze(), actions.squeeze()] *
                       self.visit_count[self.batch_range, parent_id.squeeze(), actions.squeeze()]) + returns) \
                     / (self.visit_count[self.batch_range, parent_id.squeeze(), actions.squeeze()] + 1)
            values *= not_done_mask.squeeze()
            self.q[self.batch_range, parent_id.squeeze(), actions.squeeze()] = values
            values = values.squeeze()

            mins = torch.where(not_done_mask.squeeze() > 0, values, self.init_min_q)
            maxes = torch.where(not_done_mask.squeeze() > 0, values, self.init_max_q)
            self.min_q = torch.min(self.min_q, mins)
            self.max_q = torch.max(self.max_q, maxes)

            # Decrement the depth counter used for actions
            self.search_depths -= not_done_mask.long()
            # Ensure that it is nonnegative to not crash in gathering.
            self.search_depths.clamp_min_(0)

            id_current = parent_id

            if torch.all(parent_id == self.id_null):
                break

    def ucb_select_child(self, depth):
        # We have one extra visit of only the parent node that must be added
        # to the sum.  Otherwise, all values will be 0.
        total_visits = torch.sum(self.visit_count[:, :depth], -1, keepdim=True) + 1
        pb_c = self.pb_c_init * (torch.sqrt(total_visits) / (1 + self.visit_count[:, :depth])) * self.prior[:, :depth]
        value_score = self.value_score(depth)
        return torch.argmax(pb_c + value_score[:, :depth], dim=-1)

    def select_action(self):
        t = self.visit_softmax_temperature()
        policy = torch.distributions.Categorical(probs=self.visit_count[:, 0])
        action = torch.distributions.Categorical(probs=self.visit_count[:, 0]**(1/t)).sample()
        return action, policy.probs

    def visit_softmax_temperature(self):
        # TODO: Change the temperature schedule
        return self.visit_temp
        # if self.env_steps < 1e5:
        #     return 1.
        # if self.env_steps < 1e6:
        #     return 0.5
        # return 0.25

    def evaluate(self, env_step):
        env = gym.vector.make('atari-v0', num_envs=self.n_runs, asynchronous=False, args=self.args)
        for e in env.envs:
            e.eval()
        T_rewards = []
        dones, reward_sums, envs_done = [False] * self.n_runs, np.array([0.] * self.n_runs), 0

        obs = env.reset()
        obs = torch.from_numpy(obs)
        while envs_done < self.n_runs:
            actions, policy, value, _ = self.run(obs)
            next_obs, reward, done, _ = env.step(actions.cpu().numpy())
            reward_sums += np.array(reward)
            for i, d in enumerate(done):
                if done[i] and not dones[i]:
                    T_rewards.append(reward_sums[i])
                    dones[i] = True
                    envs_done += 1
            obs.copy_(torch.from_numpy(next_obs))
        env.close()

        avg_reward = sum(T_rewards) / len(T_rewards)
        return avg_reward


class VectorizedQMCTS(VectorizedMCTS):
    # def __init__(self, *args, **kwargs):
    #     super(VectorizedQMCTS, self).__init__(*args, **kwargs)
    #     self.root_exploration_fraction = 0.05

    def reset_tensors(self):
        super().reset_tensors()
        self.visit_count.fill_(1)

    @torch.no_grad()
    def run(self, obs):
        self.reset_tensors()
        obs = obs.to(self.device).float() / 255.

        hidden_state, reward, policy_logits, initial_value = self.network.initial_inference(obs)
        self.hidden_state[:, 0, :] = hidden_state
        self.q[:, 0] = initial_value.to(self.search_device)
        self.min_q = torch.min(self.q[:, 0], dim=-1)[0]
        self.max_q = torch.max(self.q[:, 0], dim=-1)[0]
        if self.args.q_dirichlet:
            self.add_exploration_noise()

        for sim_id in range(1, self.n_sims+1):
            # Pre-compute action to select at each node in case it is visited in this sim
            actions = self.ucb_select_child(sim_id)
            self.id_current.fill_(0)
            self.search_depths.fill_(0)

            # Because the tree has exactly sim_id nodes, we are guaranteed
            # to take at most sim_id transitions (including expansion).
            for depth in range(sim_id):
                # Select the tensor of children of the current node
                current_children = self.id_children.gather(1, self.id_current.unsqueeze(-1).expand(-1, -1, self.num_actions))

                # Select the children corresponding to the current actions
                current_actions = actions.gather(1, self.id_current.clamp_max(sim_id-1))
                id_next = current_children.squeeze().gather(-1, current_actions)
                self.search_actions[:, depth] = current_actions.squeeze()

                # Create a mask for live runs that will be true on the
                # exact step that a run terminates
                # A run terminates when its next state is unexpanded (null)
                # However, terminated runs also have this condition, so we
                # check that the current state is not yet null.
                done_mask = (id_next == self.id_null)
                live_mask = (self.id_current != self.id_null)
                final_mask = live_mask * done_mask

                # Note the final node id and action of terminated runs
                # to use in expansion.
                self.id_final[final_mask] = self.id_current[final_mask]
                self.actions_final[final_mask] = current_actions[final_mask]

                # If not done, increment search depths by one.
                self.search_depths[~done_mask] += 1

                self.id_current = id_next

                if torch.all(done_mask):
                    break

            input_state = self.hidden_state.gather(1, self.id_final[:, :, None, None, None].expand(-1, -1, 256, 6, 6).to(self.device)).squeeze()
            hidden_state, reward, policy_logits, value = self.network.inference(
                input_state, self.actions_final.to(self.device))
            value = value.to(self.search_device)

            # The new node is stored at entry sim_id
            self.hidden_state[:, sim_id, :] = hidden_state
            self.reward[self.batch_range, sim_id, self.actions_final.squeeze()] = reward.to(self.search_device)
            # self.prior[:, sim_id] = F.softmax(policy_logits, dim=-1)
            self.q[:, sim_id] = value

            # Store the pointers from parent to new node and back.
            self.id_children[self.batch_range, self.id_final.squeeze(), self.actions_final.squeeze()] = sim_id
            self.id_parent[:, sim_id] = self.id_final.squeeze()

            # The backup starts from the new node
            self.id_final.fill_(sim_id)
            self.backup(self.id_final, sim_id, value)

        # Get action, policy and value from the root after the search has finished
        action = self.select_action()
        if self.args.no_search_value_targets:
            value = initial_value.max(dim=-1)[0]
        else:
            value = self.q[:, 0].max(dim=-1)[0]

        return action, F.softmax(self.q[:, 0], dim=-1), value, initial_value.max(dim=-1)[0]

    def value_score(self, sim_id):
        """normalized_q(s,a)."""
        valid_indices = torch.where(self.visit_count > 0., self.dummy_ones, self.dummy_zeros)
        # if sim_id <= self.warmup_sims:
        #     return -self.visit_count
        values = self.q - (valid_indices * self.min_q[:, None, None])
        values /= (self.max_q - self.min_q)[:, None, None]
        values = valid_indices * values
        return values

    def add_exploration_noise(self):
        concentrations = torch.tensor([self.root_dirichlet_alpha] * self.num_actions, device=self.search_device)
        noise = torch.distributions.dirichlet.Dirichlet(concentrations).sample((self.n_runs,))
        frac = self.root_exploration_fraction

        q_dist = F.softmax(self.q[:, 0], -1)
        mixed_dist = (q_dist * (1-frac)) + (noise * frac)
        est_q = torch.log(mixed_dist)
        mean_offset = self.q[:, 0].mean(-1, keepdim=True) - est_q.mean(-1, keepdim=True)

        self.q[:, 0] = est_q + mean_offset

    def backup(self, id_final, depth, value_final):
        returns = value_final.max(dim=-1)[0]
        id_current = id_final

        # Same number of steps as when we expanded
        for d in range(depth, 0, -1):

            # Determine the parent of the current node
            parent_id = self.id_parent.gather(1, id_current)
            actions = self.search_actions.gather(1, self.search_depths)

            # A backup has terminated if the parent id is null
            not_done_mask = (parent_id != self.id_null).float()

            # Get the rewards observed when transitioning to the current node
            reward = self.reward[self.batch_range, parent_id.squeeze(), actions.squeeze()]
            # Calculate the return as observed by the new parent.
            returns = returns*self.args.discount + reward

            # Update q and count at the parent for the actions taken then
            self.visit_count[self.batch_range, parent_id.squeeze(), actions.squeeze()] += not_done_mask.squeeze()
            values = ((self.q[self.batch_range, parent_id.squeeze(), actions.squeeze()] *
                       self.visit_count[self.batch_range, parent_id.squeeze(), actions.squeeze()]) + returns) \
                     / (self.visit_count[self.batch_range, parent_id.squeeze(), actions.squeeze()] + 1)
            values *= not_done_mask.squeeze()
            self.q[self.batch_range, parent_id.squeeze(), actions.squeeze()] = values
            values = values.squeeze()

            mins = torch.where(not_done_mask.squeeze() > 0, values, self.init_min_q)
            maxes = torch.where(not_done_mask.squeeze() > 0, values, self.init_max_q)
            self.min_q = torch.min(self.min_q, mins)
            self.max_q = torch.max(self.max_q, maxes)

            # Decrement the depth counter used for actions
            self.search_depths -= not_done_mask.long()
            # Ensure that it is nonnegative to not crash in gathering.
            self.search_depths.clamp_min_(0)

            id_current = parent_id

            if torch.all(parent_id == self.id_null):
                break

    def ucb_select_child(self, depth):
        # We have one extra visit of only the parent node that must be added
        # to the sum.  Otherwise, all values will be 0.
        # total_visits = torch.sum(self.visit_count[:, :depth], -1, keepdim=True) + 1
        # pb_c = self.pb_c_init * (torch.sqrt(total_visits) / (1 + self.visit_count[:, :depth])) * self.prior[:, :depth]
        total_visits = torch.sum(self.visit_count[:, :depth], -1, keepdim=True)
        pb_c = self.pb_c_init * torch.sqrt((torch.log(total_visits) / (self.visit_count[:, :depth])))
        value_score = self.value_score(depth)
        return torch.argmax(pb_c + value_score[:, :depth], dim=-1)

    def select_action(self):
        epsilon = self.args.epsilon
        if self.eval:
            epsilon *= 0.1
        e_action = (torch.rand_like(self.q[:, 0, 0], device=self.search_device) < epsilon).long()
        random_actions = torch.randint(self.num_actions, size=(self.n_runs,), device=self.search_device)
        max_actions = self.q[:, 0].argmax(dim=-1)
        actions = e_action * random_actions + (1-e_action) * max_actions
        return actions
