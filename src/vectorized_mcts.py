import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions

MAXIMUM_FLOAT_VALUE = torch.finfo().max
MINIMUM_FLOAT_VALUE = torch.finfo().min


class VectorizedMCTS:
    def __init__(self, args, n_actions, n_runs, network):
        self.num_actions = n_actions
        self.n_runs = n_runs
        self.network = network
        self.args = args
        self.args.pb_c_base = 19652
        self.args.pb_c_init = 1.25
        self.args.root_exploration_fraction = 0.25
        self.args.root_dirichlet_alpha = 0.25

        self.device = args.device
        self.n_runs, self.n_sims = n_runs, args.num_simulations
        self.id_null = self.n_sims

        # Initialize search tensors on the current device.
        # These are overwritten rather than reinitalized.
        self.q = torch.zeros((n_runs, self.n_sims + 1, self.num_actions), device=self.device)
        self.prior = torch.zeros((n_runs, self.n_sims + 1, self.num_actions), device=self.device)
        self.visit_count = torch.zeros((n_runs, self.n_sims + 1, self.num_actions), device=self.device)
        self.reward = torch.zeros((n_runs, self.n_sims + 1), device=self.device)
        self.hidden_state = torch.zeros((n_runs, self.n_sims + 1, args.hidden_size, 6, 6), device=self.device)
        self.min_q, self.max_q = torch.zeros((n_runs,), device=self.device).fill_(MAXIMUM_FLOAT_VALUE), \
                                 torch.zeros((n_runs,), device=self.device).fill_(MINIMUM_FLOAT_VALUE)
        self.search_depths = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=self.device)

        # Initialize pointers defining the tree structure.
        self.id_children = torch.zeros((n_runs, self.n_sims + 1, self.num_actions),
                                       dtype=torch.int64, device=self.device)
        self.id_parent = torch.zeros((n_runs, self.n_sims + 1),
                                     dtype=torch.int64, device=self.device)

        # Pointers used during the search.
        self.id_current = torch.zeros((self.n_runs, 1), dtype=torch.int64, device=self.device)
        self.id_final = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=self.device)

        # Tensors defining the actions taken during the search.
        self.actions_final = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=self.device)
        self.search_actions = torch.zeros((n_runs, self.n_sims + 1),
                                          dtype=torch.int64, device=self.device)

        # A helper tensor used in indexing.
        self.batch_range = torch.arange(self.n_runs, device=self.device)

    def normalize(self):
        """Normalize Q-values to be b/w 0 and 1 for each search tree."""
        if (self.max_q <= self.min_q)[0]:
            return self.q
        return (self.q - self.min_q[:, None, None]) / (self.max_q - self.min_q)[:, None, None]

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
        obs = obs.to(self.device)

        hidden_state, reward, policy_logits, initial_value = self.network.initial_inference(obs)
        self.hidden_state[:, 0, :] = hidden_state
        self.prior[:, 0] = F.softmax(policy_logits, dim=-1)
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

            input_state = self.hidden_state.gather(1, self.id_final[:, :, None, None, None].expand(-1, -1, 256, 6, 6)).squeeze()
            hidden_state, reward, policy_logits, value = self.network.inference(
               input_state, self.actions_final)

            # The new node is stored at entry sim_id
            self.hidden_state[:, sim_id, :] = hidden_state
            self.reward[:, sim_id] = reward
            self.prior[:, sim_id] = F.softmax(policy_logits, dim=-1)

            # Store the pointers from parent to new node and back.
            self.id_children[self.batch_range, self.id_final.squeeze(), self.actions_final.squeeze()] = sim_id
            self.id_parent[:, sim_id] = self.id_final.squeeze()

            # The backup starts from the new node
            self.id_final.fill_(sim_id)
            self.backup(self.id_final, sim_id, value)

        # Get action, policy and value from the root after the search has finished
        action, policy = self.select_action()
        if self.args.no_search_value_targets:
            value = initial_value
        else:
            value = torch.sum(self.visit_count[:, 0] * self.q[:, 0], dim=-1)/torch.sum(self.visit_count[:, 0], dim=-1)

        if self.args.no_search_value_targets:
            value = initial_value
        return action, policy, value

    def add_exploration_noise(self):
        concentrations = torch.tensor([self.args.root_dirichlet_alpha] * self.num_actions, device=self.device)
        noise = torch.distributions.dirichlet.Dirichlet(concentrations).sample((self.n_runs,))
        frac = self.args.root_exploration_fraction
        self.prior[:, 0, :] = self.prior[:, 0, :] * (1-frac) + noise * frac

    def backup(self, id_final, depth, value_final):
        returns = value_final
        id_current = id_final

        # Same number of steps as when we expanded
        for d in range(depth, 0, -1):

            # Determine the parent of the current node
            parent_id = self.id_parent.gather(1, id_current)

            # A backup has terminated if the parent id is null
            not_done_mask = (parent_id != self.id_null).float()

            # Get the rewards observed when transitioning to the current node
            reward = self.reward.gather(1, id_current).squeeze()
            # Calculate the return as observed by the new parent.
            returns = returns*self.args.discount + reward
            actions = self.search_actions.gather(1, self.search_depths)

            # Update q and count at the parent for the actions taken then
            self.visit_count[self.batch_range, parent_id.squeeze(), actions.squeeze()] += not_done_mask.squeeze()
            self.q[self.batch_range, parent_id.squeeze(), actions.squeeze()] += not_done_mask.squeeze() * \
                                              ((returns - self.q[self.batch_range, parent_id.squeeze(), actions.squeeze()])
                                              / (self.visit_count[self.batch_range, parent_id.squeeze(), actions.squeeze()] + 1))

            # Decrement the depth counter used for actions
            self.search_depths -= not_done_mask.long()
            # Ensure that it is nonnegative to not crash in gathering.
            self.search_depths.clamp_min_(0)

            id_current = parent_id

        # Calculate new max and min q values per-tree for normalization
        self.min_q = self.q[:, :depth+1].min(1)[0].min(1)[0]
        self.max_q = self.q[:, :depth+1].max(1)[0].max(1)[0]

    def ucb_select_child(self, depth):
        # We have one extra visit of only the parent node that must be added
        # to the sum.  Otherwise, all values will be 0.
        total_visits = torch.sum(self.visit_count[:, :depth], -1, keepdim=True) + 1
        pb_c = torch.log((total_visits + self.args.pb_c_base + 1) / self.args.pb_c_base) + self.args.pb_c_init
        pb_c = pb_c * (torch.sqrt(total_visits) / (1 + self.visit_count[:, :depth])) * self.prior[:, :depth]
        normalized_q = self.normalize()
        return torch.argmax(pb_c + normalized_q[:, :depth], dim=-1)

    def select_action(self):
        t = self.visit_softmax_temperature()
        policy = torch.distributions.Categorical(probs=self.visit_count[:, 0]**(1/t))
        action = policy.sample()
        return action, policy

    def visit_softmax_temperature(self, training_steps=0):
        # TODO: Change the temperature schedule
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25
