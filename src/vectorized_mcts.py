import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions

MAXIMUM_FLOAT_VALUE = float('inf')


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
        self.id_null = self.n_sims + 1
        self.id_children = torch.zeros((n_runs, self.n_sims + 1, self.num_actions), dtype=torch.int64).fill_(self.id_null)
        self.id_parent = torch.zeros((n_runs, self.n_sims + 1), dtype=torch.int64).fill_(self.id_null)
        self.search_actions = torch.zeros((n_runs, self.n_sims + 1), dtype=torch.int64)
        self.q = torch.zeros((n_runs, self.n_sims + 1, self.num_actions))
        self.prior = torch.zeros((n_runs, self.n_sims + 1, self.num_actions))
        self.visit_count = torch.zeros((n_runs, self.n_sims + 1, self.num_actions))
        self.reward = torch.zeros((n_runs, self.n_sims + 1))
        self.hidden_state = torch.zeros((n_runs, self.n_sims + 1, args.hidden_size, 6, 6))
        # self.ucb_scores = torch.zeros((n_runs, self.n_sims + 1, self.num_actions))
        self.min_q, self.max_q = torch.zeros((n_runs,)).fill_(MAXIMUM_FLOAT_VALUE), \
                                 torch.zeros((n_runs,)).fill_(-MAXIMUM_FLOAT_VALUE)
        # self.value_sum = torch.zeros(n_runs, self.n_sims + 1) # + 1 for null node

    def normalize(self):
        if (self.max_q < self.min_q)[0]:
            return self.q
        return (self.q - self.min_q.unsqueeze(dim=-1)) / (self.max_q - self.min_q).unsqueeze(dim=-1)

    @torch.no_grad()
    def run(self, obs):
        self.id_children.fill_(self.id_null)
        self.id_parent.fill_(self.id_null)
        self.visit_count.fill_(0)
        self.q.fill_(0)
        self.search_actions.fill_(0)

        obs = obs.to(self.device)
        id_current = torch.zeros((self.n_runs, 1), dtype=torch.int64)
        self.add_exploration_noise()
        hidden_state, reward, policy_logits, value = self.network.initial_inference(obs)
        self.hidden_state[:, 0, :] = hidden_state
        self.prior[:, 0] = F.softmax(policy_logits, dim=-1)

        id_final = torch.zeros(self.n_runs, 1, dtype=torch.int64)
        actions_final = torch.zeros(self.n_runs, 1, dtype=torch.int64)

        for sim_id in range(1, self.n_sims+1):
            actions = self.ucb_select_child(sim_id)
            for depth in range(sim_id):
                id_next = self.id_children.gather(1, id_current.unsqueeze(-1).expand(-1, -1, self.num_actions))
                current_actions = actions.gather(1, id_current)
                id_next = id_next.squeeze().gather(-1, current_actions)
                self.search_actions[:, depth] = current_actions.squeeze()

                done_mask = (id_next == self.id_null)
                live_mask = (id_current != id_next)
                final_mask = live_mask * done_mask

                id_final[final_mask] = id_current[final_mask]
                actions_final[final_mask] = current_actions[final_mask]

                id_current = id_next

            hidden_state, reward, policy_logits, value = self.network.inference(
                self.hidden_state.gather(1, id_final[:, :, None, None, None].expand(-1, -1, 256, 6, 6)).squeeze(),
                actions_final)
            self.hidden_state[:, sim_id, :] = hidden_state
            self.reward[:, sim_id] = reward
            self.prior[:, sim_id] = F.softmax(policy_logits, dim=-1)

            self.id_children[id_final, actions_final] = sim_id
            self.id_parent[:, sim_id] = id_final.squeeze()

            self.backup(id_final, sim_id, value)

        action, policy = self.select_action()
        value = (self.visit_count[:, 0] * self.q[:, 0])/(self.visit_count[:, 0])

        return action, policy, value

    def add_exploration_noise(self):
        concentrations = torch.tensor([self.args.root_dirichlet_alpha] * self.num_actions)
        noise = torch.distributions.dirichlet.Dirichlet(concentrations).sample((self.n_runs,))
        frac = self.args.root_exploration_fraction
        self.prior[:, 0, :] = self.prior[:, 0, :] * (1-frac) + noise * frac

    def backup(self, id_final, depth, value_final):
        returns = value_final
        id_current = id_final
        for d in range(depth, -1, -1):
            reward = self.reward.gather(1, id_current)
            returns = returns*self.args.discount + reward
            actions = self.search_actions[:, d]
            new_id_current = self.id_parent.gather(1, id_current)
            done_mask = (id_current != new_id_current).float()
            id_current = new_id_current
            self.visit_count[:, id_current, actions] += done_mask
            self.q[:, id_current, actions] += done_mask * \
                                              ((returns - self.q[:, id_current, actions])
                                              /self.visit_count[:, id_current, actions])

        self.min_q = self.q[:, :depth+1].min(1)[0].min(1)[0]
        self.max_q = self.q[:, :depth+1].max(1)[0].max(1)[0]

    def ucb_select_child(self, depth):
        total_visits = torch.sum(self.visit_count[:, :depth+1], -1, keepdim=True)
        pb_c = torch.log((total_visits + self.args.pb_c_base + 1) / self.args.pb_c_base) + self.args.pb_c_init
        pb_c = pb_c * (torch.sqrt(total_visits) / (1 + self.visit_count[:, :depth+1])) * self.prior[:, :depth]
        normalized_q = self.normalize()
        return torch.argmax(pb_c + normalized_q[:, :depth+1], dim=-1)

    def select_action(self):
        t = self.visit_softmax_temperature()
        policy = torch.distributions.Categorical(logits=self.visit_count[:, 0]/t)
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
