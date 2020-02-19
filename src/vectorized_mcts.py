import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions
import gym
import torch.multiprocessing as mp
import time
import traceback
import sys
import wandb

MAXIMUM_FLOAT_VALUE = torch.finfo().max / 10
MINIMUM_FLOAT_VALUE = torch.finfo().min / 10


class VectorizedMCTS:
    def __init__(self, args, n_actions, n_runs, network, eval=False):
        self.num_actions = n_actions
        self.n_runs = n_runs
        self.network = network
        self.args = args
        self.pb_c_base = 19652
        self.pb_c_init = args.c1
        self.root_exploration_fraction = 0.25
        self.root_dirichlet_alpha = 0.25
        self.visit_temp = args.visit_temp
        self.device = args.device
        self.n_runs, self.n_sims = n_runs, args.num_simulations
        if eval:
            self.n_sims = 50
        self.id_null = self.n_sims + 1
        self.warmup_sims = min(self.n_sims // 3 + 1, n_actions)

        # Initialize search tensors on the current device.
        # These are overwritten rather than reinitalized.
        # Store tensors to have [N_RUNS, N_SIMS] leading dimensions.
        self.q = torch.zeros((n_runs, self.n_sims + 2, self.num_actions), device=self.device)
        self.prior = torch.zeros((n_runs, self.n_sims + 2, self.num_actions), device=self.device)
        self.visit_count = torch.zeros((n_runs, self.n_sims + 2, self.num_actions), device=self.device)
        self.reward = torch.zeros((n_runs, self.n_sims + 2), device=self.device)
        self.hidden_state = torch.zeros((n_runs, self.n_sims + 2, args.hidden_size, 6, 6), device=self.device)
        self.min_q, self.max_q = torch.zeros((n_runs,), device=self.device).fill_(MAXIMUM_FLOAT_VALUE), \
                                 torch.zeros((n_runs,), device=self.device).fill_(MINIMUM_FLOAT_VALUE)
        self.init_min_q, self.init_max_q = torch.zeros((n_runs,), device=self.device).fill_(MAXIMUM_FLOAT_VALUE), \
                                           torch.zeros((n_runs,), device=self.device).fill_(MINIMUM_FLOAT_VALUE)
        self.search_depths = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=self.device)
        self.dummy_ones = torch.ones_like(self.visit_count, device=self.device)
        self.dummy_zeros = torch.zeros_like(self.visit_count, device=self.device)

        # Initialize pointers defining the tree structure.
        self.id_children = torch.zeros((n_runs, self.n_sims + 2, self.num_actions),
                                       dtype=torch.int64, device=self.device)
        self.id_parent = torch.zeros((n_runs, self.n_sims + 2),
                                     dtype=torch.int64, device=self.device)

        # Pointers used during the search.
        self.id_current = torch.zeros((self.n_runs, 1), dtype=torch.int64, device=self.device)
        self.id_final = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=self.device)

        # Tensors defining the actions taken during the search.
        self.actions_final = torch.zeros(self.n_runs, 1, dtype=torch.int64, device=self.device)
        self.search_actions = torch.zeros((n_runs, self.n_sims + 2),
                                          dtype=torch.int64, device=self.device)

        # A helper tensor used in indexing.
        self.batch_range = torch.arange(self.n_runs, device=self.device)

    def normalize(self, sim_id):
        """Normalize Q-values to be b/w 0 and 1 for each search tree."""
        valid_indices = torch.where(self.visit_count > 0., self.dummy_ones, self.dummy_zeros)
        if sim_id <= self.warmup_sims:
            return -self.visit_count
        values = self.q - valid_indices * self.min_q[:, None, None]
        values /= (self.max_q - self.min_q)[:, None, None]
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
        concentrations = torch.tensor([self.root_dirichlet_alpha] * self.num_actions, device=self.device)
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

            # A backup has terminated if the parent id is null
            not_done_mask = (parent_id != self.id_null).float()

            # Get the rewards observed when transitioning to the current node
            reward = self.reward.gather(1, id_current).squeeze()
            # Calculate the return as observed by the new parent.
            returns = returns*self.args.discount + reward
            actions = self.search_actions.gather(1, self.search_depths)

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

    def ucb_select_child(self, depth):
        # We have one extra visit of only the parent node that must be added
        # to the sum.  Otherwise, all values will be 0.
        total_visits = torch.sum(self.visit_count[:, :depth], -1, keepdim=True) + 1
        pb_c = self.pb_c_init * (torch.sqrt(total_visits) / (1 + self.visit_count[:, :depth])) * self.prior[:, :depth]
        normalized_q = self.normalize(depth)
        return torch.argmax(pb_c + normalized_q[:, :depth], dim=-1)

    def select_action(self):
        t = self.visit_temp
        policy = torch.distributions.Categorical(probs=self.visit_count[:, 0]**(1/t))
        action = policy.sample()
        return action, policy

    def visit_softmax_temperature(self, training_steps=0):
        # TODO: Change the temperature schedule
        return self.args.visit_temp

    def evaluate(self, env_step=None):
        env = gym.vector.make('atari-v0', num_envs=self.n_runs, asynchronous=False, args=self.args)
        for e in env.envs:
            e.eval()
        T_rewards = []
        dones, reward_sums, envs_done = [False] * self.n_runs, np.array([0.] * self.n_runs), 0

        obs = env.reset()
        obs = torch.from_numpy(obs)
        while envs_done < self.n_runs:
            actions, policy, value = self.run(obs)
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


class AsyncEval:
    def __init__(self, eval_mcts):
        ctx = mp.get_context('spawn')
        self.error_queue = ctx.Queue()
        self.send_queue = ctx.Queue()
        self.receive_queue = ctx.Queue()
        process = ctx.Process(target=eval_wrapper,
                              name='EvalWorker',
                              args=((
                                  eval_mcts,
                                  'EvalWorker',
                                  self.send_queue,
                                  self.receive_queue,
                                  self.error_queue,
                                  )))
        process.start()

    def get_eval_results(self):
        try:
            result, success = self.receive_queue.get()
            return result
        except:
            return None


def eval_wrapper(eval_mcts, name, send_queue, recieve_queue, error_queue):
    try:
        while True:
            command, env_step = send_queue.get()
            if command == 'evaluate':
                avg_reward = eval_mcts.evaluate(env_step)
                recieve_queue.put(((env_step, avg_reward), True))
            else:
                time.sleep(10.)
    except (KeyboardInterrupt, Exception):
        error_queue.put((name,) + sys.exc_info())
        traceback.print_exc()
        recieve_queue.put((None, False))
    finally:
        return

