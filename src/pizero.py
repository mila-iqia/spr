import collections
import math
import numpy as np
from typing import Dict, List, Optional
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
import numpy as np
from itertools import islice
import multiprocessing
import time

import gym
from src.mcts_memory import Transition, blank_batch_trans
from src.model_trainer import MCTSModel
import time
import wandb
from recordclass import dataobject

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class ListEpisode(dataobject):
    obs: list
    actions: list
    rewards: list
    dones: list


class NPEpisode(dataobject):
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds=None):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ReanalyzeWorker(multiprocessing.Process):
    def __init__(self, args, input_queue, output_queue, obs_shape):
        super().__init__()
        self.reanalyze_heads = 4*args.num_envs
        self.current_write_episodes = [ListEpisode([], [], [], []) for _ in range(self.reanalyze_heads)]
        self.current_read_episodes = []
        self.tr_indices = np.zeros((self.reanalyze_heads,), dtype="int")
        self.inqueue = input_queue
        self.outqueue = output_queue
        self.buffer = []
        self.current_indices = []
        self.directory = args.savedir
        self.total_episodes = 0
        self.obs_shape = obs_shape
        self.args = args

    def run(self, forever=True):
        while True:
            self.write_to_buffer()

            # Don't reanalyze until we have data for it.
            if self.total_episodes > 0 and self.outqueue.qsize() < 100:
                new_samples = self.sample_for_reanalysis()
                self.outqueue.put(new_samples)
            else:
                time.sleep(0.1) # Don't just spin the processor
            if not forever:
                return

    def write_to_buffer(self):
        while not self.inqueue.empty():
            obs, action, reward, done = self.inqueue.get()

            for i in range(self.args.num_envs):
                episode = self.current_write_episodes[i]
                episode.obs.append(obs[i])
                episode.actions.append(action[i])
                episode.rewards.append(reward[i])
                episode.dones.append(done[i])

                if done[i]:
                    self.save_episode(episode)

            self.current_write_episodes[i] = ListEpisode([], [], [], [])

    def save_episode(self, episode):

        obs = np.stack(episode.obs)
        actions = np.array(episode.actions)
        rewards = np.array(episode.rewards)
        dones = np.array(episode.dones)

        filename = self.directory + "/ep_{}.npz".format(self.total_episodes)
        np.savez_compressed(file=filename, obs=obs, actions=actions,
                            dones=dones, rewards=rewards)
        self.total_episodes += 1

    def load_episode(self, index):
        filename = self.directory + "/ep_{}.npz".format(index)
        file = np.load(filename)
        return NPEpisode(file["obs"], file["actions"],
                         file["rewards"], file["dones"])

    def sample_for_reanalysis(self):
        """
        :param buffer: list of lists of Transitions.  Each sublist is an episode.
        :return: list of new transitions, representing a reanalyzed episode.
        """
        # We only want to reanalyze done episodes for now.

        observations = np.zeros((self.reanalyze_heads,
                                    self.args.framestack,
                                    *self.obs_shape),
                                   dtype=np.uint8)
        actions = []
        rewards = np.zeros(self.reanalyze_heads)
        dones = np.zeros(self.reanalyze_heads)

        for i in range(self.reanalyze_heads):
            # Should only happen at initialization.
            if i >= len(self.current_read_episodes):
                new_ep = self.load_episode(np.random.randint(self.total_episodes))
                self.current_read_episodes.append(new_ep)
                self.tr_indices[i] = 0

            episode = self.current_read_episodes[i]
            ind = self.tr_indices[i]

            bottom_ind = max(0, ind - self.args.framestack + 1)
            start_point = self.args.framestack - (ind - bottom_ind + 1)
            observations[i, start_point:] = episode.obs[bottom_ind:ind+1]
            actions.append(episode.actions[ind])
            rewards[i] = episode.rewards[ind]
            dones[i] = episode.dones[ind]
            self.tr_indices[i] += 1

            # Check if we've finished the episode.
            if self.tr_indices[i] >= episode.obs.shape[0]:
                self.tr_indices[i] = 0
                # Load in a random new episode
                self.current_read_episodes[i] = self.load_episode(np.random.randint(self.total_episodes))

        return torch.from_numpy(observations).float() / 255.,\
               actions, rewards, dones


class PiZero:
    def __init__(self, args):
        self.args = args
        self.args.pb_c_base = 19652
        self.args.pb_c_init = 1.25
        self.args.root_exploration_fraction = 0.25
        self.args.root_dirichlet_alpha = 0.25
        self.env = gym.vector.make('atari-v0', num_envs=args.num_envs, args=args,
                                   asynchronous=not args.sync_envs)
        self.network = MCTSModel(args, self.env.action_space[0].n)
        self.network.to(self.args.device)
        self.mcts = MCTS(args, self.env, self.network)

    def evaluate(self):
        num_envs = self.args.evaluation_episodes
        env = gym.vector.make('atari-v0', num_envs=num_envs, asynchronous=False, args=self.args)
        env.seed([self.args.seed] * num_envs)
        for e in env.envs:
            e.eval()
        T_rewards, T_Qs = [], []
        dones, reward_sums, envs_done = [False] * num_envs, np.array([0.] * num_envs), 0

        obs = torch.from_numpy(env.reset())
        while envs_done < num_envs:
            roots = self.mcts.batched_run(obs)
            actions = []
            for root in roots:
                # Select action for each obs
                action, p_logit = self.mcts.select_action(root)
                actions.append(action)
            next_obs, reward, done, _ = env.step(actions)
            reward_sums += np.array(reward)
            for i, d in enumerate(done):
                if done[i] and not dones[i]:
                    T_rewards.append(reward_sums[i])
                    dones[i] = True
                    envs_done += 1
        env.close()

        avg_reward = sum(T_rewards) / len(T_rewards)
        return avg_reward


class MCTS:
    def __init__(self, args, env, network):
        self.args = args
        self.env = env
        self.network = network
        self.min_max_stats = MinMaxStats()

    def run(self, obs):
        root = Node(0)
        obs = obs.to(self.args.device)
        root.hidden_state = obs
        self.expand_node(root, network_output=self.network.initial_inference(obs))
        for s in range(self.args.num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            with torch.no_grad():
                action = torch.tensor(action, device=self.args.device)
                network_output = self.network.inference(parent.hidden_state, action)
            self.expand_node(node, network_output)
            self.backpropagate(search_path, network_output.value)
        return root

    def batched_run(self, obs_tensor):
        roots = []
        obs_tensor = obs_tensor.to(self.args.device)
        network_output = self.network.initial_inference(obs_tensor)

        for i in range(obs_tensor.shape[0]):
            root = Node(0)
            root.hidden_state = obs_tensor[i]
            self.expand_node(root, network_output[i])
            self.add_exploration_noise(root)
            roots.append(root)

        for s in range(self.args.num_simulations):
            nodes, search_paths, actions, hidden_states = [], [], [], []
            for i in range(obs_tensor.shape[0]):
                node = roots[i]
                search_path = [node]

                while node.expanded():
                    action, node = self.optimized_select_child(node, node.children.items())
                    search_path.append(node)

                # Inside the search tree we use the dynamics function to obtain the next
                # hidden state given an action and the previous hidden state.
                parent = search_path[-2]
                actions.append(torch.tensor(action))
                hidden_states.append(parent.hidden_state)
                nodes.append(node)
                search_paths.append(search_path)

            with torch.no_grad():
                actions = torch.stack(actions, 0).to(self.args.device)
                hidden_states = torch.stack(hidden_states, 0).to(self.args.device)
                network_output = self.network.inference(hidden_states, actions)

            for i in range(obs_tensor.shape[0]):
                self.expand_node(nodes[i], network_output[i])
                self.backpropagate(search_paths[i], network_output[i].value)
        return roots

    def expand_node(self, node, network_output):
        node.hidden_state = network_output.next_state
        node.reward = network_output.reward
        policy_probs = F.softmax(network_output.policy_logits, dim=-1)
        for action in range(self.env.action_space[0].n):
            node.children[action] = Node(policy_probs[action].item())

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    def add_exploration_noise(self, node: Node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.args.root_dirichlet_alpha] * self.env.action_space[0].n)
        frac = self.args.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    # Select the child with the highest UCB score.
    def select_child(self, node: Node):
        _, action, child = max(
            (self.ucb_score(node, child), action,
             child) for action, child in node.children.items())
        return action, child

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    def ucb_score(self, parent: Node, child: Node) -> float:
        pb_c = math.log((parent.visit_count + self.args.pb_c_base + 1) /
                        self.args.pb_c_base) + self.args.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = self.min_max_stats.normalize(child.value())
        return prior_score + value_score

    def optimized_select_child(self, parent, children):
        # TODO: Vectorize this
        pb_c = np.array([math.log((parent.visit_count + self.args.pb_c_base + 1) /
                        self.args.pb_c_base) + self.args.pb_c_init] * len(children))
        child_visit_counts, child_priors, child_values = zip(*[(child.visit_count, child.prior, child.value())
                                                               for action, child in children])
        child_visit_counts = np.array(child_visit_counts)
        child_visit_counts += 1
        pb_c *= math.sqrt(parent.visit_count)
        pb_c = pb_c / child_visit_counts
        prior_scores = pb_c * child_priors

        value_scores = np.array([child.value() for action, child in children])
        value_scores = self.min_max_stats.normalize(value_scores)
        action_argmax = np.argmax(prior_scores + value_scores)
        children = list(children)
        return children[action_argmax][0], children[action_argmax][1]

    def backpropagate(self, search_path: List[Node], value: float):
        # TODO: Rename to backup
        # TODO: Vectorize this using masking
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            self.min_max_stats.update(node.value().item())

            value = node.reward + self.args.discount * value

    def select_action(self, node: Node):
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        visit_counts = torch.tensor([x[0] for x in visit_counts], dtype=torch.float32)
        t = self.visit_softmax_temperature()
        policy = Categorical(logits=F.log_softmax(visit_counts / t, dim=-1))
        action = policy.sample().item()
        return action, policy

    def visit_softmax_temperature(self, training_steps=0):
        # TODO: Change the temperature schedule
        if training_steps < 500e3:
            return 1.0
        elif training_steps < 750e3:
            return 0.5
        else:
            return 0.25
