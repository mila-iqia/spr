import collections
import math
import numpy as np
from typing import Dict, List, Optional
import torch.nn.functional as F
import torch
from torch.distributions import Categorical

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


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
    def __init__(self, prior: float, value: float = 0., c1: float = 0.5):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = value
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.c1 = c1

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def compute_pi_bar(self):
        q_values = np.array([self.children[i].value() for i in self.children])
        prior = np.array([self.children[i].prior for i in self.children])
        lambda_n = self.c1 * math.sqrt(self.visit_count) / self.visit_count
        alpha = self.binary_search_alpha(lambda_n, q_values, prior)
        pi_bar = lambda_n * self.prior / (alpha - q_values)
        return pi_bar

    def binary_search_alpha(self, lambda_n, q_values, prior,  eps=1e-8):
        """
        Find the value of alpha with binary search, using the procedure described in B.3.
        """
        alpha_min = max([q_values[b] + (lambda_n * prior[b]) for b in range(len(q_values))])
        alpha_max = max([q_values[b] + lambda_n for b in range(len(q_values))])
        low, high = alpha_min, alpha_max
        alpha = (low + high) / 2
        while high - low > eps:
            sum_pi_bar = sum([lambda_n * prior[a] / (alpha - q_values[a]) for a in range(len(prior))])
            if sum_pi_bar < 1:
                high = alpha
            else:
                low = alpha
            alpha = (low + high) / 2
        return alpha

    def select_action(self, pi_bar=None):
        if pi_bar is None:
            pi_bar = self.compute_pi_bar()
        d = torch.distributions.Categorical(probs=pi_bar)
        action = d.sample()
        return action


class MCTS:
    def __init__(self, args, n_actions, network, eval=False):
        self.args = args
        self.network = network
        self.n_actions = n_actions
        self.min_max_stats = MinMaxStats()
        self.eval = eval

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
            self.backup(search_path, network_output.value)
        pi_bar = root.compute_pi_bar()
        action = root.select_action(pi_bar)
        # TODO: Clarify: should the returned values be pi_bar*q_values?
        return action, pi_bar, root.value()

    def expand_node(self, node, network_output):
        node.hidden_state = network_output.next_state
        node.reward = network_output.reward
        policy_probs = F.softmax(network_output.policy_logits, dim=-1)
        for action in range(self.n_actions):
            node.children[action] = Node(policy_probs[action].item(), self.min_max_stats.minimum)

    # At the start of each search, we add dirichlet noise to the prior of the root
    # to encourage the search to explore new actions.
    def add_exploration_noise(self, node: Node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.args.root_dirichlet_alpha] * self.n_actions)
        frac = self.args.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def select_child(self, node):
        pi_bar = node.compute_pi_bar(node)
        # pi_bar is a torch categorical distribution
        if self.eval:
            action = pi_bar.probs.mode.indices.item()
        else:
            action = pi_bar.sample()
        child = node[action]
        return action, child

    def backup(self, search_path: List[Node], value: float):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            self.min_max_stats.update(node.value())

            value = node.reward + self.args.discount * value
