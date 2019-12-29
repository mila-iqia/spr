import collections
import math
from typing import Dict, List, Optional
import torch.nn.functional as F
import torch
from torch.distributions import Categorical

from src.envs import Env
from src.mcts_memory import ReplayMemory
from src.model_trainer import MCTSModel

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


class PiZero():
    def __init__(self, args):
        self.args = args
        self.args.pb_c_base = 19652
        self.args.pb_c_init = 1.25
        self.env = Env(args)
        self.network = MCTSModel(args, self.env.action_space())
        self.mcts = MCTS(args, self.env, self.network)

    def evaluate(self):
        env = Env(self.args)
        env.eval()
        T_rewards, T_Qs = [], []

        # Test performance over several episodes
        done = True
        for _ in range(self.args.evaluation_episodes):
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False

                root = self.mcts.run(state)
                action = self.mcts.select_action(root)
                state, reward, done = env.step(action)  # Step
                reward_sum += reward

                if done:
                    T_rewards.append(reward_sum)
                    break
        env.close()

        avg_reward = sum(T_rewards) / len(T_rewards)
        return avg_reward


class MCTS():
    def __init__(self, args, env, network):
        self.args = args
        self.env = env
        self.network = network
        self.min_max_stats = MinMaxStats()

    def run(self, obs):
        root = Node(0)
        root.hidden_state = obs
        self.expand_node(root, network_output=self.network.initial_inference(obs))
        for _ in range(self.args.num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(node)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            with torch.no_grad():
                network_output = self.network.inference(parent.hidden_state, torch.tensor(action))
            self.expand_node(node, network_output)
            self.backpropagate(search_path, network_output.value)
        return root

    def expand_node(self, node, network_output):
        node.hidden_state = network_output.next_state
        node.reward = network_output.reward
        policy = {a: math.exp(network_output.policy_logits.squeeze()[a]) for a in range(self.env.action_space())}
        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

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

    def backpropagate(self, search_path: List[Node], value: float):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            self.min_max_stats.update(node.value())

            value = node.reward + self.args.discount * value

    def select_action(self, node: Node):
        visit_counts = [
            (child.visit_count, action) for action, child in node.children.items()
        ]
        visit_counts = torch.tensor([x[0] for x in visit_counts])
        t = self.visit_softmax_temperature()
        policy = Categorical(logits=F.log_softmax(visit_counts / t))
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