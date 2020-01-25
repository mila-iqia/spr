import torch
import torch.multiprocessing as mp
import sys
from enum import Enum
import time
import wandb

from gym.vector.utils import CloudpickleWrapper
from torch.multiprocessing.queue import Queue

from src.pizero import MinMaxStats, Node


class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'


class AsyncMCTS:
    def __init__(self, args, mcts):
        self.args = args
        self.mcts = mcts
        self.min_max_stats = MinMaxStats()

        ctx = mp.get_context('spawn')
        self.processes = []
        self.error_queue = ctx.Queue()
        target = worker

        self.send_queue, self.receive_queue = ctx.Queue(), ctx.Queue()
        for idx in range(self.args.num_envs):
            process = ctx.Process(target=target,
                                  name='Worker<{0}>-{1}'.format(type(self).__name__, idx),
                                  args=(idx, CloudpickleWrapper(mcts), self.send_queue, self.receive_queue, self.error_queue))
            self.processes.append(process)
            process.start()

        self._state = AsyncState.DEFAULT

    def run(self, obs_tensor):
        obs_tensor = obs_tensor.to(self.args.device)
        with torch.no_grad():
            hidden_state, reward, policy_logits, value = self.mcts.network.initial_inference(obs_tensor)
            # Need to send these to CPU becuase we are passing back roots from child process.
            # Otherwise PyTorch complains:
            # RuntimeError: Attempted to send CUDA tensor received from another process;
            # this is not currently supported. Consider cloning before sending.
            hidden_state, reward, policy_logits, value = hidden_state.cpu(), reward.cpu(), policy_logits.cpu(), value.cpu()
            hidden_state.share_memory_()
            reward.share_memory_()
            policy_logits.share_memory_()
            value.share_memory_()
        roots = [Node(0)] * self.args.num_envs
        roots = self.expand_node(roots, [[] for _ in range(self.args.num_envs)], hidden_state, reward, policy_logits)
        roots = self.add_exploration_noise(roots)

        for i in range(self.args.num_simulations):
            actions, all_actions, search_paths, hidden_states, roots = self.run_selection(roots)

            with torch.no_grad():
                actions = torch.tensor(actions).unsqueeze(1).to(self.args.device)
                hidden_states = torch.stack(hidden_states, 0).to(self.args.device)
                hidden_state, reward, policy_logits, value = self.mcts.network.inference(hidden_states, actions)
                hidden_state, reward, policy_logits, value = hidden_state.cpu(), reward.cpu(), policy_logits.cpu(), value.cpu()
                hidden_state.share_memory_()
                reward.share_memory_()
                policy_logits.share_memory_()
                value.share_memory_()

            roots = self.expand_and_backup(roots, all_actions, hidden_state, reward, policy_logits, value)

        return roots

    def expand_node(self, roots, all_actions, hidden_states, rewards, policy_logitss):
        self.expand_node_async(roots, all_actions, hidden_states, rewards, policy_logitss)
        return self.expand_node_wait(timeout=5)

    def expand_node_async(self, roots, all_actions, hidden_states, rewards, policy_logitss):
        for root, actions, hidden_state, reward, policy_logits in \
                zip(roots, all_actions, hidden_states, rewards, policy_logitss):
            self.send_queue.put(('expand_node', (root, actions, hidden_state, reward, policy_logits)))

    def expand_node_wait(self, timeout=None):
        results, successes = map(list, zip(*[self.receive_queue.get() for _ in range(self.args.num_envs)]))
        self._raise_if_errors(successes)
        return results

    def add_exploration_noise(self, roots):
        self.add_exploration_noise_async(roots)
        return self.add_exploration_noise_wait()

    def add_exploration_noise_async(self, roots):
        for root in roots:
            self.send_queue.put(('add_exploration_noise', root))

    def add_exploration_noise_wait(self):
        results, successes = map(list, zip(*[self.receive_queue.get() for _ in range(self.args.num_envs)]))
        self._raise_if_errors(successes)
        return results

    def run_selection(self, roots):
        self.run_selection_async(roots)
        return self.run_selection_wait()

    def run_selection_async(self, roots):
        for root in roots:
            self.send_queue.put(('run_selection', root))

    def run_selection_wait(self):
        results, successes = zip(*[self.receive_queue.get() for _ in range(self.args.num_envs)])
        actions, all_actions, search_paths, hidden_states, roots = map(list, zip(*results))
        self._raise_if_errors(successes)
        return actions, all_actions, search_paths, hidden_states, roots

    def backup(self, all_actions, values, roots):
        self.backup_async(all_actions, values, roots)
        return self.backup_wait()

    def backup_async(self, all_actions, values, roots):
        for actions, value, root in zip(all_actions, values, roots):
            self.send_queue.put(('backup', (actions, value, root)))

    def backup_wait(self):
        results, successes = map(list, zip(*[self.receive_queue.get() for _ in range(self.args.num_envs)]))
        self._raise_if_errors(successes)
        return results

    def expand_and_backup(self, roots, all_actions, hidden_states, rewards, policy_logits, values):
        self.expand_and_backup_async(roots, all_actions, hidden_states, rewards, policy_logits, values)
        return self.expand_node_wait()

    def expand_and_backup_async(self, roots, all_actions, hidden_states, rewards, policy_logits, values):
        for root, actions, hidden_state, reward, policy_logit, value in zip(roots, all_actions, hidden_states, rewards, policy_logits, values):
            self.send_queue.put(('expand_and_backup', (root, actions, hidden_state, reward, policy_logit, value)))

    def expand_and_backup_wait(self):
        results, successes = map(list, zip(*[self.receive_queue.get() for _ in range(self.args.num_envs)]))
        self._raise_if_errors(successes)
        return results

    def close(self):
        self.send_queue.put(('close', None))

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.args.num_envs - sum(successes)
        assert num_errors > 0
        for _ in range(num_errors):
            index, exctype, value = self.error_queue.get()
            print('Received the following error from Worker-{0}: '
                         '{1}: {2}'.format(index, exctype.__name__, value))
            print('Shutting down Worker-{0}.'.format(index))

        print('Raising the last exception back to the main process.')
        raise exctype(value)


def worker(index, mcts, send_queue, recieve_queue, error_queue):
    mcts = mcts.fn
    try:
        while True:
            command, data = send_queue.get()
            if command == 'expand_node':
                root = mcts.expand_node(*data)
                recieve_queue.put((root, True))

            elif command == 'add_exploration_noise':
                root = mcts.add_exploration_noise(data)
                recieve_queue.put((root, True))

            elif command == 'run_selection':
                action, actions, search_path, hidden_state, root = mcts.run_selection(data)
                recieve_queue.put(((action, actions, search_path, hidden_state, root), True))

            elif command == 'backup':
                root = mcts.backpropagate(*data)
                recieve_queue.put((root, True))

            elif command == 'expand_and_backup':
                root = mcts.expand_and_backup(*data)
                recieve_queue.put((root, True))

            elif command == 'close':
                recieve_queue.put((None, True))
                break
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        print(sys.exc_info()[:2])
        recieve_queue.put((None, False))
    finally:
        return