import torch
import torch.multiprocessing as mp
import sys
from enum import Enum
import time

from baselines.common.vec_env import CloudpickleWrapper

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
        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = worker

        for idx in range(self.args.num_envs):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(target=target,
                                  name='Worker<{0}>-{1}'.format(type(self).__name__, idx),
                                  args=(idx, CloudpickleWrapper(mcts), child_pipe, parent_pipe, self.error_queue))
            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)
            process.start()
            child_pipe.close()

        self._state = AsyncState.DEFAULT

    def run(self, obs_tensor):
        obs_tensor = obs_tensor.to(self.args.device)
        with torch.no_grad():
            hidden_state, reward, policy_logits, value = self.mcts.network.initial_inference(obs_tensor)
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
                hidden_state.share_memory_()
                reward.share_memory_()
                policy_logits.share_memory_()
                value.share_memory_()

            roots = self.expand_node(roots, all_actions, hidden_state, reward, policy_logits)
            roots = self.backup(all_actions, value, roots)
        return roots

    def expand_node(self, roots, all_actions, hidden_states, rewards, policy_logitss):
        self.expand_node_async(roots, all_actions, hidden_states, rewards, policy_logitss)
        return self.expand_node_wait(timeout=5)

    def expand_node_async(self, roots, all_actions, hidden_states, rewards, policy_logitss):
        for pipe, root, actions, hidden_state, reward, policy_logits in \
                zip(self.parent_pipes, roots, all_actions, hidden_states, rewards, policy_logitss):
            pipe.send(('expand_node', (root, actions, hidden_state, reward, policy_logits)))

    def expand_node_wait(self, timeout=None):
        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError('The call to `step_wait` has timed out after '
                                  '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        results, successes = map(list, zip(*[pipe.recv() for pipe in self.parent_pipes]))
        self._raise_if_errors(successes)
        return results

    def add_exploration_noise(self, roots):
        self.add_exploration_noise_async(roots)
        return self.add_exploration_noise_wait()

    def add_exploration_noise_async(self, roots):
        for pipe, root in zip(self.parent_pipes, roots):
            pipe.send(('add_exploration_noise', root))

    def add_exploration_noise_wait(self):
        results, successes = map(list, zip(*[pipe.recv() for pipe in self.parent_pipes]))
        self._raise_if_errors(successes)
        return results

    def run_selection(self, roots):
        self.run_selection_async(roots)
        return self.run_selection_wait()

    def run_selection_async(self, roots):
        for pipe, root in zip(self.parent_pipes, roots):
            pipe.send(('run_selection', root))
        pass

    def run_selection_wait(self):
        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        actions, all_actions, search_paths, hidden_states, roots = map(list, zip(*results))
        self._raise_if_errors(successes)
        return actions, all_actions, search_paths, hidden_states, roots

    def backup(self, all_actions, values, roots):
        self.backup_async(all_actions, values, roots)
        return self.backup_wait()

    def backup_async(self, all_actions, values, roots):
        for pipe, actions, value, root in zip(self.parent_pipes, all_actions, values, roots):
            pipe.send(('backup', (actions, value, root)))

    def backup_wait(self):
        results, successes = map(list, zip(*[pipe.recv() for pipe in self.parent_pipes]))
        self._raise_if_errors(successes)
        return results

    def close(self):
        for pipe in self.parent_pipes:
            pipe.send(('close', None))

    def _poll(self, timeout=None):
        if timeout is None:
            return True
        end_time = time.time() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.time(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

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
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

        print('Raising the last exception back to the main process.')
        raise exctype(value)


def worker(index, mcts, pipe, parent_pipe, error_queue):
    parent_pipe.close()
    mcts = mcts.x
    try:
        while True:
            command, data = pipe.recv()
            if command == 'expand_node':
                root = mcts.expand_node(*data)
                pipe.send((root, True))

            elif command == 'add_exploration_noise':
                root = mcts.add_exploration_noise(data)
                pipe.send((root, True))

            elif command == 'run_selection':
                action, actions, search_path, hidden_state, root = mcts.run_selection(data)
                pipe.send(((action, actions, search_path, hidden_state, root), True))

            elif command == 'backup':
                root = mcts.backpropagate(*data)
                pipe.send((root, True))

            elif command == 'close':
                pipe.send((None, True))
                break
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        print(sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        return