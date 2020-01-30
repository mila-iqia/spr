import torch
import torch.multiprocessing as mp
import sys
from enum import Enum
import time
import gym
from gym.vector.utils import CloudpickleWrapper
from torch.multiprocessing.queue import Queue

from src.pizero import MinMaxStats, Node, MCTS


class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'


class AsyncMCTS:
    def __init__(self, args, network):
        self.args = args
        self.num_workers = args.num_workers

        ctx = mp.get_context('spawn')
        self.processes = []
        self.error_queue = ctx.Queue()
        target = worker

        self.send_queues = []
        self.receive_queues = []
        self.network = network
        for idx in range(self.args.num_workers):
            receive_queue = ctx.Queue()
            send_queue = ctx.Queue()
            process = ctx.Process(target=target,
                                  name='Worker<{0}>-{1}'.format(type(self).__name__, idx),
                                  args=(idx, self.network, args, send_queue, receive_queue, self.error_queue))
            self.receive_queues.append(receive_queue)
            self.send_queues.append(send_queue)
            self.processes.append(process)
            process.start()

        self._state = AsyncState.DEFAULT

    def run_mcts(self):
        self.run_mcts_async()
        return self.run_mcts_wait()

    def run_mcts_async(self):
        for q in self.send_queues:
            q.put(('run_mcts', None))

    def run_mcts_wait(self):
        results, successes = zip(*[receive_queue.get() for receive_queue in self.receive_queues])
        obs, actions, reward, done, policy_probs, values = map(torch.cat, zip(*results))
        return obs, actions, reward, done, policy_probs, values

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


def worker(index, network, args, send_queue, receive_queue, error_queue):
    envs_per_worker = args.num_envs // args.num_workers
    env = gym.vector.make('atari-v0', num_envs=envs_per_worker, args=args,
                          asynchronous=False)
    mcts = MCTS(args, env.action_space[0].n, network)
    obs = env.reset()

    try:
        while True:
            if send_queue.qsize() > 0:
                command, data = send_queue.get()
                if command == 'close':
                    receive_queue.put((None, True))
                    break
            elif receive_queue.qsize() < 10:
                obs = torch.from_numpy(obs)
                roots = mcts.batched_run(obs)
                actions, policy_probs, values = [], [], []
                for root in roots:
                    # Select action for each obs
                    action, policy = mcts.select_action(root)
                    actions.append(action)
                    policy_probs.append(policy.probs)
                    values.append(root.value())
                next_obs, reward, done, infos = env.step(actions)
                actions = torch.tensor(actions)
                reward, done, policy_probs, values = torch.from_numpy(reward).float(),\
                                                     torch.from_numpy(done).float(), \
                                                     torch.stack(policy_probs).float(),\
                                                     torch.tensor(values).float()
                receive_queue.put(((obs, actions, reward, done, policy_probs, values), True))
                obs = next_obs
            else:
                time.sleep(0.1)

    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        print(sys.exc_info()[:2])
        receive_queue.put((None, False))
    finally:
        return