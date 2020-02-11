import torch
import torch.multiprocessing as mp
from enum import Enum
import sys
import gym
import time
import glob
import numpy as np
import traceback

from src.vectorized_mcts import VectorizedMCTS as MCTS
from recordclass import dataobject


class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'


class ListEpisode(dataobject):
    obs: list
    actions: list
    rewards: list
    dones: list


class TensorEpisode(dataobject):
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class AsyncReanalyze:
    def __init__(self, args, network, debug=False):
        self.args = args
        self.num_workers = args.num_workers

        self.processes = []
        ctx = mp.get_context('spawn')
        self.error_queue = ctx.Queue()
        self.debug = debug

        self.send_queues = []
        self.receive_queues = []
        self.write_heads = args.num_envs//args.num_reanalyze_workers
        self.read_heads = args.num_reanalyze_envs//args.num_reanalyze_workers
        self.network = network
        dummy_env = gym.vector.make('atari-v0', num_envs=1, args=args,
                                    asynchronous=False)
        self.n_actions = dummy_env.action_space[0].n
        self.obs_shape = dummy_env.observation_space.shape[2:]
        dummy_env.close()
        for idx in range(self.args.num_reanalyze_workers):
            receive_queue = ctx.Queue()
            send_queue = ctx.Queue()
            if not self.debug:
                process = ctx.Process(target=reanalyze_wrapper,
                                      name='ReanalyzeWorker-{}'.format(idx),
                                      args=((
                                          args,
                                          idx,
                                          'ReanalyzeWorker - {}'.format(idx),
                                          self.network,
                                          self.write_heads,
                                          self.read_heads,
                                          send_queue,
                                          receive_queue,
                                          self.error_queue,
                                          self.obs_shape,
                                          self.n_actions)))
                process.start()

            else:
                process = ReanalyzeWorker(args,
                                          idx,
                                          'ReanalyzeWorker - {}'.format(idx),
                                          self.network,
                                          self.write_heads,
                                          self.read_heads,
                                          send_queue,
                                          receive_queue,
                                          self.error_queue,
                                          self.obs_shape,
                                          self.n_actions)

            self.receive_queues.append(receive_queue)
            self.send_queues.append(send_queue)
            self.processes.append(process)

        self._state = AsyncState.DEFAULT

    def store_transitions(self, obs, actions, rewards, dones):
        for i in range(len(self.processes)):
            i_obs = obs[i*self.write_heads:(i+1)*self.write_heads]
            i_actions = actions[i*self.write_heads:(i+1)*self.write_heads]
            i_rewards = rewards[i*self.write_heads:(i+1)*self.write_heads]
            i_dones = dones[i*self.write_heads:(i+1)*self.write_heads]
            if self.debug:
                self.processes[i].save_data(i_obs, i_actions, i_rewards, i_dones)
            else:
                self.send_queues[i].put((i_obs, i_actions, i_rewards, i_dones))

    def get_blank_transitions(self):
        obs = torch.zeros(self.args.num_reanalyze_envs,
                          self.args.framestack,
                          *self.obs_shape,
                          dtype=torch.uint8)

        actions = torch.zeros(self.args.num_reanalyze_envs, dtype=torch.long)
        rewards = torch.zeros(self.args.num_reanalyze_envs)
        values = torch.zeros(self.args.num_reanalyze_envs)
        dones = torch.ones(self.args.num_reanalyze_envs)
        policies = torch.ones(self.args.num_reanalyze_envs, self.n_actions)
        policies = policies/torch.sum(policies, -1, keepdim=True)

        return obs, actions, rewards, dones, policies, values

    def get_transitions(self, total_episodes):
        if total_episodes <= 0:
            return self.get_blank_transitions()
        if self.debug:
            results = [p.sample_for_reanalysis() for p in self.processes]
            successes = [True]*len(self.processes)
        else:
            results, successes = zip(*[receive_queue.get() for receive_queue in self.receive_queues])
        self._raise_if_errors(successes)

        print("stepped")
        obs, actions, reward, done, policy_probs, values = map(torch.cat, zip(*results))
        return obs, actions, reward, done, policy_probs.cpu(), values.cpu()

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.args.num_envs - sum(successes)
        assert num_errors > 0
        for _ in range(num_errors):
            exc_info = self.error_queue.get()
            index = exc_info[0]
            exctype = exc_info[1]
            value = exc_info[2]
            traceback.print_exception(*exc_info[1:])
            print('Received the following error from Worker-{0}: '
                         '{1}: {2}'.format(index, exctype.__name__, value))
            print('Shutting down Worker-{0}.'.format(index))

        print('Raising the last exception back to the main process.')
        raise exctype(value)


def reanalyze_wrapper(*args):
    worker = ReanalyzeWorker(*args)
    worker.run()


class ReanalyzeWorker:
    def __init__(self, args, index, name, network, write_heads, read_heads,
                 send_queue, receive_queue, error_queue,
                 obs_shape, n_actions):
        self.send_queue = send_queue
        self.receive_queue = receive_queue
        self.error_queue = error_queue
        self.name = name
        self.network = network
        self.read_heads = read_heads
        self.write_heads = write_heads
        self.current_write_episodes = [ListEpisode([], [], [], []) for _ in range(self.write_heads)]
        self.current_read_episodes = []
        self.tr_indices = np.zeros((read_heads,), dtype="int")
        self.buffer = []
        self.current_indices = []
        self.directory = args.savedir
        self.index = index
        self.total_episodes = 0
        self.mcts = MCTS(args, n_actions, self.read_heads, network)
        self.obs_shape = obs_shape
        self.args = args

        self.can_reanalyze = False

    def run(self, forever=True):
        try:
            while True:
                self.write_to_buffer()

                # Check to see if there's any data to reanalyze yet.
                self.can_reanalyze = self.can_reanalyze or len(glob.glob(self.directory + "/ep*")) > 0
                # Don't reanalyze until we have data for it.
                if self.can_reanalyze and self.receive_queue.qsize() < 100:
                    new_samples = self.sample_for_reanalysis()
                    self.receive_queue.put((new_samples, True))
                elif not forever:
                    time.sleep(1.)  # Don't just spin the processor
                else:
                    return

        except (KeyboardInterrupt, Exception):
            self.error_queue.put((self.index,) + sys.exc_info())
            traceback.print_exc()
            self.receive_queue.put((None, False))
        finally:
            return

    def write_to_buffer(self):
        while not self.send_queue.empty():
            obs, action, reward, done = self.send_queue.get()
            print("Received transition")
            self.save_data(obs, action, reward, done)

    def save_data(self, obs, action, reward, done):
        for i in range(self.write_heads):
            episode = self.current_write_episodes[i]
            episode.obs.append(obs[i])
            episode.actions.append(action[i])
            episode.rewards.append(reward[i])
            episode.dones.append(done[i])

            if done[i]:
                self.save_episode(episode)

        self.current_write_episodes[i] = ListEpisode([], [], [], [])

    def save_episode(self, episode):
        obs = torch.stack(episode.obs).numpy()
        actions = torch.stack(episode.actions).numpy()
        rewards = torch.stack(episode.rewards).numpy()
        dones = torch.stack(episode.dones).numpy()

        filename = self.directory + "/ep_{}_{}.npz".format(self.index,
                                                           self.total_episodes)

        np.savez_compressed(file=filename, obs=obs, actions=actions,
                            dones=dones, rewards=rewards)
        self.total_episodes += 1

    def load_episode(self):
        success = False
        while not success:
            try:
                episodes = glob.glob(self.directory + "/ep*")
                index = np.random.randint(0, len(episodes))
                filename = episodes[index]
                file = np.load(filename)
                episode = TensorEpisode(obs=torch.from_numpy(file["obs"]),
                                        actions=torch.from_numpy(file["actions"]),
                                        rewards=torch.from_numpy(file["rewards"]),
                                        dones=torch.from_numpy(file["dones"]))
                file.close()
                return episode

            except IOError:
                traceback.print_exc()

    def sample_for_reanalysis(self):
        """
        :param buffer: list of lists of Transitions.  Each sublist is an episode.
        :return: list of new transitions, representing a reanalyzed episode.
        """
        observations = torch.zeros((self.read_heads,
                                    self.args.framestack,
                                    *self.obs_shape),
                                   dtype=torch.uint8,)
        actions = torch.zeros(self.read_heads, dtype=torch.long)
        rewards = torch.zeros(self.read_heads)
        dones = torch.zeros(self.read_heads)

        for i in range(self.read_heads):
            # Should only happen at initialization.
            if i >= len(self.current_read_episodes):
                new_ep = self.load_episode()
                self.current_read_episodes.append(new_ep)
                self.tr_indices[i] = 0

            episode = self.current_read_episodes[i]
            ind = self.tr_indices[i]

            bottom_ind = max(0, ind - self.args.framestack + 1)
            start_point = self.args.framestack - (ind - bottom_ind + 1)
            observations[i, start_point:] = episode.obs[bottom_ind:ind+1]
            actions[i] = episode.actions[ind]
            rewards[i] = episode.rewards[ind]
            dones[i] = episode.dones[ind]
            self.tr_indices[i] += 1

            # Check if we've finished the episode.
            if self.tr_indices[i] >= episode.obs.shape[0]:
                self.tr_indices[i] = 0
                # Load in a random new episode
                self.current_read_episodes[i] = \
                    self.load_episode()

        _, policies, values = self.mcts.run(observations)
        policies = policies.probs

        return observations, actions, rewards, dones, policies, values

