from src.async.worker import Worker
from src.dqn import DQN
from src.envs import Env
from src.episodes import get_random_agent_episodes, get_current_policy_episodes


class WorkerData(Worker):
    def __init__(self, args, dqn, forward_model, encoder):
        super().__init__(args)
        self.dqn = dqn
        self.forward_model = forward_model
        self.encoder = encoder

    def prepare_start(self):
        transitions = get_random_agent_episodes(self.args)
        self.push(transitions)

    def pull(self):
        self.policy_state_dict = self.queue_prev.get()

    def push(self, samples):
        self.queue.put(samples)

    def step(self):
        self.dqn.load_state_dict(self.policy_state_dict)
        transitions = get_current_policy_episodes(self.args, 1, self.dqn, self.forward_model, self.encoder, 0.05)
        self.push(transitions)


