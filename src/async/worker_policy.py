from src.async.worker import Worker


class WorkerPolicy(Worker):
    def __init__(self, args, dqn, forward_model):
        super().__init__(args)
        self.dqn = dqn
        self.forward_model = forward_model

    def prepare_start(self):
        pass

    def pull(self):
        self.forward_model.load_state_dict(self.queue_prev.get())

    def push(self, parameters):
        self.queue.push(self.dqn.state_dict())

    def generate_model_transitions(self):
        return ''

    def step(self):
        model_transitions = self.generate_model_transitions()
        self.dqn.learn(model_transitions)
