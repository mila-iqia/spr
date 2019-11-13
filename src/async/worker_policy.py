from src.async.worker import Worker


class WorkerPolicy(Worker):
    def __init__(self, args, dqn, forward_model):
        super().__init__(args)
        self.dqn = dqn
        self.forward_model = forward_model

    def prepare_start(self):
        pass

    def pull(self):
        pass

    def push(self, parameters):
        pass

    def step(self):
        pass
