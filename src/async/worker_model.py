from src.async.worker import Worker
from src.forward_model import ForwardModel


class WorkerModel(Worker):
    def __init__(self, args, forward_model):
        super().__init__(args)
        self.forward_model = forward_model

    def prepare_start(self):
        pass

    def pull(self):
        samples = self.queue_prev.get()
        self.step(samples)

    def step(self, samples):
        self.forward_model.do_one_epoch(samples)

    def push(self):
        self.queue.push(self.forward_model.state_dict())
