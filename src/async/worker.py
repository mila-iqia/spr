

class Worker(object):
    """
        Abstract class for worker instantiations.
    """
    def __init__(self, args):
        self.args = args

    def __call__(self, worker_conn, queue_prev, queue, queue_next):
        self.queue_prev = queue_prev
        self.queue = queue
        self.queue_next = queue_next

        assert worker_conn.recv() == 'prepare to launch'
        self.prepare_start()
        worker_conn.send('worker ready')
        while True:
            self.pull()
            self.step()
            self.push()

    def prepare_start(self):
        raise NotImplementedError

    def pull(self):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def push(self, *args, **kwargs):
        raise NotImplementedError