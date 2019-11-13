from collections import deque
from multiprocessing import Process, Pipe, Queue

import torch
import numpy as np
import os
import wandb

from src.agent import Agent
from src.async.worker_data import WorkerData
from src.async.worker_model import WorkerModel
from src.async.worker_policy import WorkerPolicy
from src.memory import ReplayMemory
from src.encoders import NatureCNN, ImpalaCNN
from src.envs import Env
from src.eval import test
from src.forward_model import ForwardModel
from src.stdim import InfoNCESpatioTemporalTrainer
from src.dqn_multi_step_stdim_with_actions import MultiStepActionInfoNCESpatioTemporalTrainer
from src.utils import get_argparser, log
from src.episodes import get_random_agent_episodes, Transition, sample_real_transitions


class ParallelTrainer(object):
    def __init__(self, args):
        self.args = args
        env = Env(args)
        dqn = Agent(args, env)
        encoder, forward_model = None, None  # FIXME: Initialize these properly
        self.names = ["Data", "Model", "Policy"]
        queues = [Queue(-1) for _ in range(3)]
        parent_conns, worker_conns = zip(*[Pipe() for _ in range(3)])
        worker_instances = [
            WorkerData(args, env, dqn, encoder),
            WorkerModel(args, forward_model),
            WorkerPolicy(args, dqn, forward_model)
        ]
        self.ps = [
            Process(
                target=worker_instance,
                name=name,
                args=(
                    worker_conn, queue_prev, queue, queue_next
                )
            ) for (worker_instance, name, worker_conn,
                   queue_prev, queue, queue_next) in zip(
                worker_instances, self.names, worker_conns,
                queues[2:] + queues[:2], queues, queues[1:] + queues[:1]
            )
        ]
        self.parent_conns = parent_conns
        self.queues = queues

    def train(self):
        data_parent_conn, model_parent_conn, policy_parent_conn = self.parent_conns
        data_queue, model_queue, policy_queue = self.queues

        data_parent_conn.send('prepare to launch')
        assert data_parent_conn.recv() == 'data ready'

        model_parent_conn.send('prepare to launch')
        assert model_parent_conn.recv() == 'model ready'

        policy_parent_conn.send('prepare to launch')
        assert policy_parent_conn.recv() == 'policy ready'

        for conn in self.parent_conns:
            conn.send('step')


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()

    tags = []
    if len(args.name) > 0:
        wandb.init(project=args.wandb_proj, tags=tags, name=args.name, entity="abs-world-models")
    else:
        wandb.init(project=args.wandb_proj, tags=tags, entity="abs-world-models")
    wandb.config.update(vars(args))

    results_dir = os.path.join('results', args.id)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')

    trainer = ParallelTrainer(args)
    trainer.train()
