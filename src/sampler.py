from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args

from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.gpu.action_server import ActionServer
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector,
    GpuEvalCollector)
from rlpyt.utils.collections import namedarraytuple, AttrDict
from rlpyt.utils.synchronize import drain_queue
from rlpyt.samplers.collectors import (DecorrelatingStartCollector,
    BaseEvalCollector)
from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.collections import (Samples, AgentSamples, AgentSamplesBsv,
    EnvSamples)




import multiprocessing as mp
import numpy as np

# For sampling, serial sampler can use Cpu collectors.
class SerialEvalCollector(BaseEvalCollector):
    """Does not record intermediate data."""

    def __init__(
            self,
            envs,
            agent,
            TrajInfoCls,
            max_T,
            max_trajectories=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        prev_dones = set()
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    if not b in prev_dones:
                        completed_traj_infos.append(traj_infos[b].terminate(o))
                    prev_dones.add(b)
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (self.max_trajectories is not None and
                    len(completed_traj_infos) >= self.max_trajectories):
                logger.log("Evaluation reached max num trajectories "
                    f"({self.max_trajectories}).")
                break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                f"({self.max_T}).")
        return completed_traj_infos


class SerialSampler(BaseSampler):
    """The simplest sampler; no parallelism, everything occurs in same, master
    Python process.  This can be easier for debugging (e.g. can use
    ``breakpoint()`` in master process) and might be fast enough for
    experiment purposes.  Should be used with collectors which generate the
    agent's actions internally, i.e. CPU-based collectors but not GPU-based
    ones.
    """

    def __init__(self, *args, CollectorCls=CpuResetCollector,
            eval_CollectorCls=SerialEvalCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    def initialize(
            self,
            agent,
            affinity=None,
            seed=None,
            bootstrap_value=False,
            traj_info_kwargs=None,
            rank=0,
            world_size=1,
            ):
        """Store the input arguments.  Instantiate the specified number of environment
        instances (``batch_B``).  Initialize the agent, and pre-allocate a memory buffer
        to hold the samples collected in each batch.  Applies ``traj_info_kwargs`` settings
        to the `TrajInfoCls` by direct class attribute assignment.  Instantiates the Collector
        and, if applicable, the evaluation Collector.

        Returns a structure of inidividual examples for data fields such as `observation`,
        `action`, etc, which can be used to allocate a replay buffer.
        """
        B = self.batch_spec.B
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(envs[0].spaces, share_memory=False,
            global_B=global_B, env_ranks=env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
            self.batch_spec, bootstrap_value, agent_shared=False,
            env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=agent,
            global_B=global_B,
            env_ranks=env_ranks,  # Might get applied redundantly to agent.
        )
        if self.eval_n_envs > 0:  # May do evaluation.
            eval_envs = [self.EnvCls(**self.eval_env_kwargs)
                for _ in range(self.eval_n_envs)]
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=agent,
                TrajInfoCls=self.TrajInfoCls,
                max_T=self.eval_max_steps // self.eval_n_envs,
                max_trajectories=self.eval_max_trajectories,
            )

        agent_inputs, traj_infos = collector.start_envs(
            self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent
        self.samples_pyt = samples_pyt
        self.samples_np = samples_np
        self.collector = collector
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        logger.log("Serial Sampler initialized.")
        return examples

    def obtain_samples(self, itr):
        """Call the collector to execute a batch of agent-environment interactions.
        Return data in torch tensors, and a list of trajectory-info objects from
        episodes which ended.
        """
        # self.samples_np[:] = 0  # Unnecessary and may take time.
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)
        self.collector.reset_if_needed(agent_inputs)
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return self.samples_pyt, completed_infos

    def evaluate_agent(self, itr):
        """Call the evaluation collector to execute agent-environment interactions."""
        return self.eval_collector.collect_evaluation(itr)


StepBuffer = namedarraytuple("StepBuffer",
    ["observation", "action", "reward", "done", "agent_info"])


class GpuSamplerBase(ParallelSamplerBase):
    """Base class for parallel samplers which use worker processes to execute
    environment steps on CPU resources but the master process to execute agent
    forward passes for action selection, presumably on GPU.  Use GPU-based
    collecter classes.

    In addition to the usual batch buffer for data samples, allocates a step
    buffer over shared memory, which is used for communication with workers.
    The step buffer includes `observations`, which the workers write and the
    master reads, and `actions`, which the master write and the workers read.
    (The step buffer has leading dimension [`batch_B`], for the number of
    parallel environments, and each worker gets its own slice along that
    dimension.)  The step buffer object is held in both numpy array and torch
    tensor forms over the same memory; e.g. workers write to the numpy array
    form, and the agent is able to read the torch tensor form.

    (Possibly more information about how the stepping works, but write
    in action-server or smwr like that.)
    """

    gpu = True

    def __init__(self, *args, CollectorCls=GpuResetCollector,
            eval_CollectorCls=GpuEvalCollector, **kwargs):
        # e.g. or use GpuWaitResetCollector, etc...
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    def obtain_samples(self, itr):
        """Signals worker to begin environment step execution loop, and drops
        into ``serve_actions()`` method to provide actions to workers based on
        the new observations at each step.
        """
        # self.samples_np[:] = 0  # Reset all batch sample values (optional).
        self.agent.sample_mode(itr)
        self.ctrl.barrier_in.wait()
        self.serve_actions(itr)  # Worker step environments here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        """Signals workers to begin agent evaluation loop, and drops into
        ``serve_actions_evaluation()`` to provide actions to workers at each
        step.
        """
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.agent.eval_mode(itr)
        self.ctrl.barrier_in.wait()
        traj_infos = self.serve_actions_evaluation(itr)
        self.ctrl.barrier_out.wait()
        traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
            n_sentinel=self.n_worker))  # Block until all finish submitting.
        self.ctrl.do_eval.value = False
        return traj_infos

    def _agent_init(self, agent, env, global_B=1, env_ranks=None):
        """Initializes the agent, having it *not* share memory, because all
        agent functions (training and sampling) happen in the master process,
        presumably on GPU."""
        agent.initialize(env.spaces, share_memory=False,  # No share memory.
            global_B=global_B, env_ranks=env_ranks)
        self.agent = agent

    def _build_buffers(self, *args, **kwargs):
        examples = super()._build_buffers(*args, **kwargs)
        self.step_buffer_pyt, self.step_buffer_np = build_step_buffer(
            examples, self.batch_spec.B)
        self.agent_inputs = AgentInputs(self.step_buffer_pyt.observation,
            self.step_buffer_pyt.action, self.step_buffer_pyt.reward)
        if self.eval_n_envs > 0:
            self.eval_step_buffer_pyt, self.eval_step_buffer_np = \
                build_step_buffer(examples, self.eval_n_envs)
            self.eval_agent_inputs = AgentInputs(
                self.eval_step_buffer_pyt.observation,
                self.eval_step_buffer_pyt.action,
                self.eval_step_buffer_pyt.reward,
            )
        return examples

    def _build_parallel_ctrl(self, n_worker):
        super()._build_parallel_ctrl(n_worker)
        self.sync.obs_ready = [mp.Semaphore(0) for _ in range(n_worker)]
        self.sync.act_ready = [mp.Semaphore(0) for _ in range(n_worker)]

    def _assemble_common_kwargs(self, *args, **kwargs):
        common_kwargs = super()._assemble_common_kwargs(*args, **kwargs)
        common_kwargs["agent"] = None  # Remove.
        return common_kwargs

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        workers_kwargs = super()._assemble_workers_kwargs(affinity, seed,
            n_envs_list)
        i_env = 0
        for rank, w_kwargs in enumerate(workers_kwargs):
            n_envs = n_envs_list[rank]
            slice_B = slice(i_env, i_env + n_envs)
            w_kwargs["sync"] = AttrDict(
                stop_eval=self.sync.stop_eval,
                obs_ready=self.sync.obs_ready[rank],
                act_ready=self.sync.act_ready[rank],
            )
            w_kwargs["step_buffer_np"] = self.step_buffer_np[slice_B]
            if self.eval_n_envs > 0:
                eval_slice_B = slice(self.eval_n_envs_per * rank,
                    self.eval_n_envs_per * (rank + 1))
                w_kwargs["eval_step_buffer_np"] = \
                    self.eval_step_buffer_np[eval_slice_B]
            i_env += n_envs
        return workers_kwargs


class GpuSampler(ActionServer, GpuSamplerBase):
    pass


def build_step_buffer(examples, B):
    step_bufs = {k: buffer_from_example(examples[k], B, share_memory=True)
        for k in ["observation", "action", "reward", "done", "agent_info"]}
    step_buffer_np = StepBuffer(**step_bufs)
    step_buffer_pyt = torchify_buffer(step_buffer_np)
    return step_buffer_pyt, step_buffer_np


class GpuEvalCollector(BaseEvalCollector):
    """Offline agent evaluation collector which communicates observations
    to an action-server, which in turn provides the agent's actions.
    """

    def collect_evaluation(self, itr):
        """Param itr unused."""
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        act_ready, obs_ready = self.sync.act_ready, self.sync.obs_ready
        step = self.step_buffer_np
        for b, env in enumerate(self.envs):
            step.observation[b] = env.reset()
        step.done[:] = False
        obs_ready.release()

        prev_dones = set()
        for t in range(self.max_T):
            act_ready.acquire()
            if self.sync.stop_eval.value:
                obs_ready.release()  # Always release at end of loop.
                break
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r, d,
                    step.agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    if b not in prev_dones:
                        self.traj_infos_queue.put(traj_infos[b].terminate(o))
                    prev_dones.add(b)
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                step.observation[b] = o
                step.reward[b] = r
                step.done[b] = d
            obs_ready.release()
        self.traj_infos_queue.put(None)  # End sentinel.
