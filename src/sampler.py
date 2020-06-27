from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.collectors import SerialEvalCollector
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

import torch
import numpy as np


class OneToOneGpuEvalCollector(BaseEvalCollector):
    """Offline agent evaluation collector which communicates observations
    to an action-server, which in turn provides the agent's actions.
    Note that the number of eval environments must be the same as the number
    of eval trajectories.  Environments that have terminated will not be
    step()ed and their traj_infos will not be updated, but action selection
    will still be carried out for them.
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
        live_envs = set(range(len(self.envs)))
        for t in range(self.max_T):
            act_ready.acquire()
            if self.sync.stop_eval.value or len(live_envs) == 0:
                obs_ready.release()  # Always release at end of loop.
                break
            for b, env in enumerate(self.envs):
                if b in live_envs:
                    o, r, d, env_info = env.step(step.action[b])
                    traj_infos[b].step(step.observation[b], step.action[b], r, d,
                                       step.agent_info[b], env_info)
                    if getattr(env_info, "traj_done", d):
                        self.traj_infos_queue.put(traj_infos[b].terminate(o))
                        traj_infos[b] = self.TrajInfoCls()
                        o = env.reset()
                        live_envs.remove(b)  # This env is done, stop updating it
                    step.observation[b] = o
                    step.reward[b] = r
                    step.done[b] = d
            obs_ready.release()
        self.traj_infos_queue.put(None)  # End sentinel.


def delete_ind_from_tensor(tensor, ind):
    tensor = torch.cat([tensor[:ind], tensor[ind+1:]], 0)
    return tensor


def delete_ind_from_array(array, ind):
    tensor = np.concatenate([array[:ind], array[ind+1:]], 0)
    return tensor


class OneToOneSerialEvalCollector(SerialEvalCollector):
    def collect_evaluation(self, itr):
        assert self.max_trajectories == len(self.envs)
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
        live_envs = list(range(len(self.envs)))
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)

            b = 0
            while b < len(live_envs):  # don't want to do a for loop on live envs, since we change it during the process.
                env_id = live_envs[b]
                o, r, d, env_info = self.envs[env_id].step(action[b])
                traj_infos[env_id].step(observation[b],
                                        action[b], r, d,
                                        agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[env_id].terminate(o))

                    observation = delete_ind_from_array(observation, b)
                    reward = delete_ind_from_array(reward, b)
                    action = delete_ind_from_array(action, b)
                    obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))

                    del live_envs[b]
                    b -= 1  # live_envs[b] is now the next env, so go back one.
                else:
                    observation[b] = o
                    reward[b] = r

                b += 1

                if (self.max_trajectories is not None and
                        len(completed_traj_infos) >= self.max_trajectories):
                    logger.log("Evaluation reached max num trajectories "
                               f"({self.max_trajectories}).")
                    return completed_traj_infos

        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                       f"({self.max_T}).")
        return completed_traj_infos


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
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
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
        envs = [self.EnvCls(id=i, **self.env_kwargs) for i in range(B)]
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
            eval_envs = [self.EnvCls(id=i, **self.eval_env_kwargs)
                for i in range(self.eval_n_envs)]
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
