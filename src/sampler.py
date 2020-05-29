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


class SerialEvalCollectorFixed(SerialEvalCollector):
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
        live_envs = list(range(len(self.envs)))
        for t in range(self.max_T):

            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)

            b = 0
            while b < len(live_envs):  # don't want to do a for loop on live envs, since we change it during the process.
                env_id = live_envs[b]
                env = self.envs[env_id]
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                                   agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    observation = delete_ind_from_array(observation, b)
                    reward = delete_ind_from_array(reward, b)
                    action = delete_ind_from_array(action, b)
                    obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))

                    completed_traj_infos.append(traj_infos[env_id].terminate(o))
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
