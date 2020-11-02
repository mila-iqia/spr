from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed
from rlpyt.runners.minibatch_rl import MinibatchRlEval

import wandb
import psutil

import torch
import numpy as np
import time


atari_human_scores = dict(
    alien=7127.7, amidar=1719.5, assault=742.0, asterix=8503.3,
    bank_heist=753.1, battle_zone=37187.5, boxing=12.1,
    breakout=30.5, chopper_command=7387.8, crazy_climber=35829.4,
    demon_attack=1971.0, freeway=29.6, frostbite=4334.7,
    gopher=2412.5, hero=30826.4, jamesbond=302.8, kangaroo=3035.0,
    krull=2665.5, kung_fu_master=22736.3, ms_pacman=6951.6, pong=14.6,
    private_eye=69571.3, qbert=13455.0, road_runner=7845.0,
    seaquest=42054.7, up_n_down=11693.2
)

atari_der_scores = dict(
    alien=739.9, amidar=188.6, assault=431.2, asterix=470.8,
    bank_heist=51.0, battle_zone=10124.6, boxing=0.2,
    breakout=1.9, chopper_command=861.8, crazy_climber=16185.3,
    demon_attack=508, freeway=27.9, frostbite=866.8,
    gopher=349.5, hero=6857.0, jamesbond=301.6,
    kangaroo=779.3, krull=2851.5, kung_fu_master=14346.1,
    ms_pacman=1204.1, pong=-19.3, private_eye=97.8, qbert=1152.9,
    road_runner=9600.0, seaquest=354.1, up_n_down=2877.4,
)

atari_nature_scores = dict(
    alien=3069, amidar=739.5, assault=3359,
    asterix=6012, bank_heist=429.7, battle_zone=26300.,
    boxing=71.8, breakout=401.2, chopper_command=6687.,
    crazy_climber=114103, demon_attack=9711., freeway=30.3,
    frostbite=328.3, gopher=8520., hero=19950., jamesbond=576.7,
    kangaroo=6740., krull=3805., kung_fu_master=23270.,
    ms_pacman=2311., pong=18.9, private_eye=1788.,
    qbert=10596., road_runner=18257., seaquest=5286., up_n_down=8456.
)

atari_random_scores = dict(
    alien=227.8, amidar=5.8, assault=222.4,
    asterix=210.0, bank_heist=14.2, battle_zone=2360.0,
    boxing=0.1, breakout=1.7, chopper_command=811.0,
    crazy_climber=10780.5, demon_attack=152.1, freeway=0.0,
    frostbite=65.2, gopher=257.6, hero=1027.0, jamesbond=29.0,
    kangaroo=52.0, krull=1598.0, kung_fu_master=258.5,
    ms_pacman=307.3, pong=-20.7, private_eye=24.9,
    qbert=163.9, road_runner=11.5, seaquest=68.4, up_n_down=533.4
)


def maybe_update_summary(key, value):
    if key not in wandb.run.summary.keys():
        wandb.run.summary[key] = value
    else:
        wandb.run.summary[key] = max(value, wandb.run.summary[key])


class MinibatchRlEvalWandb(MinibatchRlEval):

    def __init__(self, final_eval_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_eval_only = final_eval_only

    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size
        self.wandb_info = {'cum_steps': cum_steps}
        super().log_diagnostics(itr, eval_traj_infos, eval_time)
        wandb.log(self.wandb_info)

    def startup(self):
        """
        Sets hardware affinities, initializes the following: 1) sampler (which
        should initialize the agent), 2) agent device and data-parallel wrapper (if applicable),
        3) algorithm, 4) logger.
        """
        p = psutil.Process()
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        set_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.rank = rank = getattr(self, "rank", 0)
        self.world_size = world_size = getattr(self, "world_size", 1)
        examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets initialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size
        n_itr = self.get_n_itr()
        self.agent.to_device(self.affinity.get("cuda_idx", None))
        if world_size > 1:
            self.agent.data_parallel()
        self.algo.initialize(
            agent=self.agent,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples,
            world_size=world_size,
            rank=rank,
        )
        self.initialize_logging()
        return n_itr

    def _log_infos(self, traj_infos=None):
        """
        Writes trajectory info and optimizer info into csv via the logger.
        Resets stored optimizer info.
        """
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    values = [info[k] for info in traj_infos]
                    logger.record_tabular_misc_stat(k,
                                                    values)

                    wandb.run.summary[k] = np.average(values)
                    self.wandb_info[k + "Average"] = np.average(values)
                    self.wandb_info[k + "Std"] = np.std(values)
                    self.wandb_info[k + "Min"] = np.min(values)
                    self.wandb_info[k + "Max"] = np.max(values)
                    self.wandb_info[k + "Median"] = np.median(values)
                    if k == 'GameScore':
                        game = self.sampler.env_kwargs['game']
                        random_score = atari_random_scores[game]
                        der_score = atari_der_scores[game]
                        nature_score = atari_nature_scores[game]
                        human_score = atari_human_scores[game]
                        normalized_score = (np.average(values) - random_score) / (human_score - random_score)
                        der_normalized_score = (np.average(values) - random_score) / (der_score - random_score)
                        nature_normalized_score = (np.average(values) - random_score) / (nature_score - random_score)
                        self.wandb_info[k + "Normalized"] = normalized_score
                        self.wandb_info[k + "DERNormalized"] = der_normalized_score
                        self.wandb_info[k + "NatureNormalized"] = nature_normalized_score

                        maybe_update_summary(k+"Best", np.average(values))
                        maybe_update_summary(k+"NormalizedBest", normalized_score)
                        maybe_update_summary(k+"DERNormalizedBest", der_normalized_score)
                        maybe_update_summary(k+"NatureNormalizedBest", nature_normalized_score)

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
                self.wandb_info[k] = np.average(v)
                wandb.run.summary[k] = np.average(v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)

    def evaluate_agent(self, itr):
        """
        Record offline evaluation of agent performance, by ``sampler.evaluate_agent()``.
        """
        if itr > 0:
            self.pbar.stop()

        if self.final_eval_only:
            eval = itr == 0 or itr >= self.n_itr - 1
        else:
            eval = itr == 0 or itr >= self.min_itr_learn - 1
        if eval:
            logger.log("Evaluating agent...")
            self.agent.eval_mode(itr)  # Might be agent in sampler.
            eval_time = -time.time()
            traj_infos = self.sampler.evaluate_agent(itr)
            eval_time += time.time()
        else:
            traj_infos = []
            eval_time = 0.0
        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time

    def train(self):
        """
        Performs startup, evaluates the initial agent, then loops by
        alternating between ``sampler.obtain_samples()`` and
        ``algo.optimize_agent()``.  Pauses to evaluate the agent at the
        specified log interval.
        """
        n_itr = self.startup()
        self.n_itr = n_itr
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
        self.shutdown()


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
            while b < len(live_envs):  # don't want to do a for loop since live envs changes over time
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


class SerialSampler(BaseSampler):
    """The simplest sampler; no parallelism, everything occurs in same, master
    Python process.  This can be easier for debugging (e.g. can use
    ``breakpoint()`` in master process) and might be fast enough for
    experiment purposes.  Should be used with collectors which generate the
    agent's actions internally, i.e. CPU-based collectors but not GPU-based
    ones.
    NOTE: We modify this class from rlpyt to pass an id to EnvCls when creating
    environments.
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
