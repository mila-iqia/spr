from collections import deque
import torch.multiprocessing as mp

from rlpyt.utils.synchronize import find_port
import traceback

from src.mcts_memory import LocalBuffer, initialize_replay_buffer, samples_to_buffer
from src.model_trainer import TrainingWorker
from src.async_reanalyze import AsyncReanalyze
from src.utils import get_args, DillWrapper
from src.logging import log_results

import os
import time
import copy
import torch
import wandb
import numpy as np
import gym

from src.vectorized_mcts import VectorizedMCTS, AsyncEval


def torch_set_device(args):
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = True
    else:
        args.device = torch.device('cpu')


def run_pizero(args):
    buffer = initialize_replay_buffer(args)
    ctx = mp.get_context("fork")
    error_queue = ctx.Queue()
    send_queue = ctx.Queue()
    receive_queues = []
    port = find_port(args.seed)
    for i in range(args.num_trainers):
        receive_queue = ctx.Queue()
        worker = TrainingWorker(i,
                                args.num_trainers,
                                args,
                                port,
                                send_queue,
                                error_queue,
                                receive_queue)
        process = ctx.Process(target=worker.optimize,
                              args=(DillWrapper(buffer),))
        process.start()
        receive_queues.append(receive_queue)

    # Need to get the target network from the training agent that created it.
    torch_set_device(args)
    network = send_queue.get()
    print("Received target network from trainer")
    target_network = copy.deepcopy(network)
    args.device = torch.device('cuda:{}'.format(0))
    target_network.to(args.device)
    target_network.share_memory()

    if args.reanalyze:
        async_reanalyze = AsyncReanalyze(args, target_network, debug=args.debug_reanalyze)

    local_buf = LocalBuffer()
    env_steps = 0
    eprets = np.zeros(args.num_envs, 'f')
    episode_rewards = deque(maxlen=10)
    wandb.log({'env_steps': 0})

    env = gym.vector.make('atari-v0', num_envs=args.num_envs, args=args,
                          asynchronous=not args.sync_envs)

    # TODO return int observations
    obs = env.reset()
    obs = torch.from_numpy(obs)
    vectorized_mcts = VectorizedMCTS(args, env.action_space[0].n, args.num_envs,
                                     args.num_simulations,
                                     target_network)
    eval_vectorized_mcts = VectorizedMCTS(args,
                                          env.action_space[0].n,
                                          args.evaluation_episodes,
                                          args.eval_simulations,
                                          target_network,

                                          eval=True)
    async_eval = AsyncEval(eval_vectorized_mcts)
    total_episodes = 0
    total_train_steps, target_train_steps = 0, 0
    training_started = False
    try:
        while env_steps < args.total_env_steps:
            # Run MCTS for the vectorized observation
            actions, policies, values = vectorized_mcts.run(obs)
            policy_probs = policies.probs
            next_obs, reward, done, infos = env.step(actions.cpu().numpy())
            reward, done = torch.from_numpy(reward).float(), torch.from_numpy(done).float()
            obs, actions, reward, done, policy_probs, values = obs.cpu(), actions.cpu(), reward.cpu(),\
                                                               done.cpu(), policy_probs.cpu(), values.cpu()

            eprets += np.array(reward)
            for i in range(args.num_envs):
                if done[i]:
                    episode_rewards.append(eprets[i])
                    wandb.log({'Episode Reward': eprets[i],
                               'env_steps': env_steps})
                    eprets[i] = 0

            if args.reanalyze:
                async_reanalyze.store_transitions(
                    obs[:, -1],
                    actions,
                    reward,
                    done,
                )

                total_episodes += torch.sum(done)

                # Add the reanalyzed transitions to the real data.
                new_samples = async_reanalyze.get_transitions(total_episodes)
                cat_obs = torch.cat([obs, new_samples[0]], 0)
                actions = torch.cat([actions, new_samples[1]], 0)
                reward = torch.cat([reward, new_samples[2]], 0)
                done = torch.cat([done, new_samples[3]], 0)
                policy_probs = torch.cat([policy_probs, new_samples[4]], 0)
                values = torch.cat([values, new_samples[5]], 0)
                local_buf.append(cat_obs, actions, reward, done, policy_probs, values)

            else:
                local_buf.append(obs, actions, reward, done, policy_probs, values)

            if env_steps % args.jumps == 0 and env_steps > 0:
                # Send transitions from the local buffer to the replay buffer
                buffer.append_samples(samples_to_buffer(*local_buf.stack()))
                local_buf.clear()

            force_wait = (total_train_steps *
                          args.batch_size < env_steps*args.replay_ratio_lower) and \
                         args.replay_ratio_lower > 0 and training_started

            if force_wait:
                print("Runner waiting; needs {} more train steps to continue".format(
                                      -args.batch_size
                                      * total_train_steps +
                                      args.replay_ratio_lower * env_steps))

            if force_wait or not send_queue.empty():
                steps, log = send_queue.get()
                log_results(log, steps)
                total_train_steps = steps

                if 0 < args.target_update_interval <= total_train_steps - target_train_steps:
                    print("Updated target weights at step {}".format(total_train_steps))
                    target_network.load_state_dict(network.state_dict())
                    target_train_steps = total_train_steps

            # Send a command to start training if ready
            if args.num_envs*101 >= env_steps > args.num_envs*100:
                [q.put("train") for q in receive_queues]
                training_started = True

            if args.replay_ratio_upper > 0 and training_started:
                [q.put(env_steps) for q in receive_queues]

            if env_steps % args.log_interval == 0 and len(episode_rewards) > 0:
                print('Env Steps: {}, Mean Reward: {}, Median Reward: {}'.format(env_steps, np.mean(episode_rewards),
                                                                                 np.median(episode_rewards)))
                wandb.log({'Mean Reward': np.mean(episode_rewards), 'Median Reward': np.median(episode_rewards),
                           'env_steps': env_steps})
                eval_result = async_eval.get_eval_results()
                if eval_result:
                    eval_env_step, avg_reward = eval_result
                    print('Env steps: {}, Avg_Reward: {}'.format(eval_env_step, avg_reward))
                    wandb.log({'env_steps': env_steps, 'avg_reward': avg_reward})

            if env_steps % args.evaluation_interval == 0 and env_steps >= 0:
                print("Starting evaluation run")
                async_eval.send_queue.put(('evaluate', env_steps))

            obs.copy_(torch.from_numpy(next_obs))
            env_steps += args.num_envs

    except (KeyboardInterrupt, Exception):
        traceback.print_exc()
        [q.put(None,) for q in receive_queues]
    finally:
        return


if __name__ == '__main__':
    args = get_args()
    tags = ['']
    try:
        if len(args.name) == 0:
            run = wandb.init(project=args.wandb_proj,
                             entity="abs-world-models",
                             tags=tags,
                             config=vars(args))
        else:
            run = wandb.init(project=args.wandb_proj,
                             name=args.name,
                             entity="abs-world-models",
                             tags=tags,
                             config=vars(args))
    except wandb.run_manager.LaunchError as e:
        print(e)
        pass

    if len(args.savedir) == 0:
        args.savedir = os.environ["SLURM_TMPDIR"]
    args.savedir = "{}/{}".format(args.savedir, run.id)
    os.makedirs(args.savedir, exist_ok=True)

    print("Saving episode data in {}".format(args.savedir))


    run_pizero(args)
