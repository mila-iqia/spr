import torch
import numpy as np
from collections import deque, namedtuple

from src.memory import blank_trans
from src.envs import Env

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))


def get_random_agent_episodes(args):
    env = Env(args)
    env.train()
    action_space = env.action_space()
    print('-------Collecting samples----------')
    transitions = []
    timestep, done = 0, True
    for T in range(args.initial_exp_steps):
        if done:
            state, done = env.reset(), False
        state = state[-1].mul(255).to(dtype=torch.uint8,
                                      device=torch.device('cpu'))  # Only store last frame and discretise to save memory
        action = np.random.randint(0, action_space)
        next_state, reward, done = env.step(action)
        if args.reward_clip > 0:
            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
        transitions.append(Transition(timestep, state, action, reward, not done))
        state = next_state
        timestep = 0 if done else timestep + 1

    env.close()
    return transitions


def sample_real_transitions(real_transitions, num_samples):
    samples = []
    while len(samples) < num_samples:
        idx = np.random.randint(0, len(real_transitions))
        samples.append(
            torch.stack([trans.state.float() / 255. for trans in get_framestacked_transition(idx, real_transitions)]))
    return torch.stack(samples)


def get_framestacked_transition(idx, transitions):
    history = 4
    transition = np.array([None] * history)
    transition[history - 1] = transitions[idx]
    for t in range(4 - 2, -1, -1):  # e.g. 2 1 0
        if transition[t + 1].timestep == 0:
            transition[t] = blank_trans  # If future frame has timestep 0
        else:
            transition[t] = transitions[idx - history + 1 + t]
    return transition
