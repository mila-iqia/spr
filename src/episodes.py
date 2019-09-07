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
        transitions.append(Transition(timestep, state, action, reward, not done))
        state = next_state
        timestep = 0 if done else timestep + 1

    env.close()
    return transitions


def sample_state(real_transitions, encoder):
    idx = np.random.randint(0, len(real_transitions))
    transition = get_framestacked_transition(idx, real_transitions)
    # trans_deque = deque(maxlen=4)
    # for trans in transition:
    #     trans_deque.append(trans)
    with torch.no_grad():
        z = encoder(torch.stack([trans.state.float() / 255. for trans in transition]))
    state_deque = deque(maxlen=4)
    for s in z.unbind():
        state_deque.append(s)
    return state_deque


class LatentState():
    def __init__(self, latents=None):
        if latents is not None:
            self.latents = deque(maxlen=4)
        else:
            self.latents = latents

    def append(self, latent):
        self.latents.append(latent)


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