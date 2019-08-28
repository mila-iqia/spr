import cv2
import torch
from baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym import spaces

from a2c_ppo_acktr.envs import TimeLimitMask, MaskGoal, VecPyTorch, VecNormalize, \
    VecPyTorchFrameStack
from pathlib import Path
import os
import gym
import numpy as np
import torch
from baselines import bench
from baselines.common.atari_wrappers import make_atari, EpisodicLifeEnv, FireResetEnv, WarpFrame, ScaledFloatFrame, \
    ClipRewardEnv, FrameStack
from gym.spaces import Box


def make_env(env_id, seed=42, rank=0, log_dir=None, downsample=True, color=False, wrap_pytorch=True):
    def _thunk():
        env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
            # env = AARIWrapper(env)

        env.seed(seed + rank)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=False)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env, downsample=downsample, frame_stack=True, color=color)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3, 4]:
            env = TransposeImage(env, op=[2, 0, 1])

        if wrap_pytorch:
            env = WrapPytorch(env)
        return env

    return _thunk


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return np.array(ob).transpose(self.op[0], self.op[1], self.op[2])


class WrapPytorch(gym.Wrapper):
    def __init__(self, env, device=torch.device('cpu')):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.env.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step(self, action):
        action = action.cpu().numpy()
        obs, reward, done, info = self.env.step(action)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info


def make_vec_envs(args, num_processes, log_dir='./tmp/', device=torch.device('cpu')):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    envs = [make_env(args.env_name, args.seed, i, log_dir, not args.no_downsample, args.color)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecPyTorch(envs, device)

    if args.num_frame_stack > 1:
        envs = VecPyTorchFrameStack(envs, args.num_frame_stack, device)

    return envs


class GrayscaleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.observation_space.shape[0], self.observation_space.shape[1], 1),
                                            dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        return frame


def wrap_deepmind(env, downsample=True, episode_life=True, clip_rewards=True, frame_stack=False, scale=False,
                  color=False):
    """Configure environment for DeepMind-style Atari.
    """
    if ("videopinball" in str(env.spec.id).lower()) or ('tennis' in str(env.spec.id).lower()):
        env = WarpFrame(env, width=160, height=210, grayscale=False)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if downsample:
        env = WarpFrame(env, grayscale=False)
    if not color:
        env = GrayscaleWrapper(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


# -*- coding: utf-8 -*-
from collections import deque
import random
import atari_py
import cv2


class Env():
  def __init__(self, args):
    self.device = args.device
    self.ale = atari_py.ALEInterface()
    self.ale.setInt('random_seed', args.seed)
    self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
    self.ale.setFloat('repeat_action_probability', 0.25)  # Disable sticky actions
    self.ale.setInt('frame_skip', 0)
    self.ale.setBool('color_averaging', False)
    self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
    actions = self.ale.getMinimalActionSet()
    self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
    self.lives = 0  # Life counter (used in DeepMind training)
    self.life_termination = False  # Used to check if resetting only from loss of life
    self.window = args.history_length  # Number of frames to concatenate
    self.state_buffer = deque([], maxlen=args.history_length)
    self.training = True  # Consistent with model training mode

  def _get_state(self):
    state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

  def _reset_buffer(self):
    for _ in range(self.window):
      self.state_buffer.append(torch.zeros(84, 84, device=self.device))

  def reset(self):
    if self.life_termination:
      self.life_termination = False  # Reset flag
      self.ale.act(0)  # Use a no-op after loss of life
    else:
      # Reset internals
      self._reset_buffer()
      self.ale.reset_game()
      # Perform up to 30 random no-ops before starting
      for _ in range(random.randrange(30)):
        self.ale.act(0)  # Assumes raw action 0 is always no-op
        if self.ale.game_over():
          self.ale.reset_game()
    # Process and return "initial" state
    observation = self._get_state()
    self.state_buffer.append(observation)
    self.lives = self.ale.lives()
    return torch.stack(list(self.state_buffer), 0)

  def step(self, action):
    # Repeat action 4 times, max pool over last 2 frames
    frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    reward, done = 0, False
    for t in range(4):
      reward += self.ale.act(self.actions.get(action))
      if t == 2:
        frame_buffer[0] = self._get_state()
      elif t == 3:
        frame_buffer[1] = self._get_state()
      done = self.ale.game_over()
      if done:
        break
    observation = frame_buffer.max(0)[0]
    self.state_buffer.append(observation)
    # Detect loss of life as terminal in training mode
    if self.training:
      lives = self.ale.lives()
      if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
        self.life_termination = not done  # Only set flag when not truly done
        done = True
      self.lives = lives
    # Return state, reward, done
    return torch.stack(list(self.state_buffer), 0), reward, done

  # Uses loss of life as terminal signal
  def train(self):
    self.training = True

  # Uses standard terminal signal
  def eval(self):
    self.training = False

  def action_space(self):
    return len(self.actions)

  def render(self):
    cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
    cv2.waitKey(1)

  def close(self):
    cv2.destroyAllWindows()