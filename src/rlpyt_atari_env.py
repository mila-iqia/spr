"""
Modifies the default rlpyt AtariEnv to be closer to DeepMind's setup,
tries to follow Kaixin/Rainbow's env for the most part.
"""
import numpy as np
import os
import atari_py
import cv2
from collections import namedtuple
from gym.utils import seeding

from rlpyt.envs.base import Env, EnvStep
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo


EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])


class AtariTrajInfo(TrajInfo):
    """TrajInfo class for use with Atari Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.GameScore = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.GameScore += getattr(env_info, "game_score", 0)


class AtariEnv(Env):
    """An efficient implementation of the classic Atari RL envrionment using the
    Arcade Learning Environment (ALE).

    Output `env_info` includes:
        * `game_score`: raw game score, separate from reward clipping.
        * `traj_done`: special signal which signals game-over or timeout, so that sampler doesn't reset the environment when ``done==True`` but ``traj_done==False``, which can happen when ``episodic_lives==True``.

    Always performs 2-frame max to avoid flickering (this is pretty fast).

    Screen size downsampling is done by cropping two rows and then
    downsampling by 2x using `cv2`: (210, 160) --> (80, 104).  Downsampling by
    2x is much faster than the old scheme to (84, 84), and the (80, 104) shape
    is fairly convenient for convolution filter parameters which don't cut off
    edges.

    The action space is an `IntBox` for the number of actions.  The observation
    space is an `IntBox` with ``dtype=uint8`` to save memory; conversion to float
    should happen inside the agent's model's ``forward()`` method.

    (See the file for implementation details.)


    Args:
        game (str): game name
        frame_skip (int): frames per step (>=1)
        num_img_obs (int): number of frames in observation (>=1)
        clip_reward (bool): if ``True``, clip reward to np.sign(reward)
        episodic_lives (bool): if ``True``, output ``done=True`` but ``env_info[traj_done]=False`` when a life is lost
        max_start_noops (int): upper limit for random number of noop actions after reset
        repeat_action_probability (0-1): probability for sticky actions
        horizon (int): max number of steps before timeout / ``traj_done=True``
    """

    def __init__(self,
                 game="pong",
                 frame_skip=4,  # Frames per step (>=1).
                 num_img_obs=4,  # Number of (past) frames in observation (>=1).
                 clip_reward=True,
                 episodic_lives=True,
                 max_start_noops=30,
                 repeat_action_probability=0.,
                 horizon=27000,
                 stack_actions=0,
                 grayscale=True,
                 imagesize=84,
                 seed=42,
                 id=0,
                 ):
        save__init__args(locals(), underscore=True)
        # ALE
        game_path = atari_py.get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError("You asked for game {} but path {} does not "
                " exist".format(game, game_path))
        self.ale = atari_py.ALEInterface()
        self.seed(seed, id)
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.loadROM(game_path)

        # Spaces
        self.stack_actions = stack_actions
        self._action_set = self.ale.getMinimalActionSet()
        self._action_space = IntBox(low=0, high=len(self._action_set))
        self.channels = 1 if grayscale else 3
        self.grayscale = grayscale
        self.imagesize = imagesize
        if self.stack_actions: self.channels += 1
        obs_shape = (num_img_obs, self.channels, imagesize, imagesize)
        self._observation_space = IntBox(low=0, high=255, shape=obs_shape,
            dtype="uint8")
        self._max_frame = self.ale.getScreenGrayscale() if self.grayscale \
            else self.ale.getScreenRGB()
        self._raw_frame_1 = self._max_frame.copy()
        self._raw_frame_2 = self._max_frame.copy()
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")

        # Settings
        self._has_fire = "FIRE" in self.get_action_meanings()
        self._has_up = "UP" in self.get_action_meanings()
        self._horizon = int(horizon)
        self.reset()

    def seed(self, seed=None, id=0):
        _, seed1 = seeding.np_random(seed)
        if id > 0:
            seed = seed*100 + id
        self.np_random, _ = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)

    def reset(self):
        """Performs hard reset of ALE game."""
        self.ale.reset_game()
        self._reset_obs()
        self._life_reset()
        if self._max_start_noops > 0:
            for _ in range(self.np_random.randint(1, self._max_start_noops + 1)):
                self.ale.act(0)
                if self._check_life():
                    self.reset()
        self._update_obs(0)  # (don't bother to populate any frame history)
        self._step_counter = 0
        return self.get_obs()

    def step(self, action):
        a = self._action_set[action]
        game_score = np.array(0., dtype="float32")
        for _ in range(self._frame_skip - 1):
            game_score += self.ale.act(a)
        self._get_screen(1)
        game_score += self.ale.act(a)
        lost_life = self._check_life()  # Advances from lost_life state.
        if lost_life and self._episodic_lives:
            self._reset_obs()  # Internal reset.
        self._update_obs(action)
        reward = np.sign(game_score) if self._clip_reward else game_score
        game_over = self.ale.game_over() or self._step_counter >= self.horizon
        done = game_over or (self._episodic_lives and lost_life)
        info = EnvInfo(game_score=game_score, traj_done=game_over)
        self._step_counter += 1
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10, show_full_obs=False):
        """Shows game screen via cv2, with option to show all frames in observation."""
        img = self.get_obs()
        if show_full_obs:
            shape = img.shape
            img = img.reshape(shape[0] * shape[1], shape[2])
        else:
            img = img[-1]
        cv2.imshow(self._game, img)
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    ###########################################################################
    # Helpers

    def _get_screen(self, frame=1):
        frame = self._raw_frame_1 if frame == 1 else self._raw_frame_2
        if self.grayscale:
            self.ale.getScreenGrayscale(frame)
        else:
            self.ale.getScreenRGB(frame)

    def _update_obs(self, action):
        """Max of last two frames; crop two rows; downsample by 2x."""
        self._get_screen(2)
        np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
        img = cv2.resize(self._max_frame, (self.imagesize, self.imagesize), cv2.INTER_LINEAR)
        if len(img.shape) == 2:
            img = img[np.newaxis]
        else:
            img = np.transpose(img, (2, 0, 1))
        if self.stack_actions:
            action = int(255.*action/self._action_space.n)
            action = np.ones_like(img[:1])*action
            img = np.concatenate([img, action], 0)
        # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
        self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])

    def _reset_obs(self):
        self._obs[:] = 0
        self._max_frame[:] = 0
        self._raw_frame_1[:] = 0
        self._raw_frame_2[:] = 0

    def _check_life(self):
        lives = self.ale.lives()
        lost_life = (lives < self._lives) and (lives > 0)
        if lost_life:
            self._life_reset()
        return lost_life

    def _life_reset(self):
        self.ale.act(0)
        self._lives = self.ale.lives()

    ###########################################################################
    # Properties

    @property
    def game(self):
        return self._game

    @property
    def frame_skip(self):
        return self._frame_skip

    @property
    def num_img_obs(self):
        return self._num_img_obs

    @property
    def clip_reward(self):
        return self._clip_reward

    @property
    def max_start_noops(self):
        return self._max_start_noops

    @property
    def episodic_lives(self):
        return self._episodic_lives

    @property
    def repeat_action_probability(self):
        return self._repeat_action_probability

    @property
    def horizon(self):
        return self._horizon

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

ACTION_INDEX = {v: k for k, v in ACTION_MEANING.items()}
