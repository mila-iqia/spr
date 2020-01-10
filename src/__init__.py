from gym.envs.registration import register

register(
    id='atari-v0',
    entry_point='src.envs:AtariEnv',
)