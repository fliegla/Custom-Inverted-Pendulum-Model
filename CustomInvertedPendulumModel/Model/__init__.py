from gym.envs.registration import register

register(
  id='DiffDrive-v0',
  entry_point='Model.envs:InvertedPendulumEnv',
  max_episode_steps=1000,
  reward_threshold=950.0,
)
