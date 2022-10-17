from gym.envs.registration import register

register(
  id="InvertedPendulum-v6",
  entry_point="Model.envs.inverted_pendulum:InvertedPendulumEnv",
  max_episode_steps=1000,
  reward_threshold=950.0,
)
