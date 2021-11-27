from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
#    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
#_load_env_plugins()



register(
    id="ReacherTraj-v2",
    entry_point="gym_mujoco.envs:ReacherTrajEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)
register(
    id="ReacherTrajObs-v2",
    entry_point="gym_mujoco.envs:ReacherTrajObsEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

