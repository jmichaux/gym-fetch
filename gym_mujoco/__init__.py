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

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    # Fetch
    register(
        id="FetchTrajReach{}-v1".format(suffix),
        entry_point="gym_fetch.envs:FetchTrajReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )


