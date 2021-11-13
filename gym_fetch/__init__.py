from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
#    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
#_load_env_plugins()


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


