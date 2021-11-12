from gym.envs.registration import register

# register trajectory Fetch
register(id='FetchTraj-v0',
    entry_point='gym_fetch.envs.fetch:FetchTrajReachEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
    )


# Standard Environments
for reward_type in ['dense', 'sparse', 'very_sparse']:
    if reward_type == 'dense':
        suffix = 'Dense'
    elif reward_type == 'sparse':
        suffix = 'Sparse'
    else:
        suffix = 'VerySparse'

    for i, terminate_condition in enumerate([[False, False], [True, False], [False, True], [True, True]]):
        kwargs = {
            'reward_type': reward_type,
            'terminate_success': terminate_condition[0],
            'terminate_fail': terminate_condition[1],
            }

        # Fetch
        register(
            id='FetchSlide{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch.envs:FetchSlideEnv',
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id='FetchPickAndPlace{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch.envs:FetchPickAndPlaceEnv',
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id='FetchReach{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch.envs:FetchReachEnv',
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id='FetchPush{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch.envs:FetchPushEnv',
            kwargs=kwargs,
            max_episode_steps=50,
        )

        register(
            id='FetchHook{}-v{}'.format(suffix, i + 2),
            entry_point='gym_fetch.envs:FetchHookEnv',
            kwargs=kwargs,
            max_episode_steps=100,
        )
