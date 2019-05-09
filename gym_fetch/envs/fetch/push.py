import os
from gym import utils
from gym_fetch.envs import fetch_env



# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'push.xml')


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 terminate_success=True,
                 terminate_fail=False):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, .4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, terminate_success=terminate_success,
            terminate_fail=terminate_fail)
        utils.EzPickle.__init__(self)

    def _check_done(self, obs):
        if obs['achieved_goal'][0] < 1.02 or obs['achieved_goal'][0] > 1.58:
            done = True
        elif obs['achieved_goal'][1] < 0.37 or obs['achieved_goal'][1] > 1.125:
            done = True
        else:
            done = False
        return done
