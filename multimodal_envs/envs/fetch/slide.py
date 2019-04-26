import os
import numpy as np

from gym import utils
from multimodal_envs.envs import fetch_env



# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'slide.xml')


class FetchSlideEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 terminate_success=False,
                 terminate_fail=False):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.425, .707, 0., 0., .707],
        }
        [1.7, 1.1, 0.425, .707, 0., 0., .707]
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, terminate_success=terminate_success,
            terminate_fail=terminate_fail)
        utils.EzPickle.__init__(self)

    def _check_done(self, obs):
        if obs['achieved_goal'][0] < 0.66 or obs['achieved_goal'][0] > 1.98:
            done = True
        elif obs['achieved_goal'][1] < 0.24 or obs['achieved_goal'][1] > 1.25:
            done = True
        else:
            done = False
        return done
