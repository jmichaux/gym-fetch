import numpy as np
import gym_fetch

import mujoco_py

#env = gym.make('FetchReach-v1')
env = gym_fetch.make('FetchReachSparse-v3')

env.reset()
for _ in range(1):
    env.render()
    env.step(env.action_space.sample())

env.close()