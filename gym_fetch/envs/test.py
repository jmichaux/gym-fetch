import numpy as np
import gym

import mujoco_py

#env = gym.make('FetchReach-v1')
env = gym.make('FetchReachSparse-v3')

env.reset()
for _ in range(1):
    env.render()
    env.step(env.action_space.sample())

env.close()