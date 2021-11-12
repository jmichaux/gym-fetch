import numpy as np
import gym

import mujoco_py

env = gym.make('FetchReach-v1')
env.reset()
for _ in range(30):
    env.render()
    env.step(env.action_space.sample())

env.close()