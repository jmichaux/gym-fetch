import mujoco_py
import gym
import numpy as np

env = gym.make("gym_fetch:FetchTraj-v0")
env.reset()
for i in range(1000):
    env.step(action=np.array([i * 0.1] * 7))
    env.render()
env.close()
