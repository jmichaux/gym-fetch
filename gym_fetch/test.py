import numpy as np
import gym

env = gym.make('gym_fetch:FetchTrajReach-v1')

env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())

env.close()