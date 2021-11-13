import numpy as np
import gym


#env = gym.make('FetchReach-v1')
#env = gym.make('FetchTraj-v0')

env = gym.make('FetchTrajReach-v1')

env.reset()
for _ in range(10):
    env.render()
    env.step(env.action_space.sample())

env.close()