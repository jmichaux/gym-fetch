"""
Adapted from OpenAI Baselines
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""
import os
import copy
from collections import deque

import numpy as np
import gym
from gym.spaces.box import Box
import torch

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _thunk

def make_fetch_env(env_id, num_processes, seed, log_dir, allow_early_resets, device=None):
    envs = [make_env(env_id, seed, rank, log_dir, allow_early_resets) for rank in range(num_processes)]
    if num_processes == 1:
        envs = DummyVecEnv(envs)
    else:
        envs = ShmemVecEnv(envs)
    envs = VecEnvPyTorch(envs, device)
    return VecMonitor(envs, max_history=50)


class VecEnvPyTorch(VecEnvWrapper):
    def __init__(self, venv, device=None):
        self.venv = venv
        if device is None or device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device=torch.device('cuda')
        observation_space = self.venv.observation_space
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = torch.from_numpy(obs[k]).float().to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def close(self):
        self.venv.close()


class VecMonitor(VecEnvWrapper):
    """
    https://github.com/cbschaff/pytorch-dl/blob/master/dl/util/envs.py
    """
    def __init__(self, venv, max_history=1000, tstart=0, tbX=False):
        super().__init__(venv)
        self.t = tstart
        self.enable_tbX = tbX
        self.episode_rewards = deque(maxlen=max_history)
        self.episode_lengths = deque(maxlen=max_history)
        self.rews = np.zeros(self.num_envs, dtype=np.float32)
        self.lens = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        obs = self.venv.reset()
        self.t += sum(self.lens)
        self.rews = np.zeros(self.num_envs, dtype=np.float32)
        self.lens = np.zeros(self.num_envs, dtype=np.int32)
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.rews += rews
        self.lens += 1
        for i,done in enumerate(dones):
            if done:
                self.episode_lengths.append(self.lens[i])
                self.episode_rewards.append(self.rews[i])
                self.t += self.lens[i]
                if self.enable_tbX and logger.get_summary_writer():
                    logger.add_scalar('env/episode_length', self.lens[i], self.t, time.time())
                    logger.add_scalar('env/episode_reward', self.rews[i], self.t, time.time())
                self.lens[i] = 0
                self.rews[i] = 0.
        return obs, rews, dones, infos
