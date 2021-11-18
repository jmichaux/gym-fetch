import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py

class ReacherTrajEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "reacher.xml", 2)

    def _step_traj(self, ka):
        t = 0.01 #following the original timestamp of gym reacher2d
        old_state = self.sim.get_state()
        qpos = old_state.qpos[:2]
        qvel = old_state.qvel[:2]
        new_qpos = np.copy(old_state.qpos)
        new_qvel = np.copy(old_state.qvel)
        ka = np.array(ka).reshape(qpos.shape)
        new_qpos[:2] = qpos + qvel * t + (0.5 * ka * t ** 2)
        new_qvel[:2] = qvel + ka * t
        print(new_qpos)
        print(new_qvel)
        new_state = mujoco_py.MjSimState(
            old_state.time + t, new_qpos, new_qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

        '''
        old_state = self.sim.get_state()
        traj_qpos, traj_qvel, T_plan = self.gen_traj(ka,old_state.qpos[:2],old_state.qvel[:2])
        
        
        for i in range(traj_qpos.shape[1] - 1):
            # TO DO: action demension? and qpos demension?
            #  figure out where rlsimulation is initiated?

            new_qpos = np.copy(old_state.qpos)
            new_qvel = np.copy(old_state.qvel)
            new_qpos[:2] = traj_qpos[:, i + 1]
            new_qvel[:2] = traj_qvel[:, i + 1]
            t = T_plan[i + 1]

            new_state = mujoco_py.MjSimState(
                old_state.time + t, new_qpos, new_qvel, old_state.act, old_state.udd_state
            )
            self.sim.set_state(new_state)
            self.sim.forward()
            # if i == traj_qpos.shape[1]-2:
            #    break
        '''


    def gen_traj(self,ka,q_0,q_dot_0,T_len=20):

        T = np.linspace(0,1,T_len+1)
        T_plan = T[:int(T_len/2)]
        #T_brake = T[int(T_len/2):]

        q_to_peak = q_0.reshape(-1,1) + np.outer(q_dot_0,T_plan) + .5*np.outer(ka,T_plan**2)
        q_dot_to_peak = q_dot_0.reshape(-1,1) + np.outer(ka,T_plan)

        #q_peak = q_to_peak[:,-1]
        #q_dot_peak = q_dot_to_peak[:,-1]

            
        #T_brake = T_brake - T_brake[0]
        #q_to_stop = q_peak + q_dot_peak.*T_brake + (1/2)*((0 - q_dot_peak)./t_to_stop).*T_brake.^2
        #q_dot_to_stop = q_dot_peak + ((0 - q_dot_peak)./t_to_stop).*T_brake

        return q_to_peak, q_dot_to_peak, T_plan

    def step(self, a,render_flag = 0):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        #self.do_simulation(a, self.frame_skip)
        self._step_traj(a)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )