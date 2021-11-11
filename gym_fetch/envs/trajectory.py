import numpy as np
#try:
#    import mujoco_py
#except ImportError as e:
#    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def gen_traj(ka,q_0,q_dot_0,T_len=1000):

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

def step_traj(self,ka):
    old_state = self.sim.get_state()
    traj_qpos, traj_qvel, T_plan = gen_traj(ka,old_state.qpos,old_state.qvel)

    for i in range(traj_qpos.shape[1]-1):
        # TO DO: action demension? and qpos demension?
        #  figure out where rlsimulation is initiated?
        
        new_qpos = np.copy(old_state.qpos)
        new_qvel = np.copy(old_state.qvel)
        new_qpos[7:-1] = traj_qpos[:,i+1]
        new_qvel[7:-1] = traj_qvel[:,i+1]
        t = T_plan[i+1]
        
        new_state = mujoco_py.MjSimState(
            old_state.time+t, new_qpos, new_qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()
        if i == traj_qpos.shape[1]-2:
            break
        self.render() # might be wrong?


if __name__ == '__main__':
    a = np.arange(15)
    b = np.arange(7)
    c= np.append(a[:7],b)
    print(a[7:-1])