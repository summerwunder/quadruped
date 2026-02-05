import numpy as np
from quadruped_ctrl.controllers.controller_base import BaseController
from quadruped_ctrl.datatypes import QuadrupedState

class PDController(BaseController):
    def __init__(self, model, kp=60.0, kd=3.5):
        super().__init__(model)
        self.kp = kp
        self.kd = kd
        
        # Home position
        self.default_target_q = np.array([0, 0.9, -1.8] * 4) 

    def get_action(self, state: QuadrupedState, target_q: np.ndarray = None) -> np.ndarray:
        """ 
        target_q: 12维目标关节角
        """
        full_tau = np.zeros(self.model.nu)
        current_target_q = target_q if target_q is not None else self.default_target_q
        
        leg_names = ['FL', 'FR', 'RL', 'RR']
        for i, name in enumerate(leg_names):
            leg = state[name]
            
            q_des = current_target_q[i*3 : (i+1)*3]
            
            # tau = Kp(q_des - q) - Kd(v)
            tau_leg = self.kp * (q_des - leg.qpos) - self.kd * leg.qvel

            full_tau[leg.tau_idxs] = tau_leg
            
        return full_tau