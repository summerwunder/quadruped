import numpy as np
from quadruped_ctrl.controllers.controller_base import BaseController
from quadruped_ctrl.datatypes import QuadrupedState

class PDController(BaseController):
    def __init__(self, env, kp=60.0, kd=3.5, **kwargs):
        super().__init__(env, **kwargs)
        
        self.kp = kp
        self.kd = kd
        
        # 默认目标姿态
        if not hasattr(self, 'default_target_q'):
            self.default_target_q = np.array([0, 0.9, -1.8] * 4)

    def get_action(self, state: QuadrupedState, target_q) -> np.ndarray:
        """
        按照基类接口实现，所有输入都通过 kwargs 传递
        """
        
        full_tau = np.zeros(self.env.model.nu)
        
        # 3. 计算四条腿的 PD
        leg_names = ['FL', 'FR', 'RL', 'RR']
        for i, name in enumerate(leg_names):
            leg = state[name] # 访问 QuadrupedState 里的单腿数据
            
            # 目标位置切片
            q_des = target_q[i*3 : (i+1)*3]
            
            # 核心算法：tau = Kp * (q_des - q) - Kd * q_dot
            tau_leg = self.kp * (q_des - leg.qpos) - self.kd * leg.qvel

            # 填充到总力矩向量中
            # leg.tau_idxs 对应这只腿在总向量里的索引（通常是 0-2, 3-5...）
            full_tau[leg.tau_idxs] = tau_leg
            
        return full_tau