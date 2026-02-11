from quadruped_ctrl.datatypes import QuadrupedState
import numpy as np
from typing import Optional


class WBInterface:
    """Whole-Body 控制接口基类
    
    配置参数来源：
    - swing_kp, swing_kd: 从环境的 robot_config (加载自 robot/*.yaml)
    - use_feedback_linearization, use_friction_compensation: 从环境的 sim_config (加载自 sim_config.yaml)
    """
    def __init__(self, env):
        self.env = env
        
        # 从环境的 robot 获取摆动腿 PD 参数
        # 这些参数来自 quadruped_env 加载的 robot_config 配置
        self.swing_kp = env.robot.swing_kp
        self.swing_kd = env.robot.swing_kd
        
        optimize = env.sim_config.get('optimize', {})
        self.use_feedback_linearization = optimize.get('use_feedback_linearization', False)
        self.use_friction_compensation = optimize.get('use_friction_compensation', False)
        
    def compute_tau(self, state: QuadrupedState,
                    swing_targets: Optional[dict] = None,
                    contact_sequence: Optional[np.ndarray] = None,
                    optimal_GRF: Optional[np.ndarray] = None) -> np.ndarray:
        """计算控制力矩
        
        Args:
            state: 机器人状态
            swing_targets: 摆动腿目标 {'FL': {'pos': [3,], 'vel': [3,], 'acc': [3,]}, ...}
            contact_sequence: 计划的接触序列 (4,) 或 (4, H)，用于覆盖测量接触
            optimal_GRF: MPC 输出的 GRF (12,)，用于支撑腿力矩计算
            
        Returns:
            控制力矩 (12,) - [FL(3), FR(3), RL(3), RR(3)]
        """
        tau_total = np.zeros(12)
        
        for leg_idx, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
            leg = getattr(state, leg_name) 

            if contact_sequence is not None:
                planned_contact = contact_sequence[leg_idx]
                if isinstance(planned_contact, (np.ndarray, list)):
                    planned_contact = planned_contact[0]
                is_stance = bool(planned_contact)
            else:
                is_stance = bool(leg.contact_state)

            if is_stance:
                # ========== 支撑腿力矩 ==========
                q_idx = leg.qvel_idxs 
                J_leg = leg.jac_pos_world[:, q_idx]
                if optimal_GRF is not None:
                    force = optimal_GRF[leg_idx * 3:(leg_idx + 1) * 3]
                else:
                    force = leg.contact_force
                tau = -J_leg.T @ force
                tau_total[leg_idx*3:(leg_idx+1)*3] = tau
                
            else:
                # ========== 摆动腿力矩 ==========
                target = swing_targets.get(leg_name) if swing_targets else None
                tau_swing = self.compute_swing_leg_tau(leg, target)
                tau_total[leg_idx*3:(leg_idx+1)*3] = tau_swing
        
        return tau_total
    
    def compute_swing_leg_tau(self, leg, target: Optional[dict] = None) -> np.ndarray:
        """计算摆动腿力矩：通过任务空间 PD 映射到关节空间
        """
        # 1. 统一解析目标状态 (若 target 为 None 则维持现状)
        # target: {'pos': [3,], 'vel': [3,], 'acc': [3,]}  
        target = target or {}
        d_pos = target.get('pos', leg.foot_pos)
        d_vel = target.get('vel', np.zeros(3))
        d_acc = target.get('acc', np.zeros(3))

        # 2. 计算 PD 虚拟加速度/力 (Des_acc + Kp*e + Kd*edot)
        # 这既是基础笛卡尔空间力，也是反馈线性化的核心项
        q_idx = leg.qvel_idxs
        _J = leg.jac_pos_base[:, q_idx]
        _J_dot = leg.jac_dot_base[:, q_idx]

        acc = d_acc + self.swing_kp * (d_pos - leg.foot_pos) + self.swing_kd * (d_vel - leg.foot_vel)
        acc = acc.reshape((3, ))
        tau_swing = _J.T @ (self.swing_kp * (d_pos - leg.foot_pos) + self.swing_kd * (d_vel - leg.foot_vel) )

        if self.use_feedback_linearization:
            tau_swing = leg.mass_matrix @ np.linalg.inv(_J) @ (acc - _J_dot @ leg.qvel) + leg.qfrc_bias
        if self.use_friction_compensation:
            pass
        return tau_swing
