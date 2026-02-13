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
                tau = - J_leg.T @ force
                tau_total[leg_idx*3:(leg_idx+1)*3] = tau
                
            else:
                # ========== 摆动腿力矩 ==========
                target = swing_targets.get(leg_name) 
                tau_swing = self.compute_swing_leg_tau(leg, target)
                tau_total[leg_idx*3:(leg_idx+1)*3] = tau_swing
        
        return tau_total
    
    def compute_swing_leg_tau(self, leg, target: Optional[dict] = None) -> np.ndarray:
        """计算摆动腿力矩：通过任务空间 PD 映射到关节空间
        """
        # target: {'pos': [3,], 'vel': [3,], 'acc': [3,]}  
        target = target or {}
        d_pos = target.get('pos', leg.foot_pos)
        d_vel = target.get('vel', np.zeros(3))
        d_acc = target.get('acc', np.zeros(3))

        q_idx = leg.qvel_idxs
        J = leg.jac_pos_base[:, q_idx]
        J_dot = leg.jac_dot_base[:, q_idx]
        
        # 1. 计算笛卡尔空间目标加速度 (PD)
        # 这里的 Kp/Kd 需要针对动态摆动调优，通常比站立时要小一点，防止撞地
        error_pos = d_pos - leg.foot_pos
        error_vel = d_vel - leg.foot_vel
        v_acc = d_acc + self.swing_kp * error_pos + self.swing_kd * error_vel
        # 3. 动力学补偿
        if self.use_feedback_linearization:
            J_inv = np.linalg.pinv(J)
            qdd_ref = J_inv @ (v_acc - J_dot @ leg.qvel)
            # M * qdd + b
            # 确保 leg.mass_matrix_leg 是该腿对应的 3x3 惯量矩阵
            tau_swing = leg.mass_matrix_leg @ qdd_ref + leg.qfrc_bias_leg
        else:
            # 基础雅可比转置映射 (Virtual Force)
            force = self.swing_kp * error_pos + self.swing_kd * error_vel
            tau_swing = J.T @ force
        return tau_swing
