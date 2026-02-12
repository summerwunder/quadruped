import numpy as np
import casadi as cs
from quadruped_ctrl.controllers.nmpc_gradient.quadruped_model import QuadrupedModel
from quadruped_ctrl.datatypes import QuadrupedState, ReferenceState
class QuadrupedConstraints:
    def __init__(self, model:QuadrupedModel, use_static_stability: bool, acados_infty: float):
        self.model = model
        self.use_static_stability = use_static_stability
        self.ACADOS_INFTY = acados_infty

    def get_stability_bounds(self):
        base_w = self.model.state[0:3]
        base_vel_w = self.model.state[3:6]
        pos_FL = self.model.state[12:15]
        pos_FR = self.model.state[15:18]
        pos_RL = self.model.state[18:21]
        pos_RR = self.model.state[21:24]
        
        yaw = self.model.state[8]
        R_wb = cs.SX.zeros(2,2)
        R_wb[0,0] = cs.cos(yaw)
        R_wb[0,1] = cs.sin(yaw)
        R_wb[1,0] = -cs.sin(yaw)
        R_wb[1,1] = cs.cos(yaw)
        pos_FL[0:2] = R_wb @ (pos_FL[:2]- base_w[0:2])
        pos_FR[0:2] = R_wb @ (pos_FR[:2]- base_w[0:2])
        pos_RL[0:2] = R_wb @ (pos_RL[:2]- base_w[0:2])
        pos_RR[0:2] = R_wb @ (pos_RR[:2]- base_w[0:2])
        
        if self.use_static_stability:
            x = 0.0
            y = 0.0
        else:  # ZMP
            force_FL = self.model.input[12:15]
            force_FR = self.model.input[15:18]
            force_RL = self.model.input[18:21]
            force_RR = self.model.input[21:24]
            force_all = force_FL + force_FR + force_RL + force_RR + 1e-6
            gravity = np.array([0, 0, -self.model.gravity])
            linear_com_acc = (1.0 / self.model.mass) @ force_all + gravity 
            zmp = base_w[0:2] - linear_com_acc[0:2] * (base_w[2] / (-gravity[2]))
            zmp = R_wb @ (zmp - base_w[0:2])
            x = zmp[0]
            y = zmp[1]
        
        y_FL = pos_FL[1]
        y_FR = pos_FR[1]
        y_RL = pos_RL[1]
        y_RR = pos_RR[1]

        x_FL = pos_FL[0]
        x_FR = pos_FR[0]
        x_RL = pos_RL[0]
        x_RR = pos_RR[0]
        # LF - RF : x < (x2 - x1) (y - y1) / (y2 - y1) + x1
        # RF - RH: y > (y2 - y1) (x - x1) / (x2 - x1) + y1
        # RH - LH : x > (x2 - x1) (y - y1) / (y2 - y1) + x1
        # LH - LF: y < (y2 - y1) (x - x1) / (x2 - x1) + y1
        lb = np.zeros(6)
        ub = np.zeros(6)
        
        constraint_FL_FR = x - (x_FR - x_FL) * (y - y_FL) / (y_FR - y_FL + 1e-6) - x_FL
        ub[0] = 0 
        lb[0] = -self.ACADOS_INFTY
        
        constraint_FR_RR = y - (y_RR - y_FR) * (x - x_FR) / (x_RR - x_FR + 1e-6) - y_FR
        ub[1] = self.ACADOS_INFTY
        lb[1] = 0
        
        constraint_RR_RL = x - (x_RL - x_RR) * (y - y_RR) / (y_RL - y_RR + 1e-6) - x_RR
        ub[2] = self.ACADOS_INFTY
        lb[2] = 0
        
        constraint_RL_FL = y - (y_FL - y_RL) * (x - x_RL) / (x_FL - x_RL + 1e-6) - y_RL
        ub[3] = 0
        lb[3] = -self.ACADOS_INFTY
        
        '''TODO: check this two constraints'''
        constraint_FL_RR = y - (y_RR - y_FL) * (x - x_FL) / (x_RR - x_FL + 1e-6) - y_FL
        ub[4] = self.ACADOS_INFTY
        lb[4] = -self.ACADOS_INFTY
        
        constraint_FR_RL = y - (y_RL - y_FR) * (x - x_FR) / (x_RL - x_FR + 1e-6) - y_FR
        ub[5] = self.ACADOS_INFTY
        lb[5] = -self.ACADOS_INFTY
        
        Jbu = cs.vertcat(constraint_FL_FR,
                        constraint_FR_RR,
                        constraint_RR_RL,
                        constraint_RL_FL,
                        constraint_FL_RR,
                        constraint_FR_RL)
        return Jbu, lb, ub   
        
    def get_foothold_bounds(self):
        """
        计算单条腿的上下界。
        逻辑：如果是支撑相，锁定在当前位置附近；如果是摆动相且有视觉，使用视觉约束；否则使用名义落点。
        """
        yaw = self.model.base_yaw
        base_pos = self.model.base_position
        # world -> body. horizontal rotation only
        R_wb = cs.SX.zeros(2,2)
        R_wb[0,0] = cs.cos(yaw)
        R_wb[0,1] = cs.sin(yaw)
        R_wb[1,0] = -cs.sin(yaw)
        R_wb[1,1] = cs.cos(yaw)
        
        foot_position_FL = cs.SX.zeros(3,1)
        foot_position_FL[0:2] = R_wb @ (self.model.state[12:14] - base_pos[0:2])
        foot_position_FL[2] = self.model.state[14]
        
        foot_position_FR = cs.SX.zeros(3,1)
        foot_position_FR[0:2] = R_wb @ (self.model.state[15:17] - base_pos[0:2])
        foot_position_FR[2] = self.model.state[17]
        
        foot_position_RL = cs.SX.zeros(3,1)
        foot_position_RL[0:2] = R_wb @ (self.model.state[18:20] - base_pos[0:2])
        foot_position_RL[2] = self.model.state[20]
        
        foot_position_RR = cs.SX.zeros(3,1)
        foot_position_RR[0:2] = R_wb @ (self.model.state[21:23] - base_pos[0:2])
        foot_position_RR[2] = self.model.state[23]
        Jbu = cs.vertcat(foot_position_FL, foot_position_FR, foot_position_RL, foot_position_RR)
        
        lb = np.ones(12) * self.ACADOS_INFTY
        ub = np.ones(12) * -self.ACADOS_INFTY
        return Jbu, lb, ub
        

    def get_friction_cone_bounds(self, f_min, f_max):
        n = np.array([0, 0, 1])  # 法向量
        t = np.array([1, 0, 0]) # 切向量1
        b = np.array([0, 1, 0]) # 切向量2
        mu = self.model.mu_friction
          # Derivation can be found in the paper
        # "High-slope terrain locomotion for torque-controlled quadruped robots",
        # 1. 定义单腿的 5x3 矩阵
        single_leg_cone = cs.SX.zeros(5, 3)
        single_leg_cone[0, :] = -n * mu + t   # fx - mu*fz
        single_leg_cone[1, :] = -n * mu + b   # fy - mu*fz
        single_leg_cone[2, :] =  n * mu + t   # fx + mu*fz
        single_leg_cone[3, :] =  n * mu + b   # fy + mu*fz
        single_leg_cone[4, :] =  n            # fz

        leg_cone_matrix = cs.diagcat(*[single_leg_cone for _ in range(4)])
        Jbu = leg_cone_matrix @ self.model.input[12:24]   
        
        lb = np.zeros(20)
        lb[0] = -self.ACADOS_INFTY
        lb[1] = -self.ACADOS_INFTY
        lb[2] = 0
        lb[3] = 0
        lb[4] = f_min
        lb[5:10] = lb[0:5]
        lb[10:15] = lb[0:5]
        lb[15:20] = lb[0:5]
        
        ub = np.zeros(20)
        ub[0] = 0
        ub[1] = 0
        ub[2] = self.ACADOS_INFTY
        ub[3] = self.ACADOS_INFTY
        ub[4] = f_max
        ub[5:10] = ub[0:5]
        ub[10:15] = ub[0:5]
        ub[15:20] = ub[0:5]
        return Jbu, lb, ub
    

        