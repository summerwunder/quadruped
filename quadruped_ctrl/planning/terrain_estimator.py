import numpy as np

class TerrainEstimator:
    def __init__(self, alpha=0.95):
        # 状态变量
        self.terrain_roll = 0.0
        self.terrain_pitch = 0.0
        self.terrain_height = 0.0
        self.robot_height = 0.3  
        
        # 滤波系数 (alpha 越大越平滑，响应越慢)
        self.alpha = alpha
        # 激活开关
        self.roll_activated = True
        self.pitch_activated = True

    def update(self, state):
        """
        根据 QuadrupedState 实时更新地形估计
        """
        # 1. 提取基础数据
        base_pos = state.base.pos
        yaw = state.base.euler[2]
        
        leg_names = ['FL', 'FR', 'RL', 'RR']
        feet_pos_w = np.array([getattr(state, name).foot_pos_world for name in leg_names])
        contacts = np.array([getattr(state, name).contact_state for name in leg_names])
        
        # 2. 坐标变换：世界系 -> 水平航向系 (Horizontal Frame)
        # 只绕 Yaw 旋转，忽略 Roll/Pitch，用于感知地形坡度
        c, s = np.cos(yaw), np.sin(yaw)
        R_W2H = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
        
        # 计算足端相对于基座的水平系偏移 (4, 3)
        feet_rel_h = (feet_pos_w - base_pos) @ R_W2H.T

        # 3. 计算坡度 (Pitch & Roll)
        # 利用对角或相邻足端的高度差
        # Pitch: 前后高度差 / 前后水平距离
        # (FL_z - RL_z) / dist + (FR_z - RR_z) / dist
        delta_z_pitch = (feet_rel_h[0, 2] - feet_rel_h[2, 2]) + (feet_rel_h[1, 2] - feet_rel_h[3, 2])
        delta_x_pitch = (feet_rel_h[0, 0] - feet_rel_h[2, 0]) + (feet_rel_h[1, 0] - feet_rel_h[3, 0])
        raw_pitch = np.arctan2(delta_z_pitch, np.abs(delta_x_pitch) + 1e-3)

        # Roll: 左右高度差 / 左右水平距离
        # (FL_z - FR_z) / dist + (RL_z - RR_z) / dist
        delta_z_roll = (feet_rel_h[0, 2] - feet_rel_h[1, 2]) + (feet_rel_h[2, 2] - feet_rel_h[3, 2])
        delta_y_roll = (feet_rel_h[0, 1] - feet_rel_h[1, 1]) + (feet_rel_h[2, 1] - feet_rel_h[3, 1])
        raw_roll = np.arctan2(delta_z_roll, np.abs(delta_y_roll) + 1e-3)

        # 4. 滤波更新坡度
        if self.pitch_activated:
            self.terrain_pitch = self.alpha * self.terrain_pitch + (1 - self.alpha) * raw_pitch
        if self.roll_activated:
            self.terrain_roll = self.alpha * self.terrain_roll + (1 - self.alpha) * raw_roll

        # 5. 计算地面高度 (只统计触地足)
        # 如果没有脚触地（比如跳跃），则维持现状
        if np.any(contacts > 0):
            # 统计所有在地面上的脚的 Z 坐标
            z_ground_actual = np.mean(feet_pos_w[contacts > 0, 2])
            
            # 地面海拔高度滤波
            self.terrain_height = 0.8 * self.terrain_height + 0.2 * z_ground_actual
            
            # 机器人实际离地高度 (Base_Z - Ground_Z)
            current_h = base_pos[2] - z_ground_actual
            self.robot_height = 0.8 * self.robot_height + 0.2 * current_h

        return self.terrain_roll, self.terrain_pitch, self.terrain_height, self.robot_height