import numpy as np
from scipy.spatial.transform import Rotation as R

class FootholdGenerator:
    def __init__(self, stance_time=0.25, hip_offset_y=0.08, robot_height=0.3, gravity=9.81):
        self.stance_time = stance_time
        self.hip_offset_y = hip_offset_y  # 让腿向两侧分得更开，增加稳定性
        self.robot_height = robot_height
        self.g = gravity

        self.vel_error_gain = np.sqrt(self.robot_height / self.g)

    def compute_footholds(self, state, ref_lin_vel_w):
        """
        计算参考落脚点
        state: QuadrupedState 对象
        ref_lin_vel_w: 世界坐标系下的参考线速度 [vx, vy]
        """
        # 1. 提取当前状态
        pos_w = state.base.pos
        yaw = state.base.euler[2]
        cur_vel_w = state.base.lin_vel_world[:2]
        
        # 2. 构建旋转矩阵 (世界系 <-> 水平系)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_W2H = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
        
        # 3. 计算启发式位移 (Raibert term)
        # 身体移动补偿：在支撑时间内，身体会向前跑，所以脚要前插
        raibert_offset_h = (self.stance_time / 2.0) * (R_W2H @ ref_lin_vel_w)
        
        # 4. 计算速度误差补偿 (Capture Point term)
        # 如果实际速度偏离目标，额外增加落脚位移来修正
        vel_error_h = self.vel_error_gain * (R_W2H @ (cur_vel_w - ref_lin_vel_w))
        
        # 限制修正量，防止步子迈得太大导致奇异解
        total_offset_h = np.clip(raibert_offset_h + vel_error_h, -0.15, 0.15)
        
        ref_footholds = {}
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            leg = state.get_leg_by_name(leg_name)
            
            # 5. 基于髋关节位置计算名义落脚点
            # 在水平系下，名义落脚点 = 髋关节水平投影 + 侧向偏置
            hip_pos_h = R_W2H @ (leg.hip_pos_world[:2] - pos_w[:2])
            
            target_h = hip_pos_h + total_offset_h
            
            # 加入 y 方向偏置（让步态更宽，不容易自己拌自己）
            if 'L' in leg_name:
                target_h[1] += self.hip_offset_y
            else:
                target_h[1] -= self.hip_offset_y
                
            # 6. 转回世界坐标系
            target_w = R_W2H.T @ target_h + pos_w[:2]
            
            # 高度 Z 通常跟随地面或髋关节当前高度
            z_w = leg.hip_pos_world[2] - self.robot_height
            
            ref_footholds[leg_name] = np.array([target_w[0], target_w[1], z_w])
            
        return ref_footholds



if __name__ == '__main__':
    try:
        from quadruped_ctrl.quadruped_env import QuadrupedEnv
    except ModuleNotFoundError:
        import os
        import sys

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from quadruped_ctrl.quadruped_env import QuadrupedEnv

    env = QuadrupedEnv()
    try:
        env.reset()
        state = env.get_state()
        assert state is not None

        ref_base_xy_lin_vel = np.array([0, 0.0])
        foothold_gen = FootholdGenerator(stance_time=0.25, robot_height=0.28)
        footholds_reference = foothold_gen.compute_footholds(state, ref_base_xy_lin_vel)

        print("--- 机器人状态 ---")
        print(f"机身位置: {state.base.pos}")
        print(f"当前速度: {state.base.lin_vel_world[:2]}")
        print(f"参考速度: {ref_base_xy_lin_vel}")

        print("\n--- 落脚点计算结果 (世界坐标系) ---")
        for name in ['FL', 'FR', 'RL', 'RR']:
            hip = getattr(state, name).hip_pos_world
            ref = footholds_reference[name]
            print(f"{name} 腿:")
            print(f"  髋关节位置: {hip[:2]}")
            print(f"  期望落脚点: {ref[:2]}")
            print(f"  前插距离 (Offset): {ref[:2] - hip[:2]}")
    finally:
        env.close()