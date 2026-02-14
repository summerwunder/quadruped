import numpy as np
from scipy.spatial.transform import Rotation as R
import collections
class FootholdGenerator:
    def __init__(self, stance_time=0.25, hip_offset_y=0.08, robot_height=0.3, gravity=9.81):
        self.stance_time = stance_time
        self.hip_offset_y = hip_offset_y 
        self.robot_height = robot_height
        self.g = gravity

        self.vel_error_gain = np.sqrt(self.robot_height / self.g)
        
        # 抬脚位置、落地位置
        self.lift_off_positions = {'FL': np.zeros(3), 'FR': np.zeros(3), 
                                   'RL': np.zeros(3), 'RR': np.zeros(3)}
        self.touch_down_positions = {'FL': np.zeros(3), 'FR': np.zeros(3), 
                                     'RL': np.zeros(3), 'RR': np.zeros(3)}
        
        # 记录上一帧的接触状态，用于检测 Stance -> Swing 和 Swing -> Stance 的切换
        self.prev_contact_states = {'FL': True, 'FR': True, 'RL': True, 'RR': True}
        
        self.base_vel_hist = collections.deque(maxlen=20)

    def compute_footholds(self, state, ref_lin_vel_w):
        """
        计算参考落脚点
        state: QuadrupedState 对象
        ref_lin_vel_w: 世界坐标系下的参考线速度 [vx, vy]
        """
        pos_w = state.base.pos
        yaw = state.base.euler[2]
        cur_vel_w = state.base.lin_vel_world[:2]
        # 世界系 <-> 水平系
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_W2H = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
        
        # 3. 计算启发式位移 (Raibert term)
        # 身体移动补偿：在支撑时间内，身体会向前跑，所以脚要前插
        raibert_offset_h = (self.stance_time / 2.0) * (R_W2H @ ref_lin_vel_w)
        
        # 4. 计算速度误差补偿 (Capture Point term)
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
            # z_w = leg.hip_pos_world[2] - self.robot_height
            z_w = max(leg.hip_pos_world[2] - self.robot_height, 0.00)  
            
            ref_footholds[leg_name] = np.array([target_w[0], target_w[1], z_w])
            
        return ref_footholds
    
    def update_contact_states(self, state, contact_sequence: np.ndarray):
        """
        更新足端接触状态，自动记录抬脚点和落地点
        
        Args:
            state: QuadrupedState 对象
            contact_sequence: 当前接触序列 (4,) [FL, FR, RL, RR], 1=支撑, 0=摆动
        
        逻辑：
            - Stance (1) -> Swing (0)：记录 lift_off_positions（抬脚瞬间）
            - Swing (0) -> Stance (1)：记录 touch_down_positions（落地瞬间）
        """
        leg_names = ['FL', 'FR', 'RL', 'RR']
        
        for i, leg_name in enumerate(leg_names):
            leg = state.get_leg_by_name(leg_name)
            is_stance_now = (contact_sequence[i] == 1)  
            was_stance_prev = self.prev_contact_states[leg_name]  
            
            # 检测：支撑 -> 摆动（抬脚瞬间）
            if was_stance_prev and not is_stance_now:
                self.lift_off_positions[leg_name] = leg.foot_pos_world.copy()
                # print(f"[FootholdGen] {leg_name} Lift-Off at {self.lift_off_positions[leg_name]}")
            
            # 检测：摆动 -> 支撑（落地瞬间）
            elif not was_stance_prev and is_stance_now:
                self.touch_down_positions[leg_name] = leg.foot_pos_world.copy()
                # print(f"[FootholdGen] {leg_name} Touch-Down at {self.touch_down_positions[leg_name]}")       
            self.prev_contact_states[leg_name] = is_stance_now



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

        ref_base_xy_lin_vel = np.array([0.2, 0.0])
        foothold_gen = FootholdGenerator(stance_time=0.25, robot_height=0.28)
        
        print("=" * 60)
        print("测试 1: compute_footholds() - 落脚点计算")
        print("=" * 60)
        footholds_reference = foothold_gen.compute_footholds(state, ref_base_xy_lin_vel)

        print(f"机身位置: {state.base.pos}")
        print(f"当前速度: {state.base.lin_vel_world[:2]}")
        print(f"参考速度: {ref_base_xy_lin_vel}")

        print("\n落脚点计算结果 (世界坐标系):")
        for name in ['FL', 'FR', 'RL', 'RR']:
            hip = getattr(state, name).hip_pos_world
            ref = footholds_reference[name]
            print(f"  {name}: {ref} (髋关节偏移: {ref[:2] - hip[:2]})")
        
        print("\n" + "=" * 60)
        print("测试 2: update_contact_states() - 接触状态跟踪")
        print("=" * 60)
        
        # 模拟 Trot 步态的接触序列变化
        # Trot: FL-RR 一组, FR-RL 一组
        test_sequences = [
            ("初始 (全站立)", np.array([1, 1, 1, 1])),
            ("Phase 1 (FL-RR 摆动)", np.array([0, 1, 1, 0])),
            ("Phase 2 (全站立)", np.array([1, 1, 1, 1])),
            ("Phase 3 (FR-RL 摆动)", np.array([1, 0, 0, 1])),
            ("Phase 4 (全站立)", np.array([1, 1, 1, 1])),
        ]
        
        for i, (phase_name, contact_seq) in enumerate(test_sequences):
            print(f"\n[Step {i}] {phase_name}")
            print(f"  Contact Sequence: {dict(zip(['FL', 'FR', 'RL', 'RR'], contact_seq))}")
            
            # 模拟足端位置变化（实际中从传感器/仿真获取）
            for leg_name in ['FL', 'FR', 'RL', 'RR']:
                leg = getattr(state, leg_name)
                # 简单模拟：摆动腿抬高 0.1m
                leg_idx = ['FL', 'FR', 'RL', 'RR'].index(leg_name)
                if contact_seq[leg_idx] == 0:  # 摆动中
                    leg.foot_pos_world[2] += 0.1  # 抬高
            
            # 调用状态更新
            foothold_gen.update_contact_states(state, contact_seq)
            
            # 检查是否有新记录
            print(f"  Lift-Off 记录:")
            for leg_name in ['FL', 'FR', 'RL', 'RR']:
                pos = foothold_gen.lift_off_positions[leg_name]
                if not np.allclose(pos, 0):
                    print(f"    {leg_name}: {pos}")
            
            print(f"  Touch-Down 记录:")
            for leg_name in ['FL', 'FR', 'RL', 'RR']:
                pos = foothold_gen.touch_down_positions[leg_name]
                if not np.allclose(pos, 0):
                    print(f"    {leg_name}: {pos}")
        
        print("\n" + "=" * 60)
        print("✓ 测试完成！检查上面的 Lift-Off 和 Touch-Down 记录")
        print("=" * 60)
        
    finally:
        env.close()