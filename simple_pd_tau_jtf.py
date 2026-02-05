import numpy as np
import mujoco
import mujoco.viewer
import time
from quadruped_ctrl import QuadrupedEnv 

def verify_kinematics_and_forces_visual():
    # 1. 初始化环境
    env = QuadrupedEnv()
    env.reset()
    
    # 2. PD 参数
    qpos_des = np.array([0, 0.7, -1.6] * 4)
    kp, kd = 60.0, 3.5
    
    print("正在启动可视化窗口...")
    
    # 3. 开启可视化
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.time()

            # --- A. PD 控制 ---
            # 提取关节状态 (12维)
            qpos_current = np.concatenate([
                env.state.FL.qpos, env.state.FR.qpos,
                env.state.RL.qpos, env.state.RR.qpos
            ])
            qvel_current = np.concatenate([
                env.state.FL.qvel, env.state.FR.qvel,
                env.state.RL.qvel, env.state.RR.qvel
            ])
            
            tau_pd = kp * (qpos_des - qpos_current) - kd * qvel_current
            
            # --- B. 执行仿真步 ---
            env.step(tau_pd)
            
            # 刷新可视化修改
            viewer.sync()

            # --- C. 实时数据计算与输出 (每 20 步输出一次，避免刷屏太快) ---
            if step % 20 == 0:
                print("\033[H\033[J")  # 清除终端屏幕（Linux/Mac有效，让数据看起来是原地刷新的）
                print(f"=== 实时物理一致性验证 | 步数: {step} | 时间: {env.data.time:.2f}s ===")
                print(f"{'腿名':<5} | {'接触':<5} | {'电机力矩 (tau_ctrl)':^24} | {'接触力矩 (-J.T@F)':^24} | {'残差':>8}")
                print("-" * 100)

                for leg_name in ['FL', 'FR', 'RL', 'RR']:
                    leg = getattr(env.state, leg_name)
                    q_idx = leg.qvel_idxs
                    
                    # 计算 J^T * F
                    # J 是世界系雅可比，F 是世界系接触力
                    tau_jtf = -leg.jac_pos_world[:, q_idx].T @ leg.contact_force
                    
                    tau_ctrl = leg.tau
                    res = np.linalg.norm(tau_ctrl - tau_jtf)
                    
                    # 格式化输出数据
                    contact_str = "YES" if leg.contact_state else "NO"
                    ctrl_str = "/".join([f"{x:6.2f}" for x in tau_ctrl])
                    jtf_str  = "/".join([f"{x:6.2f}" for x in tau_jtf])
                    
                    print(f"{leg_name:<5} | {contact_str:<5} | {ctrl_str:^24} | {jtf_str:^24} | {res:8.3f}")
                
                print("\n[提示] 静止时残差主要由腿部重力引起；若符号相反且数值接近，说明雅可比正确。")

            # 控制仿真频率（可选，视需要调整）
            time_until_next_step = env.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            step += 1

if __name__ == "__main__":
    verify_kinematics_and_forces_visual()