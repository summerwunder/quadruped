"""
MPC求解器Status=1诊断脚本
用于调试Acados求解器无法收敛的问题 (Status=1: 达到最大迭代次数)
"""

import numpy as np
import mujoco
from pathlib import Path
from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.controllers.controller_factory import ControllerFactory
from quadruped_ctrl.interface.reference_interface import ReferenceInterface
from quadruped_ctrl.interface.wb_interface import WBInterface


def diagnose_mpc_issue():
    """诊断MPC求解器问题的完整函数"""
    print("=" * 80)
    print("MPC求解器诊断工具 - Status=1调试")
    print("=" * 80)
    
    # 1. 初始化环境
    print("\n[1/5] 初始化环境和控制器...")
    env = QuadrupedEnv(
        robot_config='robot/go1.yaml',
        model_path='quadruped_ctrl/assets/robot/go1/scene.xml',
        sim_config_path='sim_config.yaml'
    )
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    obs, _ = env.reset()
    
    # ⭐ 关键: 运行200步让机器人稳定到平衡状态（足端接触地面）
    print("  让机器人稳定到平衡状态...")
    for _ in range(200):
        env.step(np.zeros(env.model.nu))
        mujoco.mj_forward(env.model, env.data)
    
    mpc_controller = ControllerFactory.create_controller(
        "mpc_gradient", env, mpc_config_path="mpc_config.yaml"
    )
    ref_interface = ReferenceInterface(env, mpc_config_path="mpc_config.yaml")
    
    print(f"  ✓ 环境初始化成功")
    print(f"    - 机器人质量: {env.robot.mass:.2f} kg")
    print(f"    - MPC时间步长: {env.dt:.4f} s")
    print(f"    - MPC预测时域: {mpc_controller.T_horizon:.2f} s")
    print(f"    - 最大地面反力: {mpc_controller.grf_max:.2f} N")
    
    # 2. 检查MPC配置
    print("\n[2/5] 检查MPC配置参数...")
    print(f"  求解器配置:")
    print(f"    - 使用DDP: {mpc_controller.use_DDP}")
    print(f"    - 使用RTI: {mpc_controller.use_RTI}")
    print(f"    - 最大迭代数: {mpc_controller.num_qp_iterations}")
    print(f"    - 求解器模式: {mpc_controller.solver_mode}")
    print(f"  约束配置:")
    print(f"    - 足端位置约束: {mpc_controller.use_foothold_constraint}")
    print(f"    - 稳定性约束: {mpc_controller.use_stability_constraint}")
    print(f"  权重配置:")
    Q_mat, R_mat = mpc_controller._set_weight_by_config()
    print(f"    - 状态权重(Q)对角线: {np.diag(Q_mat)[:6]}")  # 前6个
    print(f"    - 控制权重(R)对角线最值: [{np.min(np.diag(R_mat)):.6f}, {np.max(np.diag(R_mat)):.6f}]")
    
    # 3. 运行几个时间步寻找失败的情况
    print("\n[3/5] 运行模拟并检测求解失败...")
    max_steps = 200
    failure_found = False
    
    for step in range(max_steps):
        state = env.get_state()
        com_pos = state.base.com.copy()
        
        ref_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ref_ang_vel = np.zeros(3, dtype=np.float64)
        
        reference_state, contact_sequence, swing_refs = ref_interface.get_reference_state(
            current_state=state,
            com_pos=com_pos,
            heightmaps=None,
            abs_time=env.data.time,
            ref_base_lin_vel=ref_lin_vel,
            ref_base_ang_vel=ref_ang_vel,
        )
        
        # 求解MPC
        optimal_GRF, optimal_footholds, optimal_next_state, status = mpc_controller.get_action(
            state=state,
            reference=reference_state,
            contact_sequence=contact_sequence,
        )
        
        # 检测失败
        if status != 0:
            failure_found = True
            print(f"\n  ⚠️  发现求解失败! (第{step}步, 仿真时间={env.data.time:.3f}s)")
            print(f"\n  >>> 求解器信息:")
            print(f"      Status: {status} (1=达到迭代限制, 2=线搜索失败, 4=QP失败)")
            
            # 获取求解统计信息
            stats = mpc_controller.acados_ocp_solver.get_stats('time_tot')
            print(f"      求解总耗时: {stats:.6f} s")
            
            # 打印当前状态信息
            print(f"\n  >>> 当前系统状态:")
            num_stance = int(sum(contact_sequence[:, 0]))
            print(f"      支撑腿数: {num_stance}")
            print(f"      接触序列第0步: {contact_sequence[:, 0]}")
            print(f"      基座位置: {state.base.pos}")
            print(f"      基座姿态(roll,pitch,yaw): {np.degrees(state.base.euler)}")
            print(f"      基座线速度: {state.base.lin_vel_world}")
            print(f"      基座角速度: {state.base.ang_vel_world}")
            
            # 打印足端位置
            print(f"\n  >>> 足端位置 (当前 vs 参考):")
            legs = [('FL', state.FL, reference_state.ref_foot_FL),
                   ('FR', state.FR, reference_state.ref_foot_FR),
                   ('RL', state.RL, reference_state.ref_foot_RL),
                   ('RR', state.RR, reference_state.ref_foot_RR)]
            max_foot_delta = 0
            for name, leg, ref in legs:
                delta = leg.foot_pos - ref
                delta_norm = np.linalg.norm(delta)
                max_foot_delta = max(max_foot_delta, delta_norm)
                print(f"      {name}: cur={leg.foot_pos}, ref={ref}")
                print(f"           delta={delta}, norm={delta_norm:.4f}m")
            if max_foot_delta > 0.1:
                print(f"\n      ⚠️  最大足端偏离: {max_foot_delta:.4f}m >> 期望 <0.01m")
                print(f"      这是导致约束无可行解的主要原因!")
            
            # 检查约束可行性
            print(f"\n  >>> 检查约束可行性:")
            _check_constraint_feasibility(mpc_controller, state, reference_state, contact_sequence)
            
            # 打印数值相关信息
            print(f"\n  >>> 数值问题诊断:")
            print(f"      成本函数权重缩放: {np.max(Q_mat) / np.min(Q_mat[Q_mat > 0]):.2e}")
            
            # 计算R矩阵条件数，避免除零
            R_diag = np.diag(R_mat)
            R_min = np.min(R_diag[R_diag > 1e-10])
            R_max = np.max(R_diag)
            if R_min > 1e-10:
                print(f"      控制权重缩放: {R_max / R_min:.2e}")
            else:
                print(f"      控制权重缩放: ∞ (存在零或极小权重 - 这会导致数值问题!)")
                print(f"      R矩阵对角线: {R_diag}")
            
            print(f"      上次求解状态: {mpc_controller.previous_status}")
            
            break
        
        # 更新环境
        for i, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
            leg = state.get_leg_by_name(leg_name)
            leg.contact_force = optimal_GRF[i*3:(i+1)*3].copy()
        
        last_action = WBInterface(env).compute_tau(
            state,
            swing_targets=swing_refs,
            contact_sequence=contact_sequence[:, 0],
            optimal_GRF=optimal_GRF
        )
        env.step(last_action)
    
    if not failure_found:
        print(f"  ✓ 前{max_steps}步运行正常，未发现求解失败")
        print(f"  → 提示: Status=1可能在特定动作或姿态才出现")
    
    print("\n" + "=" * 80)
    print("诊断完成 - 修复方案:")
    print("=" * 80)
    if max_foot_delta > 0.1 if 'max_foot_delta' in locals() else False:
        print("""
        🔴 主要问题：足端位置初始化错误！
        
        快速修复：
        1. 检查 sim_config.yaml 中的机器人初始高度设置
        2. 确保初始化后足端z位置 = 接触地面的高度
        3. 在 stay_demo.py 的 env.reset() 后添加稳定化步骤:
           
           for _ in range(200):
               env.step(np.zeros(env.model.nu))
               mujoco.mj_forward(env.model, env.data)
        
        4. 然后再初始化MPC控制器
        """)


def _check_constraint_feasibility(mpc_controller, state, reference_state, contact_sequence):
    """检查约束是否过紧或冲突"""
    
    # 检查支撑腿约束
    stance_offset = 0.005  # 从set_stage_constraint中获取
    yaw = state.base.euler[2]
    base_xy = state.base.pos[:2]
    R_wb = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    
    for i, (name, leg, contact) in enumerate([
        ('FL', state.FL, contact_sequence[0, 0]),
        ('FR', state.FR, contact_sequence[1, 0]),
        ('RL', state.RL, contact_sequence[2, 0]),
        ('RR', state.RR, contact_sequence[3, 0]),
    ]):
        if contact == 1:  # 支撑腿
            # 局部坐标
            rel_xy = R_wb @ (leg.foot_pos_centered[:2] - base_xy)
            constraint_radius = stance_offset
            print(f"      {name} (支撑): 约束半径={constraint_radius:.4f}m, " +
                  f"相对位置=({rel_xy[0]:.3f}, {rel_xy[1]:.3f})")
    
    # 检查摩擦力是否都是正的
    num_stance = int(sum(contact_sequence[:, 0]))
    if num_stance > 0:
        f_z_nominal = mpc_controller.grf_max / num_stance
        print(f"      单腿名义垂直力: {f_z_nominal:.2f} N " +
              f"(总:{mpc_controller.grf_max:.2f}N, 支撑腿数:{num_stance})")
        if f_z_nominal <= 0.1:
            print(f"      ⚠️  警告: 单腿力很小，可能导致约束冲突!")


if __name__ == "__main__":
    try:
        diagnose_mpc_issue()
    except Exception as e:
        print(f"\n❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
