from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.controllers.controller_factory import ControllerFactory
from quadruped_ctrl.interface.reference_interface import ReferenceInterface
from quadruped_ctrl.interface.wb_interface import WBInterface
'''
full stance 站立测试支撑腿
'''

def main() -> None:
    env = QuadrupedEnv(
        robot_config='robot/go1.yaml',
        model_path='quadruped_ctrl/assets/robot/go1/scene.xml',
        sim_config_path='sim_config.yaml'
    )
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    obs, _ = env.reset()
    
    mpc_config_path = "go1_mpc_config.yaml"
    mpc_controller = ControllerFactory.create_controller("mpc_gradient", env, mpc_config_path=mpc_config_path)
    ref_interface = ReferenceInterface(env, mpc_config_path=mpc_config_path)
    wb_interface = WBInterface(env)

    mpc_decimation = int(env.sim_config.get("physics", {}).get("mpc_frequency", 10))
    mpc_decimation = max(1, mpc_decimation)
    last_action = np.zeros(env.model.nu, dtype=np.float64)

    print("启动 MuJoCo 可视化查看器... (MPC 原地踏步)")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running():
            step_start = time.time()

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

            if step % mpc_decimation == 0:
                # MPC 计算最优 GRF 和落脚点
                optimal_GRF, optimal_footholds, optimal_next_state, status = mpc_controller.get_action(
                    state=state,
                    reference=reference_state,
                    contact_sequence=contact_sequence,
                    mass = env.robot.mass,
                    inertia = env.robot.inertia,
                    mu = env.mu
                )   
                # 将 GRF 写入 state（供 WBInterface 使用）
                for i, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
                    leg = state.get_leg_by_name(leg_name)
                    leg.contact_force = optimal_GRF[i*3:(i+1)*3].copy()
                
                # 使用 WBInterface 将 GRF 转换为关节力矩
                last_action = wb_interface.compute_tau(
                    state,
                    swing_targets=swing_refs,
                    contact_sequence=contact_sequence[:, 0],
                    optimal_GRF=optimal_GRF
                )

            env.step(last_action)
            viewer.sync()

            if step % 300 == 0:
                # 打印状态监控信息
                num_contact = sum([state.FL.contact_state, state.FR.contact_state, 
                                   state.RL.contact_state, state.RR.contact_state])
                base_height = state.base.pos[2]
                print(f"步数: {step:5d} | 时间: {env.data.time:6.2f}s | "
                      f"支撑腿数: {num_contact} | 身体高度: {base_height:.3f}m")
                print("  当前状态:")
                print(f"    base_pos: {np.array2string(state.base.pos, precision=3)}")
                print(f"    base_lin_vel: {np.array2string(state.base.lin_vel_world, precision=3)}")
                print(f"    base_euler: {np.array2string(state.base.euler, precision=3)}")
                print(f"    base_ang_vel: {np.array2string(state.base.ang_vel_world, precision=3)}")
                print("  参考状态:")
                print(f"    ref_pos: {np.array2string(reference_state.ref_position, precision=3)}")
                print(f"    ref_lin_vel: {np.array2string(reference_state.ref_linear_velocity, precision=3)}")
                print(f"    ref_orient: {np.array2string(np.asarray(reference_state.ref_orientation), precision=3)}")
                print(f"    ref_ang_vel: {np.array2string(np.asarray(reference_state.ref_angular_velocity), precision=3)}")
                print("  足端参考/当前:")
                print(f"    FL ref: {np.array2string(reference_state.ref_foot_FL, precision=3)}, cur: {np.array2string(state.FL.foot_pos, precision=3)}")
                print(f"    FR ref: {np.array2string(reference_state.ref_foot_FR, precision=3)}, cur: {np.array2string(state.FR.foot_pos, precision=3)}")
                print(f"    RL ref: {np.array2string(reference_state.ref_foot_RL, precision=3)}, cur: {np.array2string(state.RL.foot_pos, precision=3)}")
                print(f"    RR ref: {np.array2string(reference_state.ref_foot_RR, precision=3)}, cur: {np.array2string(state.RR.foot_pos, precision=3)}")
                print(f"  contact_sequence[0]: {contact_sequence[:, 0]}")

            step += 1
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

