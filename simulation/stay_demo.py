from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.controllers.controller_factory import ControllerFactory
from quadruped_ctrl.interface.reference_interface import ReferenceInterface
from quadruped_ctrl.interface.wb_interface import WBInterface


def main() -> None:
    env = QuadrupedEnv(
        robot_config='robot/go1.yaml',
        model_path='quadruped_ctrl/assets/robot/go1/scene.xml',
        sim_config_path='sim_config.yaml'
    )
    obs, _ = env.reset()
    
    mpc_config_path = "mpc_config.yaml"
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
                )
                if status != 0 and step % 300 == 0:
                    print(f"[警告] MPC 求解器状态异常: {status}")
                    print(f"  当前高度: {state.base.pos[2]:.3f}m, 参考高度: {reference_state.ref_position[2]:.3f}m")
                    print(f"  参考Z位置(相对): {reference_state.ref_position}")
                    print(f"  支撑腿数: {sum([state.FL.contact_state, state.FR.contact_state, state.RL.contact_state, state.RR.contact_state])}")
                
                # 将 GRF 写入 state（供 WBInterface 使用）
                for i, leg_name in enumerate(['FL', 'FR', 'RL', 'RR']):
                    leg = state.get_leg_by_name(leg_name)
                    leg.contact_force = optimal_GRF[i*3:(i+1)*3].copy()
                
                # 使用 WBInterface 将 GRF 转换为关节力矩
                last_action = wb_interface.compute_tau(state, swing_targets=swing_refs)

            env.step(last_action)
            viewer.sync()

            if step % 300 == 0:
                # 打印状态监控信息
                num_contact = sum([state.FL.contact_state, state.FR.contact_state, 
                                   state.RL.contact_state, state.RR.contact_state])
                base_height = state.base.pos[2]
                print(f"步数: {step:5d} | 时间: {env.data.time:6.2f}s | "
                      f"支撑腿数: {num_contact} | 身体高度: {base_height:.3f}m")

            step += 1
            time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()

