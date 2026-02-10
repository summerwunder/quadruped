from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.controllers.controller_factory import ControllerFactory
from quadruped_ctrl.interface.reference_interface import ReferenceInterface


def main() -> None:
	repo_root = Path(__file__).resolve().parents[1]

	env = QuadrupedEnv(
		robot_config=str(repo_root / "quadruped_ctrl/config/robot/go1.yaml"),
		model_path=str(repo_root / "quadruped_ctrl/assets/robot/go1/scene.xml"),
		sim_config_path=str(repo_root / "quadruped_ctrl/config/sim_config.yaml"),
	)
	env.reset()

	mpc_config_path = str(repo_root / "quadruped_ctrl/config/mpc_config.yaml")
	mpc_controller = ControllerFactory.create_controller("mpc_gradient", env, mpc_config_path=mpc_config_path)
	ref_interface = ReferenceInterface(env, mpc_config_path=mpc_config_path)

	mpc_decimation = int(env.sim_config.get("physics", {}).get("mpc_frequency", 10))
	mpc_decimation = max(1, mpc_decimation)
	last_action = np.zeros(env.model.nu, dtype=np.float64)

	print("启动 MuJoCo 可视化查看器... (MPC 原地踏步)")

	with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
		step = 0
		while viewer.is_running():
			step_start = time.time()

			state = env.get_state()
			com_pos = state.base.pos.copy()

			ref_lin_vel = np.array([0.01, 0.0, 0.0], dtype=np.float64)
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
				last_action = mpc_controller.get_action(
					state=state,
					reference=reference_state,
					contact_sequence=contact_sequence,
					swing_refs=swing_refs,
				)

			env.step(last_action)
			viewer.sync()

			if step % 50 == 0:
				print(f"步数: {step}, 仿真时间: {env.data.time:.2f}s")

			step += 1
			time_until_next_step = env.model.opt.timestep - (time.time() - step_start)
			if time_until_next_step > 0:
				time.sleep(time_until_next_step)


if __name__ == "__main__":
	main()
