# des_foot_pos, des_foot_vel, des_foot_acc 
from quadruped_ctrl.datatypes import ReferenceState
from quadruped_ctrl.datatypes import QuadrupedState
from quadruped_ctrl.utils.config_loader import ConfigLoader
from quadruped_ctrl.quadruped_env import QuadrupedEnv
from quadruped_ctrl.planning.swing_trajectory_generator import SwingTrajectoryGenerator
from quadruped_ctrl.planning.periodic_gait_generator import PeriodicGaitGenerator
from quadruped_ctrl.planning.foothold_reference_generator import FootholdGenerator
from quadruped_ctrl.planning.terrain_estimator import TerrainEstimator
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R

class ReferenceInterface:
    def __init__(self, env: QuadrupedEnv, mpc_config_path:str):
        self.env = env
        self.robot = env.robot
        self.sim_config = env.sim_config
        self.mpc_config = ConfigLoader.load_mpc_config(mpc_config_path)
        
        dt = self.sim_config.get('physics', {}).get('dt', 0.002)
        horizon = self.mpc_config.get('horizon')
        if self.mpc_config.get('solver').get('use_nonuniform_discretization'):
            # TODO: 支持非均匀时间步长
            pass
        else:
            self.contact_sequence_dts = np.full(horizon, dt)
          
        self.gait_params = ConfigLoader.load_gait_params(self.env.sim_config_path)
        self.gait_active = self.sim_config.get('gait').get('active', 'trot')
        gait_param = self.gait_params.get(self.gait_active)
        # 初始化步态生成器
        self.gait_generator = PeriodicGaitGenerator(
            duty_factor=gait_param['duty_factor'],
            step_freq=gait_param['step_freq'],
            phase_offsets=gait_param['phase_offsets']
        )
        
        # 初始化摇摆轨迹生成器
        swing_height = self.robot.step_height
        swing_duration = (1 - self.gait_generator.duty_factor) * (1 / self.gait_generator.step_freq)
        self.swing_generator = SwingTrajectoryGenerator(
             swing_height=swing_height,
             swing_duration=swing_duration)
        
        # 落脚点生成器
        stance_time = (1 / self.gait_generator.step_freq) * self.gait_generator.duty_factor
        self.foothold_generator = FootholdGenerator(
            stance_time=stance_time,
            robot_height=self.robot.hip_height,
            gravity=self.env.sim_config.get('physics').get('gravity', 9.81)
        )
        
        self.terrain_estimator = TerrainEstimator(alpha=0.95)
        # feet_pos: np.ndarray,
        # hip_pos: np.ndarray,
        # joints_pos: np.ndarray,
    def get_reference_state(
        self,
        current_state: QuadrupedState,
        com_pos: np.ndarray,
        heightmaps,
        abs_time: float,
        ref_base_lin_vel: np.ndarray,
        ref_base_ang_vel: np.ndarray,
    ) -> Tuple[ReferenceState, np.ndarray, dict]:
        """计算参考状态      
        Args:
            current_state: 当前机器人状态
            com_pos: 机器人质心位置
            heightmaps: 地形高度图
            abs_time: 当前绝对时间
            ref_base_lin_vel: 参考线速度
            ref_base_ang_vel: 参考角速度
        
        Returns:
            ReferenceState对象
            contact_sequence: 支撑/摆动序列 (4, H)
            swing_refs: 摆动腿参考轨迹 {'FL': {'pos':(H,3),'vel':(H,3),'acc':(H,3)}, ...}
            
        """
        # TODO: 调整期望速度，如果机器人处于异常位置
        
        # 更新 gait 的 start/stop 判定
        self.gait_generator.update_start_and_stop(
            base_lin_vel=current_state.base.lin_vel,
            base_ang_vel=current_state.base.ang_vel,
            ref_lin_vel=ref_base_lin_vel,
            ref_ang_vel=ref_base_ang_vel,
            feet_dist_to_hip_max=current_state.get_max_feet_dist_to_hip(),
            base_rpy=current_state.base.euler
        )

        contact_sequence = self.gait_generator.get_horizon_sequence(abs_time, self.contact_sequence_dts)

        # Adjust reference velocity based on terrain estimation BEFORE foothold computation
        terrain_roll, terrain_pitch, terrain_height, robot_height = self.terrain_estimator.update(current_state)
        ref_base_lin_vel = R.from_euler("xyz", [terrain_roll, terrain_pitch, 0]).as_matrix() @ ref_base_lin_vel
        if(terrain_pitch > 0.0):
            ref_base_lin_vel[2] = -ref_base_lin_vel[2]
        if(np.abs(terrain_pitch) > 0.2):
            ref_base_lin_vel[0] = ref_base_lin_vel[0]/2.
            ref_base_lin_vel[2] = ref_base_lin_vel[2]*2

        # foothold generator expects reference velocity in world/base XY (2,)
        ref_lin_w = ref_base_lin_vel[0:2]
        ref_footholds = self.foothold_generator.compute_footholds(
            current_state,
            ref_lin_w
        )
        
        ref_pos = np.array([0, 0, self.robot.hip_height])
        
        
        ref_pos[2] -= current_state.base.pos[2] - com_pos[2]
        
        # TODO: 目前只考虑 Roll 和 Pitch ，YAW应该是任务为导向
        reference_orientation =  [terrain_roll, terrain_pitch, 0]    
        reference_state = ReferenceState(
            ref_foot_FL = ref_footholds['FL'],
            ref_foot_FR = ref_footholds['FR'],
            ref_foot_RL = ref_footholds['RL'],
            ref_foot_RR = ref_footholds['RR'],
            ref_linear_velocity=ref_base_lin_vel,
            ref_angular_velocity=ref_base_ang_vel,
            ref_orientation=reference_orientation,
            ref_position=ref_pos,
        )
        
        # 计算 swing 参考轨迹（pos, vel, acc），以供控制器/日志使用
        swing_refs = self._compute_swing_references(
            current_state=current_state,
            contact_sequence=contact_sequence,
            foothold_targets=ref_footholds,
            dt_list=self.contact_sequence_dts
        )

        return reference_state, contact_sequence, swing_refs

    def _compute_swing_references(self,
                                  current_state: QuadrupedState,
                                  contact_sequence: np.ndarray,
                                  foothold_targets: dict,
                                  dt_list: np.ndarray):
        """Compute swing pos/vel/acc references for the first timestep only.

        Returns dict: {leg: {'pos':(3,),'vel':(3,),'acc':(3,)}}
        """
        legs = ['FL', 'FR', 'RL', 'RR']
        dt_arr = np.asarray(dt_list, dtype=np.float64)

        results = {}
        for i, leg_name in enumerate(legs):
            seq = contact_sequence[i, :]
            leg = current_state.get_leg_by_name(leg_name)
            
            # 只计算第一个时间步 (k=0)
            if seq[0] == 0:
                # 摆动相：计算摆动轨迹
                lift_off_pos = np.asarray(leg.foot_pos, dtype=np.float64).copy()
                touch_down = np.asarray(foothold_targets[leg_name], dtype=np.float64)
                p, v, a = self.swing_generator.get_wing_reference_trajectory(0.0, lift_off_pos, touch_down)
            else:
                # 支撑相：脚保持当前位置
                p = np.asarray(leg.foot_pos, dtype=np.float64).copy()
                v = np.zeros(3, dtype=np.float64)
                a = np.zeros(3, dtype=np.float64)
            
            results[leg_name] = {'pos': p, 'vel': v, 'acc': a}

        return results