"""
Data types and dataclasses for quadruped robot state representation.
Independent from gym_quadruped.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class LegJointMap:
    """存储单条腿的关节信息和状态"""
    leg_name: str
    
    # ========== 索引信息 (从 MuJoCo 模型获取，一次性设置) ==========
    qpos_idxs: np.ndarray = None      # 关节角度在 data.qpos 中的位置 (3,)
    qvel_idxs: np.ndarray = None      # 关节速度在 data.qvel 中的位置 (3,)
    tau_idxs: np.ndarray = None       # 控制力矩在 data.ctrl 中的位置 (3,)
    
    # ========== 关节空间状态 ==========
    qpos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))    # 关节角度 [rad]
    qvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))    # 关节角速度 [rad/s]
    
    tau: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))     # 控制力矩 [N·m]
    
    # ========== 笛卡尔空间状态 (基座坐标系) ==========
    foot_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))      # 足端相对于基座的位置 [m]
    foot_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))      # 足端相对于基座的速度 [m/s]
 
    hip_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))       # 髋关节相对于基座的位置 [m]
    
    # ========== 笛卡尔空间状态 (世界坐标系) ==========
    foot_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64)) # 足端在世界系下的位置 [m]
    foot_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64)) # 足端在世界系下的速度 [m/s]
    
    foot_pos_centered: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # 足端相对于质心的位置 [m]
    hip_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # 髋关节在世界系下的位置 [m]
    
    # ========== 接触信息 ==========
    contact_state: bool = False                # 是否接触地面
    contact_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # 接触力 (3D GRF) [N]
    contact_normal_force: float = 0.0          # 垂直地面反作用力 (标量) [N]
    
    # ========== Jacobian 矩阵 (世界坐标系) ==========
    jac_pos_world: np.ndarray = None           # 足端位置 Jacobian (3, 18) - 世界坐标系
    jac_rot_world: np.ndarray = None           # 足端旋转 Jacobian (3, 18) - 世界坐标系
    jac_dot_world: np.ndarray = None           # 足端位置 Jacobian 导数 (3, 18) - 世界坐标系
    
    # ========== Jacobian 矩阵 (基座坐标系) ==========
    jac_pos_base: np.ndarray = None            # 足端位置 Jacobian (3, 18) - 基座坐标系
    jac_rot_base: np.ndarray = None            # 足端旋转 Jacobian (3, 18) - 基座坐标系
    jac_dot_base: np.ndarray = None            # 足端位置 Jacobian 导数 (3, 18) - 基座坐标系
    
    # ========== 惯性矩阵 (关节空间) ==========
    mass_matrix: np.ndarray = None             # 该腿的 3×3 质量/惯性矩阵 (3, 3)
    qfrc_bias: np.ndarray = None               # 该腿的 Coriolis/离心/重力项 (3,)
    def __post_init__(self):
        """验证数据一致性"""
        if self.qpos_idxs is not None:
            self.qpos = np.zeros(len(self.qpos_idxs), dtype=np.float64)
        if self.qvel_idxs is not None:
            self.qvel = np.zeros(len(self.qvel_idxs), dtype=np.float64)
        if self.tau_idxs is not None:
            self.tau = np.zeros(len(self.tau_idxs), dtype=np.float64)
    
    def get_dof(self) -> int:
        """获取该腿的自由度数"""
        if self.qpos_idxs is not None:
            return len(self.qpos_idxs)
        return 3


@dataclass
class BaseState:
    """机器人基座（身体）的状态"""
    
    # ========== 位置和姿态 (世界坐标系) ==========
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))          # 基座位置 [m]
    pos_centered: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # 基座位置（相对于质心）[m]
    com: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))          # 质心位置（世界坐标系）[m]
    com_centered: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))  # 质心位置（相对于base.pos）[m]
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float64))  # 四元数 [w, x, y, z]
    rot_mat: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))       # 旋转矩阵 (3x3)
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))       # 欧拉角 [roll, pitch, yaw] [rad]
    
    # ========== 速度和角速度 (通常是机身坐标系) ==========
    lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))     # 线速度 [m/s]
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))     # 角速度 [rad/s]
    
    # ========== 加速度 ==========
    lin_acc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))     # 线加速度 [m/s²]
    ang_acc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))     # 角加速度 [rad/s²]
    
    # ========== 世界坐标系速度 (可选) ==========
    lin_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))   # 世界坐标系线速度 [m/s]
    ang_vel_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))   # 世界坐标系角速度 [rad/s]
    
    # ========== 重力向量 (基座坐标系中的重力方向) ==========
    gravity_vec: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1], dtype=np.float64))  # 重力向量 (归一化)
    
    def get_rot_matrix(self) -> np.ndarray:
        """获取旋转矩阵"""
        return self.rot_mat.copy()
    
    def set_from_quat(self, quat: np.ndarray):
        """从四元数设置旋转
        
        Args:
            quat: 四元数 [w, x, y, z]
        """
        from scipy.spatial.transform import Rotation
        self.quat = quat / np.linalg.norm(quat)  # 归一化
        rot = Rotation.from_quat(np.roll(self.quat, -1))  # [w,x,y,z] -> [x,y,z,w]
        self.rot_mat = rot.as_matrix()
        self.euler = rot.as_euler('xyz')


@dataclass
class QuadrupedState:
    """四足机器人完整状态容器"""
    
    # ========== 四条腿的状态 ==========
    FL: LegJointMap = field(default_factory=lambda: LegJointMap("FL"))
    FR: LegJointMap = field(default_factory=lambda: LegJointMap("FR"))
    RL: LegJointMap = field(default_factory=lambda: LegJointMap("RL"))
    RR: LegJointMap = field(default_factory=lambda: LegJointMap("RR"))
    
    # ========== 基座状态 ==========
    base: BaseState = field(default_factory=BaseState)
    
    # ========== 时间戳 ==========
    time: float = 0.0                          # 仿真时间 [s]
    step_num: int = 0                          # 步数
    
    # ========== 全局状态数组 ==========
    qpos: np.ndarray = None                    # 完整关节位置 (nq,)
    qvel: np.ndarray = None                    # 完整关节速度 (nv,)
    tau_ctrl: np.ndarray = None                # 完整控制力矩 (nu,)
    
    def get_legs(self) -> Dict[str, LegJointMap]:
        """获取所有腿的字典"""
        return {'FL': self.FL, 'FR': self.FR, 'RL': self.RL, 'RR': self.RR}
    
    def get_leg_by_name(self, name: str) -> LegJointMap:
        """通过名称获取腿"""
        return getattr(self, name)
    
    def __getitem__(self, key: str) -> LegJointMap:
        """支持字典式访问，如 state['FL']"""
        return self.get_leg_by_name(key)
    
    def get_feet_pos(self, frame: str = 'base') -> Dict[str, np.ndarray]:
        """获取所有足端位置"""
        if frame == 'base':
            return {
                'FL': self.FL.foot_pos.copy(),
                'FR': self.FR.foot_pos.copy(),
                'RL': self.RL.foot_pos.copy(),
                'RR': self.RR.foot_pos.copy(),
            }
        elif frame == 'world':
            return {
                'FL': self.FL.foot_pos_world.copy(),
                'FR': self.FR.foot_pos_world.copy(),
                'RL': self.RL.foot_pos_world.copy(),
                'RR': self.RR.foot_pos_world.copy(),
            }
        else:
            raise ValueError(f"Invalid frame: {frame}")
    
    def get_feet_vel(self, frame: str = 'base') -> Dict[str, np.ndarray]:
        """获取所有足端速度"""
        if frame == 'base':
            return {
                'FL': self.FL.foot_vel.copy(),
                'FR': self.FR.foot_vel.copy(),
                'RL': self.RL.foot_vel.copy(),
                'RR': self.RR.foot_vel.copy(),
            }
        elif frame == 'world':
            return {
                'FL': self.FL.foot_vel_world.copy(),
                'FR': self.FR.foot_vel_world.copy(),
                'RL': self.RL.foot_vel_world.copy(),
                'RR': self.RR.foot_vel_world.copy(),
            }
        else:
            raise ValueError(f"Invalid frame: {frame}")
        
    def get_num_contact(self) -> int:
        """计算当前接触的腿数"""
        return int(self.FL.contact_state) + \
            int(self.FR.contact_state) + \
            int(self.RL.contact_state) + \
            int(self.RR.contact_state)
        
    def get_contact_states(self) -> Dict[str, bool]:
        """获取所有接触状态"""
        return {
            'FL': self.FL.contact_state,
            'FR': self.FR.contact_state,
            'RL': self.RL.contact_state,
            'RR': self.RR.contact_state,
        }
    
    def get_contact_forces(self) -> Dict[str, np.ndarray]:
        """获取所有接触力"""
        return {
            'FL': self.FL.contact_force.copy(),
            'FR': self.FR.contact_force.copy(),
            'RL': self.RL.contact_force.copy(),
            'RR': self.RR.contact_force.copy(),
        }
    
    def get_max_feet_dist_to_hip(self) -> float:
        """
        计算四只脚相对于各自髋关节(Hip)在水平面(XY)上的最大偏离距离。
        """
        
        dists = []
        for leg_name in ['FL', 'FR', 'RL', 'RR']:
            leg = getattr(self, leg_name)
            delta_xy = leg.foot_pos[:2] - leg.hip_pos[:2]
            dist = np.linalg.norm(delta_xy)
            dists.append(dist)
            
        return max(dists)
    



@dataclass
class Trajectory:
    """存储一段轨迹数据的容器"""
    
    states: List[QuadrupedState] = field(default_factory=list)  # 状态序列
    times: np.ndarray = None                                     # 时间戳 (T,)
    actions: np.ndarray = None                                   # 控制输入 (T, nu)
    
    def __len__(self) -> int:
        return len(self.states)
    
    def get_state_at(self, idx: int) -> QuadrupedState:
        """获取索引处的状态"""
        return self.states[idx]
    
    def get_feet_positions_history(self, leg: str) -> np.ndarray:
        """获取某条腿的足端位置历史 (T, 3)"""
        return np.array([getattr(state, leg).foot_pos for state in self.states])
    
    def get_contact_sequence(self, leg: str) -> np.ndarray:
        """获取某条腿的接触序列 (T,)"""
        return np.array([getattr(state, leg).contact_state for state in self.states], dtype=bool)


@dataclass
class ReferenceState:
    """参考状态容器，用于将 MPC/控制器的参考信息传入算法
    - `ref_foot_*`: 每条腿的参考足端位置（世界坐标系），形状 (3,)
    - `ref_foot_*_centered`: 每条腿的参考足端位置（中心化坐标系），形状 (3,)
    - `ref_position`: 参考基座位置（世界坐标系）(3,)
    - `ref_position_centered`: 参考基座位置（中心化坐标系）(3,)
    - `ref_linear_velocity`: 参考线速度 (3,)
    - `ref_angular_velocity`: 参考角速度 (3,)
    - `ref_orientation`: 参考姿态 (roll, pitch, yaw) (3,)
    """

    # 世界坐标系下的参考值
    ref_foot_FL: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_foot_FR: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_foot_RL: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_foot_RR: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # 中心化坐标系下的参考值（由 reference_interface 或 env 计算并填充）
    ref_foot_FL_centered: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_foot_FR_centered: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_foot_RL_centered: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_foot_RR_centered: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_position_centered: np.ndarray = field(default_factory=lambda: np.zeros(3))

    ref_linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ref_orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def as_dict(self) -> dict:
        """返回一个便于序列化/传递的字典副本（numpy arrays 保持原样）。"""
        return {
            'ref_foot_FL': self.ref_foot_FL.copy(),
            'ref_foot_FR': self.ref_foot_FR.copy(),
            'ref_foot_RL': self.ref_foot_RL.copy(),
            'ref_foot_RR': self.ref_foot_RR.copy(),
            'ref_linear_velocity': self.ref_linear_velocity.copy(),
            'ref_angular_velocity': self.ref_angular_velocity.copy(),
            'ref_orientation': self.ref_orientation.copy(),
            'ref_position': self.ref_position.copy(),
        }


@dataclass
class RobotConfig:
    """机器人配置参数"""
    
    robot_name: str                            # 机器人名称
    mass: float = 10                           # 总质量 [kg]
    n_legs: int = 4                            # 腿数
    dofs_per_leg: List[int] = field(default_factory=lambda: [3, 3, 3, 3])  # 每条腿的自由度
    
    # ========== 物理参数 ==========
    gravity: float = 9.81                      # 重力加速度 [m/s²]
    friction_coeff: float = 1.0                # 摩擦系数
    inertia: np.ndarray = None                 # 惯性张量 (3, 3) 或 (9,) [kg·m²]
    
    # ========== 运动学参数 ==========
    hip_height: float = 0.3                   # 髋关节高度 [m]
    foot_radius: float = 0.01                  # 足端半径 [m]
    
    # ========== 控制参数 ==========
    swing_kp: float = 500.0                    # 摇摆相控制比例系数
    swing_kd: float = 10.0                     # 摇摆相控制微分系数
    step_height: float = 0.06                  # 步高 [m]
    
    # ========== 关节限制 ==========
    joint_limits_qpos: Optional[np.ndarray] = None  # 关节位置限制 (nq, 2)
    joint_limits_qvel: Optional[np.ndarray] = None  # 关节速度限制 (nv, 2)
    joint_limits_tau: Optional[np.ndarray] = None   # 力矩限制 (nu, 2)
    
    def get_total_dof(self) -> int:
        """获取总自由度数（不含基座）"""
        return sum(self.dofs_per_leg)
    
    def get_inertia_matrix(self) -> np.ndarray:
        """获取3x3惯性矩阵
        
        Returns:
            3x3惯性矩阵 [kg·m²]
        """
        if self.inertia is None:
            return np.eye(3, dtype=np.float64)
        
        inertia = self.inertia
        if inertia.size == 9:
            if inertia.shape == (3, 3):
                return inertia.copy()
            else:
                # 扁平化的9个元素 [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]
                return inertia.reshape(3, 3)
        else:
            raise ValueError(f"Invalid inertia shape: {inertia.shape}")






