from __future__ import annotations

import mujoco
import numpy as np
from mujoco.viewer import Handle
from typing import Optional, List, Dict

def render_sphere(viewer:Handle, position, diameter=0.05, color=None, geom_id=-1):
    """在仿真中渲染一个球体"""
    if viewer is None: return -1
    color = color if color is not None else np.array([1, 0, 0, 1])
    
    if geom_id < 0:
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1
        
    geom = viewer.user_scn.geoms[geom_id]
    mujoco.mjv_initGeom(geom, 
                        type=mujoco.mjtGeom.mjGEOM_SPHERE, 
                        size=[diameter/2, 0, 0], 
                        pos=position, 
                        mat=np.eye(3).flatten(),
                        rgba=color)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1 
    geom.objid = -1
    return geom_id


def render_line(viewer, initial_point, target_point, width=0.01, color=None, geom_id=-1):
    """在两点之间渲染一条线段（使用胶囊体实现）"""
    if viewer is None: return -1
    color = color if color is not None else np.array([1, 1, 0, 1])
    
    # 计算线段方向和长度
    d = np.asarray(target_point) - np.asarray(initial_point)
    length = np.linalg.norm(d)
    if length < 1e-6: return geom_id
    
    # 计算旋转矩阵将 Z 轴对齐到向量方向
    z_axis = d / length
    x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    mat = np.column_stack((x_axis, y_axis, z_axis)).flatten()

    if geom_id < 0:
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1
        
    geom = viewer.user_scn.geoms[geom_id]
    mujoco.mjv_initGeom(geom, 
                        type=mujoco.mjtGeom.mjGEOM_CAPSULE, 
                        size=[width, width, length/2], 
                        pos=(np.asarray(initial_point) + np.asarray(target_point))/2, 
                        mat=mat, 
                        rgba=color)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1
    return geom_id

def render_vector(viewer, vector, pos, scale=1.0, color=None, geom_id=-1):
    """渲染一个箭头向量"""
    if viewer is None: return -1
    color = color if color is not None else np.array([1, 0, 0, 1])
    v_norm = np.linalg.norm(vector)
    if v_norm < 1e-6: return geom_id
    
    target_pos = np.asarray(pos) + np.asarray(vector) * scale
    
    if geom_id < 0:
        viewer.user_scn.ngeom += 1
        geom_id = viewer.user_scn.ngeom - 1
        
    geom = viewer.user_scn.geoms[geom_id]
    # MuJoCo 的 mjGEOM_ARROW 默认朝向 Z 轴
    z_axis = np.asarray(vector) / v_norm
    x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    mat = np.column_stack((x_axis, y_axis, z_axis)).flatten()

    mujoco.mjv_initGeom(geom, 
                        type=mujoco.mjtGeom.mjGEOM_ARROW, size=[0.01, 0.01, v_norm*scale/2], 
                        pos=pos, 
                        mat=mat, 
                        rgba=color)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1
    return geom_id

def plot_swing_trajectory(
    viewer: Handle,
    swing_traj_controller,
    swing_period: float,
    swing_time: Dict[str, float],
    lift_off_positions: Dict[str, np.ndarray],
    nmpc_footholds: Dict[str, np.ndarray],
    ref_feet_pos: Dict[str, np.ndarray],
    early_stance_detector,
    geom_ids: Dict[str, List[int]] = None,
) -> Dict[str, List[int]]:
    
    NUM_TRAJ_POINTS = 6 # 5段线 + 1个球
    legs = ["FL", "FR", "RL", "RR"]

    # 1. 初始化 ID (仅在第一次调用或重置时执行)
    if geom_ids is None:
        geom_ids = {leg: [-1] * NUM_TRAJ_POINTS for leg in legs}

    if viewer is None:
        return geom_ids

    for leg_name in legs:
        # 2. 如果当前腿不在摆动，把之前的几何体“藏”起来 (设为全透明)
        if swing_time.get(leg_name, 0.0) == 0.0:
            for g_id in geom_ids[leg_name]:
                if g_id != -1:
                    viewer.user_scn.geoms[g_id].rgba[3] = 0.0 
            continue
            
        # 3. 计算轨迹点
        traj_points = []
        # 计算从当前时刻到摆动结束的预测路径
        times = np.linspace(swing_time[leg_name], swing_period, NUM_TRAJ_POINTS)
        for t in times:
            ref_foot_pos, _, _ = swing_traj_controller.swing_generator.compute_trajectory_references(
                t, lift_off_positions[leg_name], nmpc_footholds[leg_name], 
                early_stance_detector.hitmoments[leg_name], early_stance_detector.hitpoints[leg_name]
            )
            traj_points.append(ref_foot_pos.squeeze())

        # 4. 渲染线段 (红线)
        for i in range(NUM_TRAJ_POINTS - 1):
            geom_ids[leg_name][i] = render_line(
                viewer=viewer,
                initial_point=traj_points[i],
                target_point=traj_points[i + 1],
                width=0.005,
                color=np.array([1, 0, 0, 0.7]), # 稍微带点透明度更好看
                geom_id=geom_ids[leg_name][i],
            )

        # 5. 渲染目标球 (绿球)
        geom_ids[leg_name][-1] = render_sphere(
            viewer=viewer,
            position=ref_feet_pos[leg_name],
            diameter=0.04,
            color=np.array([0, 1, 0, 0.5]),
            geom_id=geom_ids[leg_name][-1],
        )
        
    return geom_ids