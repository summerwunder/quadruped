"""
Configuration loader for robot configs from YAML files.
"""

from __future__ import annotations

import os
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

from quadruped_ctrl.datatypes import RobotConfig


class ConfigLoader:
    """配置文件加载器"""
    
    @staticmethod
    def load_robot_config(config_path: str) -> RobotConfig:
        """从YAML文件加载机器人配置
        
        Args:
            config_path: 配置文件路径 (绝对路径或相对于config目录)
        Returns:
            RobotConfig对象
        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        # 处理路径
        if not os.path.isabs(config_path):
            # 相对路径：相对于本模块的config目录
            module_dir = os.path.dirname(__file__)
            config_path = os.path.join(module_dir, '..', 'config', config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise ValueError(f"Empty config file: {config_path}")
        
        return ConfigLoader._parse_config(data)
    
    @staticmethod
    def _parse_config(data: Dict[str, Any]) -> RobotConfig:
        """解析YAML数据到RobotConfig
        
        Args:
            data: YAML加载后的字典
        
        Returns:
            RobotConfig对象
        """
        robot_name = data.get('robot_name', 'go1')
        
        # 物理参数
        physics = data.get('physics', {})
        mass = float(physics.get('mass', 12.0))
        gravity = float(physics.get('gravity', 9.81))
        inertia_list = physics.get('inertia', None)
        inertia = None
        if inertia_list is not None:
            inertia = np.array(inertia_list, dtype=np.float64)
        
        # 几何参数
        geometry = data.get('geometry', {})
        hip_height = float(geometry.get('hip_height', 0.25))
        
        # 控制参数
        swing_control = data.get('swing_control', {})
        swing_kp = float(swing_control.get('kp', 60.0))
        swing_kd = float(swing_control.get('kd', 10.0))
        step_height = float(swing_control.get('step_height', 0.05))
        
        # 创建配置对象
        config = RobotConfig(
            robot_name=robot_name,
            total_mass=mass,
            n_legs=4,
            dofs_per_leg=[3, 3, 3, 3],
            gravity=gravity,
            friction_coeff=1.0,
            inertia=inertia,
            hip_height=hip_height,
            foot_radius=0.01,
            swing_kp=swing_kp,
            swing_kd=swing_kd,
            step_height=step_height,
        )
        
        return config
    
    @staticmethod
    def load_builtin_config(robot_name: str) -> RobotConfig:
        """加载内置机器人配置
        
        Args:
            robot_name: 机器人名称 ('go1', 'go2', 等)
        Returns:
            RobotConfig对象       
        Raises:
            ValueError: 不支持的机器人名称
        """
        supported_robots = {
            'go1': 'robot/go1.yaml',
            'go2': 'robot/go2.yaml',
        }
        
        if robot_name not in supported_robots:
            raise ValueError(
                f"Unsupported robot: {robot_name}. "
                f"Supported: {list(supported_robots.keys())}"
            )
        
        config_file = supported_robots[robot_name]
        return ConfigLoader.load_robot_config(config_file)
    
    @staticmethod
    def load_sim_config(config_path: str = 'sim_config.yaml') -> Dict[str, Any]:
        """加载仿真配置
        
        Args:
            config_path: 仿真配置文件路径 (相对于config目录)
            
        Returns:
            仿真配置字典，包含:
            {
                'optimize': {'use_feedback_linearization': bool, 'use_friction_compensation': bool},
                'physics': {'dt': float, 'mpc_frequency': int, 'scene': str},
                'gait': {...}
            }
        """
        if not os.path.isabs(config_path):
            module_dir = os.path.dirname(__file__)
            config_path = os.path.join(module_dir, '..', 'config', config_path)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Sim config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            sim_config = yaml.safe_load(f)
        
        if sim_config is None:
            raise ValueError(f"Empty sim config file: {config_path}")
        
        return sim_config
