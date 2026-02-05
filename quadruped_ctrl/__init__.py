"""
Quadruped control and MPC module.
Independent data structures and controllers for quadruped robots.
"""

__version__ = "0.1.0"

from .datatypes import (
    LegJointMap,
    BaseState,
    QuadrupedState,
    Trajectory,
    RobotConfig,
)

from .utils.config_loader import ConfigLoader

from .quadruped_env import QuadrupedEnv

__all__ = [
    'LegJointMap',
    'BaseState',
    'QuadrupedState',
    'Trajectory',
    'RobotConfig',
    'ConfigLoader',
    'QuadrupedEnv',
]
