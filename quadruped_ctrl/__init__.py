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

# Delay or make optional the heavy simulator import so the package can be
# imported on systems without MuJoCo installed. Consumers that need the
# environment should import it explicitly.
try:
    from .quadruped_env import QuadrupedEnv
except Exception:
    QuadrupedEnv = None

__all__ = [
    'LegJointMap',
    'BaseState',
    'QuadrupedState',
    'Trajectory',
    'RobotConfig',
    'ConfigLoader',
    # QuadrupedEnv may be None if MuJoCo is unavailable; import explicitly
    'QuadrupedEnv',
]
