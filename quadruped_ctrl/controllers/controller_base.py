from abc import ABC, abstractmethod
import numpy as np
from quadruped_ctrl.datatypes import QuadrupedState

class BaseController(ABC):
    """
    Standard interface for all quadruped controllers.
    """
    def __init__(self, env, **kwargs):
        self.env = env
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def get_action(self, **kwargs):
        """
        """
        pass

    def reset(self):
        """Optional: Reset controller internal state (e.g., integrators, filters)"""
        pass