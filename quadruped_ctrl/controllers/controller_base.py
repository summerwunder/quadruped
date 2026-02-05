from abc import ABC, abstractmethod
import numpy as np
from quadruped_ctrl.datatypes import QuadrupedState

class BaseController(ABC):
    """
    Standard interface for all quadruped controllers.
    """
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def get_action(self, state: QuadrupedState):
        """
        Input: QuadrupedState (The current full state of the robot)
        Output: numpy.ndarray (The joint torques to be applied to actuators)
        """
        pass

    def reset(self):
        """Optional: Reset controller internal state (e.g., integrators, filters)"""
        pass