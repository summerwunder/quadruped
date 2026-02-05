import os
import sys
from quadruped_ctrl.controllers.pd.pd_controller import PDController


class ControllerFactory:
    """
    Factory class to select and initialize different controllers.
    """
    @staticmethod
    def create_controller(name, model, **kwargs):
        name = name.lower()
        
        if name == "pd":
            print("[Factory] Creating PD Controller...")
            return PDController(model, **kwargs)
        
        elif name == "mpc_gradient":
            print("[Factory] Creating Gradient MPC Controller...")
            pass 
            
        else:
            raise ValueError(f"Unknown controller type: {name}")