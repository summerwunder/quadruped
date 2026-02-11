import os
import sys



class ControllerFactory:
    """
    Factory class to select and initialize different controllers.
    """
    @staticmethod
    def create_controller(name, env, **kwargs):
        name = name.lower()
        
        if name == "pd":
            from quadruped_ctrl.controllers.pd.pd_controller import PDController
            print("[Factory] Creating PD Controller...")
            return PDController(env, **kwargs)
        
        elif name == "mpc_gradient":
            from quadruped_ctrl.controllers.nmpc_gradient.controller_handler import Quadruped_NMPC_Handler
            print("[Factory] Creating Gradient MPC Controller...")
            return Quadruped_NMPC_Handler(env, **kwargs)
            
        else:
            raise ValueError(f"Unknown controller type: {name}")