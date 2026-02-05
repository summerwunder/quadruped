import casadi as cs
import numpy as np
from quadruped_ctrl.utils.config_loader import ConfigLoader
from acados_template import AcadosModel


class QuadrupedModel:
    """四足机器人动力学模型，基于CasADi实现符号计算"""
    def __init__(self):
        sim_config = ConfigLoader.load_sim_config('sim_config.yaml')
        self.gravity = sim_config.get('physics', {}).get('gravity', 9.81)
        self.foot_optimization = sim_config.get('optimize', {}).get('use_foothold_optimization', False)

        body_position_x = cs.SX.sym('position_x')
        body_position_y = cs.SX.sym('position_y')
        body_position_z = cs.SX.sym('position_z')
        body_velocity_x = cs.SX.sym('velocity_x')
        body_velocity_y = cs.SX.sym('velocity_y')
        body_velocity_z = cs.SX.sym('velocity_z')

        roll = cs.SX.sym('roll')
        pitch = cs.SX.sym('pitch')
        yaw = cs.SX.sym('yaw')
        angular_velocity_x = cs.SX.sym('angular_velocity_x')
        angular_velocity_y = cs.SX.sym('angular_velocity_y')
        angular_velocity_z = cs.SX.sym('angular_velocity_z')

        foot_position_FL = cs.SX.sym('foot_position_FL', 3)
        foot_position_FR = cs.SX.sym('foot_position_FR', 3)
        foot_position_RL = cs.SX.sym('foot_position_RL', 3)
        foot_position_RR = cs.SX.sym('foot_position_RR', 3)

        body_position_z_integral = cs.SX.sym('body_position_z_integral')
        body_velocity_x_integral = cs.SX.sym('body_velocity_x_integral')
        body_velocity_y_integral = cs.SX.sym('body_velocity_y_integral')
        body_velocity_z_integral = cs.SX.sym('body_velocity_z_integral')
        roll_integral = cs.SX.sym('roll_integral')
        pitch_integral = cs.SX.sym('pitch_integral')

        # angular_velocity_x_integral = cs.SX.sym('angular_velocity_x_integral')
        # angular_velocity_y_integral = cs.SX.sym('angular_velocity_y_integral')
        # angular_velocity_z_integral = cs.SX.sym('angular_velocity_z_integral')

        self.state = cs.vertcat(
            body_position_x, body_position_y, body_position_z,
            body_velocity_x, body_velocity_y, body_velocity_z,
            roll, pitch, yaw,
            angular_velocity_x, angular_velocity_y, angular_velocity_z,
            foot_position_FL, foot_position_FR, foot_position_RL, foot_position_RR,
            body_position_z_integral,
            body_velocity_x_integral, body_velocity_y_integral, body_velocity_z_integral,
            roll_integral, pitch_integral,
            # angular_velocity_x_integral, angular_velocity_y_integral, angular_velocity_z_integral
        )

        self.state_dot = cs.vertcat(
            cs.SX.sym('linear_body_vel',3, 1),
            cs.SX.sym('linear_body_acc',3, 1),
            cs.SX.sym('angular_body_vel',3, 1),
            cs.SX.sym('angular_body_acc',3, 1),
            cs.SX.sym('foot_vel_FL',3, 1),
            cs.SX.sym('foot_vel_FR',3, 1),
            cs.SX.sym('foot_vel_RL',3, 1),
            cs.SX.sym('foot_vel_RR',3, 1),
            cs.SX.sym('linear_body_vel_z_integral',1, 1),
            cs.SX.sym('linear_body_acc_integral',3, 1),
            cs.SX.sym('angular_vel_roll_integral',1, 1),
            cs.SX.sym('angular_vel_pitch_integral',1, 1),
        )
        
        foot_velocity_FL = cs.SX.sym('foot_velocity_FL', 3)
        foot_velocity_FR = cs.SX.sym('foot_velocity_FR', 3)
        foot_velocity_RL = cs.SX.sym('foot_velocity_RL', 3)
        foot_velocity_RR = cs.SX.sym('foot_velocity_RR', 3)
        foot_force_FL = cs.SX.sym('foot_force_FL', 3)
        foot_force_FR = cs.SX.sym('foot_force_FR', 3)
        foot_force_RL = cs.SX.sym('foot_force_RL', 3)
        foot_force_RR = cs.SX.sym('foot_force_RR', 3)

        self.input = cs.vertcat(
            foot_velocity_FL, foot_velocity_FR, foot_velocity_RL, foot_velocity_RR,
            foot_force_FL, foot_force_FR, foot_force_RL, foot_force_RR
        )

        self.y_ref = cs.vertcat(
            self.state, self.input
        )

        # define acados params which can be changed online
        self.stanceFL = cs.SX.sym('stanceFL')
        self.stanceFR = cs.SX.sym('stanceFR')
        self.stanceRL = cs.SX.sym('stanceRL')
        self.stanceRR = cs.SX.sym('stanceRR')
        self.stance_params = cs.vertcat(
            self.stanceFL, self.stanceFR, self.stanceRL, self.stanceRR
        )

        self.mu_friction = cs.SX.sym('mu_friction')
        self.stance_proximity = cs.SX.sym('stance_proximity',4,1)
        self.base_position = cs.SX.sym('base_position',3,1)
        self.base_yaw = cs.SX.sym('base_yaw')
        self.external_wrench = cs.SX.sym('external_forces',6,1)

        self.inertia = cs.SX.sym('inertia',9,1)
        self.mass = cs.SX.sym('mass')

        params = cs.vertcat(
            self.stance_params,
            self.mu_friction,
            self.stance_proximity,
            self.base_position,
            self.base_yaw,
            self.external_wrench,
            self.inertia,
            self.mass
        )

        xdot = self.forward_dynamics(self.state, self.input, params)
        self.fun_forward_dynamics = cs.Function("fun_forward_dynamics", [self.state, self.input, params], [xdot])

    
    def forward_dynamics(self, state:np.ndarray, input:np.ndarray, params:np.ndarray) -> np.ndarray:
        """前向动力学计算，返回状态导数"""
        foot_vel_FL = input[0:3]
        foot_vel_FR = input[3:6]
        foot_vel_RL = input[6:9]
        foot_vel_RR = input[9:12]
        foot_force_FL = input[12:15]
        foot_force_FR = input[15:18]
        foot_force_RL = input[18:21]
        foot_force_RR = input[21:24]

        body_position = state[0:3]
        body_velocity = state[3:6]
        body_orientation = state[6:9]
        body_angular_velocity = state[9:12]
        foot_position_FL = state[12:15]
        foot_position_FR = state[15:18]
        foot_position_RL = state[18:21]
        foot_position_RR = state[21:24]

        stanceFL, stanceFR, stanceRL, stanceRR = params[0:4]
        stance_proximity_FL, stance_proximity_FR, stance_proximity_RL, stance_proximity_RR = params[5:9]
        mu_friction = params[4]
        mass = params[-1]
        inertia = params[-10:-1].reshape((3,3))
        external_wrench_linear = params[13:16]
        external_wrench_angular = params[16:19]

        # state 1 : velocity of body  world frame 
        state_dot_body_position = body_velocity
        # state 2 : acceleration of body  world frame 
        force_total = foot_force_FL @ stanceFL +\
                    foot_force_FR @ stanceFR +\
                    foot_force_RL @ stanceRL +\
                    foot_force_RR @ stanceRR +\
                    external_wrench_linear
        gravity = np.array([0, 0, -self.gravity])
        state_dot_body_velocity = gravity + (1.0 / mass) * force_total
        # state 3 : angular velocity of body
        omega_rotation_matrix = cs.SX.zeros(3,3)
        roll, pitch, yaw = body_orientation
        omega_rotation_matrix[0,0] = 1
        omega_rotation_matrix[0,2] = -cs.sin(pitch)
        omega_rotation_matrix[1,1] = cs.cos(roll)
        omega_rotation_matrix[1,2] = cs.sin(roll)*cs.cos(pitch)
        omega_rotation_matrix[2,1] = -cs.sin(roll)
        omega_rotation_matrix[2,2] = cs.cos(roll)*cs.cos(pitch)

        state_dot_body_orientation = cs.inv(omega_rotation_matrix) @ body_angular_velocity

        # state 4 : angular acceleration of body
        torque = cs.skew(foot_position_FL - body_position) @ (foot_force_FL * stanceFL) +\
                 cs.skew(foot_position_FR - body_position) @ (foot_force_FR * stanceFR) +\
                 cs.skew(foot_position_RL - body_position) @ (foot_force_RL * stanceRL) +\
                 cs.skew(foot_position_RR - body_position) @ (foot_force_RR * stanceRR) +\
                 external_wrench_angular
        
        Rx = cs.SX.zeros(3,3)
        Rx[0,0] = 1
        Rx[1,1] = cs.cos(roll)
        Rx[1,2] = -cs.sin(roll)
        Rx[2,1] = cs.sin(roll)
        Rx[2,2] = cs.cos(roll)
        Ry = cs.SX.zeros(3,3)
        Ry[0,0] = cs.cos(pitch)
        Ry[0,2] = cs.sin(pitch)
        Ry[1,1] = 1
        Ry[2,0] = -cs.sin(pitch)
        Ry[2,2] = cs.cos(pitch)
        Rz = cs.SX.zeros(3,3)
        Rz[0,0] = cs.cos(yaw)
        Rz[0,1] = -cs.sin(yaw)
        Rz[1,0] = cs.sin(yaw)
        Rz[1,1] = cs.cos(yaw)
        R_w_b = Rx @ Ry @ Rz

        # base frame angular acceleration
        state_dot_body_angular_velocity = cs.inv(inertia) @ (R_w_b @ torque - cs.skew(body_angular_velocity) @ (inertia @ body_angular_velocity))

        # foot velocity 
        if self.foot_optimization:
            state_dot_foot_position_FL = foot_vel_FL @ (1 - stanceFL) @ (1 - stance_proximity_FL)
            state_dot_foot_position_FR = foot_vel_FR @ (1 - stanceFR) @ (1 - stance_proximity_FR)
            state_dot_foot_position_RL = foot_vel_RL @ (1 - stanceRL) @ (1 - stance_proximity_RL)
            state_dot_foot_position_RR = foot_vel_RR @ (1 - stanceRR) @ (1 - stance_proximity_RR)
        else:
            state_dot_foot_position_FL = foot_vel_FL * 0.0
            state_dot_foot_position_FR = foot_vel_FR * 0.0
            state_dot_foot_position_RL = foot_vel_RL * 0.0
            state_dot_foot_position_RR = foot_vel_RR * 0.0
        
        integral = state[24:]
        integral[0] += state[2]  # body_position_z_integral
        integral[1] += state[3]  # body_velocity_x_integral
        integral[2] += state[4]  # body_velocity_y_integral
        integral[3] += state[5]  # body_velocity_z_integral
        integral[4] += roll
        integral[5] += pitch

        return cs.vertcat(
            state_dot_body_position,
            state_dot_body_velocity,
            state_dot_body_orientation,
            state_dot_body_angular_velocity,
            state_dot_foot_position_FL,
            state_dot_foot_position_FR,
            state_dot_foot_position_RL,
            state_dot_foot_position_RR,
            integral
        )
    
    def export_quadruped_model(self):
        self.param = cs.vertcat(
            self.stance_params,
            self.mu_friction,
            self.stance_proximity,
            self.base_position,
            self.base_yaw,
            self.external_wrench,
            self.inertia,
            self.mass
        )

        f_expl = self.forward_dynamics(self.state, self.input, self.param)
        f_impl = self.state_dot - f_expl

        acados_model = AcadosModel()
        acados_model.f_impl_expr = f_impl
        acados_model.f_expl_expr = f_expl
        acados_model.x = self.state
        acados_model.u = self.input
        acados_model.xdot = self.state_dot
        acados_model.p = self.param
        acados_model.name = 'quadruped_model'
        return acados_model


