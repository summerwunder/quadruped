from acados_template import AcadosOcp, AcadosOcpSolver
from quadruped_ctrl.controllers.gradient.quadruped_model import QuadrupedModel
from quadruped_ctrl.quadruped_env import QuadrupedEnv
import numpy as np
import casadi as cs
from quadruped_ctrl.utils.config_loader import ConfigLoader
from quadruped_ctrl.controllers.gradient.controller_constraint import QuadrupedConstraints
import pathlib
import scipy
import copy

class Quadruped_NMPC_Handler:
    def __init__(self, env: QuadrupedEnv, 
                 robot_config_path: str = "go1.yaml",
                 sim_config_path: str = "sim.yaml",
                 mpc_config_path: str = "mpc_config.yaml"):
        self.env = env
        self.robot = ConfigLoader.load_robot_config(robot_config_path)
        self.sim_config = ConfigLoader.load_sim_config(sim_config_path)
        # load MPC specific config (weights, R, horizon 等)
        self.mpc_config = ConfigLoader.load_mpc_config(mpc_config_path)

        self.use_foothold_constraint = self.sim_config.get('use_foothold_constraint')
        self.use_static_stability = self.sim_config.get('use_static_stability')
        self.use_zmp_stability = self.sim_config.get('use_zmp_stability')
        self.use_stability_constraint = self.use_static_stability or self.use_zmp_stability
        
        # self.grf_max = self.sim_config.get('grf_max')
        self.grf_max = self.robot.mass * \
                self.sim_config.get('physics', {}).get('gravity')
        self.grf_min = self.sim_config.get('grf_min')
        # config parameters: 优先使用 mpc_config 中的 horizon，其次回退到 sim_config
        self.horizon = int(self.mpc_config.get('horizon', self.sim_config.get('gait', {}).get('horizon', 12)))
        self.dt = float(self.sim_config.get('physics').get('dt', 0.002))
        self.T_horizon = self.horizon * self.dt
        # solver / optimization flags (from mpc_config)
        solver_conf = self.mpc_config.get('solver', {}) if isinstance(self.mpc_config, dict) else {}
        self.use_DDP = bool(solver_conf.get('use_ddp', False))
        self.use_RTI = bool(solver_conf.get('use_rti', False))
        self.num_qp_iterations = int(solver_conf.get('num_qp_iterations', 10))
        self.as_rti_type = str(solver_conf.get('as_rti_type', 'AS-RTI-A'))
        self.solver_mode = str(solver_conf.get('solver_mode', 'balance'))
        self.qp_solver_iter_max = int(solver_conf.get('qp_solver_iter_max', 10))
        # nonuniform discretization
        self.use_nonuniform_discretization = bool(solver_conf.get('use_nonuniform_discretization', False))
        self.dt_fine_grained = float(solver_conf.get('dt_fine_grained', self.dt))
        self.horizon_fine_grained = int(solver_conf.get('horizon_fine_grained', 0))
        
        self.quadruped_model = QuadrupedModel(sim_config_path)
        acados_model = self.quadruped_model.export_quadruped_model()
        self.state_dim = acados_model.x.size()[0]
        self.control_dim = acados_model.u.size()[0]
        
        # OCP_Cost
        self.ocp = AcadosOcp()
        self.setup_ocp_cost(acados_model)
        # Constaints
        quadruped_constraints = QuadrupedConstraints(
            self.quadruped_model, self.use_static_stability, acados_infty=1e6
        )
        self.setup_ocp_constraints(quadruped_constraints)
        # initialize parameters
        self.setup_ocp_initialize_params()
        # OCP option
        self.setup_ocp_options()
        
        
        current_dir = pathlib.Path(__file__).parent.resolve()
        code_export_dir = current_dir / "acados_code_export"
        self.ocp.code_export_directory = str(code_export_dir)
        
    def setup_ocp_cost(self, acados_model=None):
        self.ocp.model = acados_model
        nx = self.state_dim
        nu = self.control_dim
        ny = nx + nu
        
        Q_mat, R_mat = self._set_weight_by_config()
        self.ocp.dims.N = self.horizon
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"
        self.ocp.cost.W_e = Q_mat
        self.ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vx[:nx,:nx] = np.eye(nx)
        self.ocp.cost.Vu[nx:nx+nu,:nu] = np.eye(nu)
        self.ocp.cost.Vx_e = np.eye(nx)

    def setup_ocp_constraints(self, quadruped_constraints: QuadrupedConstraints):
        # friction cone
        Jbu, lb, ub = quadruped_constraints.get_friction_cone_bounds(self.grf_min, self.grf_max)
        self.ocp.model.con_h_expr = Jbu
        self.ocp.constraints.uh = ub
        self.ocp.constraints.lh = lb
        self.ocp.model.con_h_expr_0 = Jbu
        self.ocp.constraints.uh_0 = ub
        self.ocp.constraints.lh_0 = lb
        nsh = Jbu.shape[0]
        nsh_state_constraint_start = copy.copy(nsh)
        if self.use_foothold_constraint:
            Jbu_foothold, lb_foothold, ub_foothold = quadruped_constraints.get_foothold_bounds()
            self.ocp.model.con_h_expr = cs.vertcat(self.ocp.model.con_h_expr, Jbu_foothold)
            self.ocp.constraints.uh = np.concatenate((self.ocp.constraints.uh, ub_foothold))
            self.ocp.constraints.lh = np.concatenate((self.ocp.constraints.lh, lb_foothold))
            nsh += Jbu_foothold.shape[0]
        if self.use_stability_constraint:
            Jbu_stability, lb_stability, ub_stability = quadruped_constraints.get_stability_bounds()
            self.ocp.model.con_h_expr = cs.vertcat(self.ocp.model.con_h_expr, Jbu_stability)
            self.ocp.constraints.uh = np.concatenate((self.ocp.constraints.uh, ub_stability))
            self.ocp.constraints.lh = np.concatenate((self.ocp.constraints.lh, lb_stability))
            nsh += Jbu_stability.shape[0]
        nsh_state_constraint_end = copy.copy(nsh)
        num_state_constraints = nsh_state_constraint_end - nsh_state_constraint_start
        if num_state_constraints > 0:
            # (Lower/Upper Bound of Soft H）
            self.ocp.constraints.lsh = np.zeros((num_state_constraints,))
            self.ocp.constraints.ush = np.zeros((num_state_constraints,))
            self.ocp.constraints.idxsh = np.array(
                range(nsh_state_constraint_start, nsh_state_constraint_end)
            )
            ns = num_state_constraints
            # linear and quadratic soft constraints weights
            self.ocp.cost.zl = 1000 * np.ones((ns,))
            self.ocp.cost.Zl = 1 * np.ones((ns,))
            self.ocp.cost.zu = 1000 * np.ones((ns,))
            self.ocp.cost.Zu = 1 * np.ones((ns,))
        list_upper_bound = []
        list_lower_bound = []
        for _ in range(self.horizon):
            list_upper_bound.append(np.zeros(shape=(nsh,)))
            list_lower_bound.append(np.zeros(shape=(nsh,)))
        self.upper_bound = np.array(list_upper_bound, dtype=object)
        self.lower_bound = np.array(list_lower_bound, dtype=object)
        # Set initial state constraint
        X0 = np.zeros(shape=(self.states_dim,))
        self.ocp.constraints.x0 = X0

    def setup_ocp_initialize_params(self):
        init_contact_sequence = np.array([1, 1, 1, 1])
        init_mu = self.sim_config.get("physics", {}).get("mu", 0.5)
        init_stance_proximity = np.array([0.0, 0.0, 0.0, 0.0])
        init_base_position = np.array([0.0, 0.0, 0.0])
        init_base_yaw = np.array([0.0])
        init_external_wrench = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        init_inertia = self.robot.inertia.reshape(9,)
        init_mass = np.array([self.robot.mass])
        self.ocp.parameter_values = np.concatenate(
            (
                init_contact_sequence,
                init_mu,
                init_stance_proximity,
                init_base_position,
                init_base_yaw,
                init_external_wrench,
                init_inertia,
                init_mass,
            )
        )

    def setup_ocp_options(self):
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.hessian_approx = "GAUSS_NEWTON"
        self.ocp.integrator_type = "ERK"
        if self.use_DDP:
            self.ocp.solver_options.nlp_solver_type = 'DDP'
            self.ocp.solver_options.nlp_solver_max_iter = self.num_qp_iterations
            # self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
            self.ocp.solver_options.with_adaptive_levenberg_marquardt = True

            self.ocp.cost.cost_type = "NONLINEAR_LS"
            self.ocp.cost.cost_type_e = "NONLINEAR_LS"
            self.ocp.model.cost_y_expr = cs.vertcat(self.ocp.model.x, self.ocp.model.u)
            self.ocp.model.cost_y_expr_e = self.ocp.model.x

            self.ocp.translate_to_feasibility_problem(keep_x0=True, keep_cost=True)

        elif self.use_RTI:
            self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
            self.ocp.solver_options.nlp_solver_max_iter = 1
            # Set the RTI type for the advanced RTI method
            # (see https://arxiv.org/pdf/2403.07101.pdf)
            if self.as_rti_type == "AS-RTI-A":
                self.ocp.solver_options.as_rti_iter = 1
                self.ocp.solver_options.as_rti_level = 0
            elif self.as_rti_type == "AS-RTI-B":
                self.ocp.solver_options.as_rti_iter = 1
                self.ocp.solver_options.as_rti_level = 1
            elif self.as_rti_type == "AS-RTI-C":
                self.ocp.solver_options.as_rti_iter = 1
                self.ocp.solver_options.as_rti_level = 2
            elif self.as_rti_type == "AS-RTI-D":
                self.ocp.solver_options.as_rti_iter = 1
                self.ocp.solver_options.as_rti_level = 3

        else:
            self.ocp.solver_options.nlp_solver_type = "SQP"
            self.ocp.solver_options.nlp_solver_max_iter = self.num_qp_iterations
        # self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"  # FIXED_STEP, MERIT_BACKTRACKING

        if self.solver_mode == "balance":
            self.ocp.solver_options.hpipm_mode = "BALANCE"
        elif self.solver_mode == "robust":
            self.ocp.solver_options.hpipm_mode = "ROBUST"
        elif self.solver_mode == "fast":
            self.ocp.solver_options.qp_solver_iter_max = self.qp_solver_iter_max
            self.ocp.solver_options.hpipm_mode = "SPEED"
        elif self.solver_mode == "crazy_speed":
            self.ocp.solver_options.qp_solver_iter_max = int(self.qp_solver_iter_max / 2)
            self.ocp.solver_options.hpipm_mode = "SPEED_ABS"
        
        self.ocp.solver_options.levenberg_marquardt = 1e-3

        # Set prediction horizon
        self.ocp.solver_options.tf = self.T_horizon
        
        # Nonuniform discretization (configurable via mpc_config -> solver)
        if self.use_nonuniform_discretization and self.horizon_fine_grained > 0:
            time_steps_fine_grained = np.tile(self.dt_fine_grained, self.horizon_fine_grained)
            time_steps = np.concatenate((time_steps_fine_grained, np.tile(self.dt, self.horizon - self.horizon_fine_grained)))
            shooting_nodes = np.zeros((self.horizon + 1,))
            for i in range(len(time_steps)):
                shooting_nodes[i + 1] = shooting_nodes[i] + time_steps[i]
            self.ocp.solver_options.shooting_nodes = shooting_nodes
        
    def _set_weight_by_config(self)-> tuple[np.ndarray, np.ndarray]:
        """设置MPC的权重矩阵Q和R
            Q_mat (np.ndarray), R_mat (np.ndarray)
        """

        weights = self.mpc_config.get('weights', {}) if isinstance(self.mpc_config, dict) else {}
        R_conf = self.mpc_config.get('R', {}) if isinstance(self.mpc_config, dict) else {}

        def pick(name):
            if name in weights:
                return np.array(weights[name], dtype=np.float64)
            raise KeyError

        Q_position = pick('Q_position')
        Q_velocity = pick('Q_velocity')
        Q_base_angle = pick('Q_base_angle')
        Q_base_angle_rates = pick('Q_base_angle_rates')
        Q_foot_pos = pick('Q_foot_pos')  
        Q_com_position_z_integral = pick('Q_com_position_z_integral')
        Q_com_velocity_integral = pick('Q_com_velocity_integral')
        Q_roll_integral_integral = pick('Q_roll_integral_integral')
        Q_pitch_integral_integral = pick('Q_pitch_integral_integral')

        R_foot_vel = R_conf.get('R_foot_vel')
        R_foot_force = R_conf.get('R_foot_force')
        # build Q and R matrices, repeating foot terms 4 times
        Q_list = np.concatenate(
            (
                Q_position,
                Q_velocity,
                Q_base_angle,
                Q_base_angle_rates,
                Q_foot_pos,
                Q_foot_pos,
                Q_foot_pos,
                Q_foot_pos,
                Q_com_position_z_integral,
                Q_com_velocity_integral,
                Q_roll_integral_integral,
                Q_pitch_integral_integral,
            )
        )

        Q_mat = np.diag(Q_list)
        R_list = np.concatenate(
            (
                R_foot_vel,
                R_foot_vel,
                R_foot_vel,
                R_foot_vel,
                R_foot_force,
                R_foot_force,
                R_foot_force,
                R_foot_force,
            )
        )
        R_mat = np.diag(R_list)
        return Q_mat, R_mat


    
