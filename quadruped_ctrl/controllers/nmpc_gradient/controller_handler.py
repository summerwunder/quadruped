from acados_template import AcadosOcp, AcadosOcpSolver
from quadruped_ctrl.controllers.nmpc_gradient.quadruped_model import QuadrupedModel
from quadruped_ctrl.quadruped_env import QuadrupedEnv
import numpy as np
import casadi as cs
from quadruped_ctrl.utils.config_loader import ConfigLoader
from quadruped_ctrl.controllers.controller_base import BaseController
from quadruped_ctrl.datatypes import QuadrupedState, ReferenceState
from quadruped_ctrl.controllers.nmpc_gradient.controller_constraint import QuadrupedConstraints
import pathlib
import scipy
import os 
import copy

ACADOS_INFTY = 1e6
class Quadruped_NMPC_Handler(BaseController):
    def __init__(self, env: QuadrupedEnv, 
                 mpc_config_path: str = "mpc_config.yaml"):
        self.env = env
        self.robot = env.robot
        self.sim_config = env.sim_config
        
        # load MPC specific config (weights, R, horizon 等)
        self.mpc_config = ConfigLoader.load_mpc_config(mpc_config_path)
        self.verbose = self.mpc_config.get("verbose", False)
        
        # config parameters: horizon 必须先定义
        self.horizon = int(self.mpc_config.get('horizon'))
        self.dt = float(self.sim_config.get('physics').get('dt', 0.002))
        self.T_horizon = self.horizon * self.dt
        # self.grf_max = self.sim_config.get('grf_max')
        self.gravity = self.sim_config.get('physics', {}).get('gravity', 9.81)
        self.grf_max = self.robot.mass * self.gravity
        self.grf_min = self.mpc_config.get('grf_min')
        
        self.previous_status = -1
        self.previous_contact_sequence = np.zeros((4, self.horizon))
        f_z_init = (self.robot.mass * self.gravity) / 4.0
        self.previous_optimal_GRF = np.array([0.0, 0.0, f_z_init] * 4)

        self.sim_optimize_config = self.sim_config.get("optimize", {})
        self.use_foothold_constraint = self.sim_optimize_config.get('use_foothold_constraint')
        self.use_static_stability = self.sim_optimize_config.get('use_static_stability')
        self.use_zmp_stability = self.sim_optimize_config.get('use_zmp_stability')
        self.use_warm_start = self.sim_optimize_config.get('use_warm_start', False)
        self.use_stability_constraint = self.use_static_stability or self.use_zmp_stability
        

        # solver / optimization flags (from mpc_config)
        solver_conf = self.mpc_config.get('solver', {}) if isinstance(self.mpc_config, dict) else {}
        self.use_DDP = bool(solver_conf.get('use_ddp', False))
        self.use_RTI = bool(solver_conf.get('use_rti', False))
        self.num_qp_iterations = int(solver_conf.get('num_qp_iterations', 10))
        self.as_rti_type = str(solver_conf.get('as_rti_type', 'AS-RTI-A'))
        self.solver_mode = str(solver_conf.get('solver_mode', 'balance'))
        self.qp_solver_iter_max = int(solver_conf.get('qp_solver_iter_max', 10))

        self.use_integrators = bool(solver_conf.get('use_integrators', False))
        self.alpha_integrator = float(solver_conf.get('alpha_integrator', 0.1))
        self.integrator_clip = np.array(solver_conf.get('integrator_clip', [0.5, 0.2, 0.2, 0.0, 0.0, 1.0]))
        self.integral_errors = np.zeros(6)
        
        # nonuniform discretization
        self.use_nonuniform_discretization = bool(solver_conf.get('use_nonuniform_discretization', False))
        self.dt_fine_grained = float(solver_conf.get('dt_fine_grained', self.dt))
        self.horizon_fine_grained = int(solver_conf.get('horizon_fine_grained', 0))
        
        self.quadruped_model = QuadrupedModel(env.sim_config_path)
        acados_model = self.quadruped_model.export_quadruped_model()
        self.state_dim = acados_model.x.size()[0]
        self.control_dim = acados_model.u.size()[0]
        self.initial_base_pos = np.array([0.0, 0.0, 0.0])
        # OCP_Cost
        self.ocp = AcadosOcp()
        self.ocp.model = acados_model
        self._setup_ocp_cost()
        # Constaints
        self.quadruped_constraints = QuadrupedConstraints(
            self.quadruped_model, self.use_static_stability, acados_infty=ACADOS_INFTY
        )
        self.constr_lh_friction = None
        self.constr_uh_friction = None
        self.upper_bound = None
        self.lower_bound = None
        self._setup_ocp_constraints(self.quadruped_constraints)
        # initialize parameters
        self._setup_ocp_initialize_params()
        # OCP option
        self._setup_ocp_options()
        code_export_dir = pathlib.Path(__file__).parent.resolve() / "acados_solver"
        self.ocp.code_export_directory = str(code_export_dir)
        
        # 检查是否已经编译过 
        import platform
        lib_ext = "dylib" if platform.system() == "Darwin" else "so"
        lib_path = code_export_dir / f"libacados_ocp_solver_{self.ocp.model.name}.{lib_ext}"
        json_path = code_export_dir / "acados_ocp.json"
        
        # 如果库文件和 JSON 都存在，跳过重新编译
        if lib_path.exists() and json_path.exists():
            self.acados_ocp_solver = AcadosOcpSolver(
                self.ocp, 
                json_file=str(json_path),
                generate=False,
                build=False
            )
        else:
            print(f"[Acados] 首次编译求解器")
            self.acados_ocp_solver = AcadosOcpSolver(
                self.ocp, 
                json_file=str(json_path)
            )
        for stage in range(self.horizon + 1):
            self.acados_ocp_solver.set(stage, "x", np.zeros((self.state_dim,)))
        for stage in range(self.horizon):
            self.acados_ocp_solver.set(stage, "u", np.zeros((self.control_dim,)))


    def get_action(
        self,
        state: QuadrupedState,
        reference: ReferenceState,
        contact_sequence: np.ndarray,    
        constraint= None,
        external_wrenches=np.zeros((6,)),
        inertia=np.zeros((9,)),
        mass = 12,
        mu = 0.5
    ):
        FL_contact_sequence = contact_sequence[0]
        FR_contact_sequence = contact_sequence[1]
        RL_contact_sequence = contact_sequence[2]
        RR_contact_sequence = contact_sequence[3]
        
        
        state, reference = self._perform_state_centering(state, reference)
        for i in range(self.horizon):
            yref = np.zeros((self.state_dim + self.control_dim,))
            # state reference
            yref[0:3] = reference.ref_position
            yref[3:6] = reference.ref_linear_velocity
            yref[6:9] = reference.ref_orientation
            yref[9:12] = reference.ref_angular_velocity
            yref[12:15] = reference.ref_foot_FL
            yref[15:18] = reference.ref_foot_FR
            yref[18:21] = reference.ref_foot_RL
            yref[21:24] = reference.ref_foot_RR
            
            # TODO: ref_foot_*可以尝试传入数组，不然会迟钝
            # It's simply mass*acc/number_of_legs_in_stance!! Force x and y are always 0
            number_of_leg_in_stance = int(
                FL_contact_sequence[i] + FR_contact_sequence[i] + RL_contact_sequence[i] + RR_contact_sequence[i]
            )
            if number_of_leg_in_stance <= 0:
                reference_force_stance = self.grf_max / 4.0
            else:
                reference_force_stance = self.grf_max / number_of_leg_in_stance
            reference_force_FL = reference_force_stance * FL_contact_sequence[i]
            reference_force_FR = reference_force_stance * FR_contact_sequence[i]
            reference_force_RL = reference_force_stance * RL_contact_sequence[i]
            reference_force_RR = reference_force_stance * RR_contact_sequence[i]
            
            # force z
            yref[self.state_dim + 12 + 2] = reference_force_FL
            yref[self.state_dim + 15 + 2] = reference_force_FR
            yref[self.state_dim + 18 + 2] = reference_force_RL
            yref[self.state_dim + 21 + 2] = reference_force_RR
            
            self.acados_ocp_solver.set(i, "yref", yref)
        
        # terminal cost reference
        yref_N = np.zeros(shape = (self.state_dim,))
        yref_N[0:3] = reference.ref_position
        yref_N[3:6] = reference.ref_linear_velocity
        yref_N[6:9] = reference.ref_orientation
        yref_N[9:12] = reference.ref_angular_velocity
        yref_N[12:15] = reference.ref_foot_FL
        yref_N[15:18] = reference.ref_foot_FR
        yref_N[18:21] = reference.ref_foot_RL
        yref_N[21:24] = reference.ref_foot_RR
        self.acados_ocp_solver.set(self.horizon, "yref", yref_N)
        
        # Fill param!!
        base_yaw = state.base.euler[2]
        stance_proximity_FL = np.zeros(self.horizon,)
        stance_proximity_FR = np.zeros(self.horizon,)
        stance_proximity_RL = np.zeros(self.horizon,)
        stance_proximity_RR = np.zeros(self.horizon,)
        
        # TODO: disable 
        for j in range(self.horizon):
                #        if FL_contact_sequence[j] == 0:
                # if (j + 1) < self.horizon:
                #     if FL_contact_sequence[j + 1] == 1:
                #         stance_proximity_FL[j] = 1 * 0
                # if (j + 2) < self.horizon:
                #     if FL_contact_sequence[j + 1] == 0 and FL_contact_sequence[j + 2] == 1:
                #         stance_proximity_FL[j] = 1 * 0
            if FL_contact_sequence[j] == 0:
                if (j + 2) < self.horizon and (FL_contact_sequence[j + 1] == 1 or FL_contact_sequence[j + 2] == 1):
                    stance_proximity_FL[j] = 1 * 0
            if FR_contact_sequence[j] == 0:
                if (j + 2) < self.horizon and (FR_contact_sequence[j + 1] == 1 or FR_contact_sequence[j + 2] == 1):
                    stance_proximity_FR[j] = 1 * 0
            if RL_contact_sequence[j] == 0:
                if (j + 2) < self.horizon and (RL_contact_sequence[j + 1] == 1 or RL_contact_sequence[j + 2] == 1):
                    stance_proximity_RL[j] = 1 * 0
            if RR_contact_sequence[j] == 0:
                if (j + 2) < self.horizon and (RR_contact_sequence[j + 1] == 1 or RR_contact_sequence[j + 2] == 1):
                    stance_proximity_RR[j] = 1 * 0
        
        for j in range(self.horizon):
            # TODO: 暂未设置外部扰相关参数，先设置为0
            external_wrenches_estimated_param = np.zeros((6,))
            param = np.array([
                FL_contact_sequence[j],
                FR_contact_sequence[j],
                RL_contact_sequence[j],
                RR_contact_sequence[j],
                mu,
                stance_proximity_FL[j],
                stance_proximity_FR[j],
                stance_proximity_RL[j],
                stance_proximity_RR[j],
                state.base.pos[0],
                state.base.pos[1],
                state.base.pos[2],
                base_yaw,
                external_wrenches_estimated_param[0],
                external_wrenches_estimated_param[1],
                external_wrenches_estimated_param[2],
                external_wrenches_estimated_param[3],
                external_wrenches_estimated_param[4],
                external_wrenches_estimated_param[5],
                inertia[0],
                inertia[1],
                inertia[2],
                inertia[3],
                inertia[4],
                inertia[5],
                inertia[6],
                inertia[7],
                inertia[8],
                mass
            ]                 
            )
            self.acados_ocp_solver.set(j, "p", copy.deepcopy(param))
        
        # Set initial state constraint. We teleported the robot foothold
        # to the previous optimal foothold. This is done to avoid the optimization
        # of a foothold that is not considered at all at touchdown! 
        if FL_contact_sequence[0] == 0:
            state.FL.foot_pos_centered = reference.ref_foot_FL

        if FR_contact_sequence[0] == 0:
            state.FR.foot_pos_centered = reference.ref_foot_FR

        if RL_contact_sequence[0] == 0:
            state.RL.foot_pos_centered = reference.ref_foot_RL

        if RR_contact_sequence[0] == 0:
            state.RR.foot_pos_centered = reference.ref_foot_RR
            
        if self.use_integrators:
            self.integral_errors[0] += (state.base.pos[2] - reference.ref_position[2]) * self.alpha_integrator
            self.integral_errors[1] += (state.base.lin_vel_world[0] - reference.ref_linear_velocity[0]) * self.alpha_integrator
            self.integral_errors[2] += (state.base.lin_vel_world[1] - reference.ref_linear_velocity[1]) * self.alpha_integrator
            self.integral_errors[3] += (state.base.lin_vel_world[2] - reference.ref_linear_velocity[2]) * self.alpha_integrator
            self.integral_errors[4] += (state.base.euler[0] - reference.ref_orientation[0]) * self.alpha_integrator
            self.integral_errors[5] += (state.base.euler[1] - reference.ref_orientation[1]) * self.alpha_integrator
            
        for i in range(len(self.integral_errors)):
            self.integral_errors[i] = np.clip(self.integral_errors[i], -self.integrator_clip[i], self.integrator_clip[i])        
        
        state_acados = np.concatenate(
            (
                state.base.pos,
                state.base.lin_vel_world,
                state.base.euler,
                state.base.ang_vel_world,
                state.FL.foot_pos_centered,
                state.FR.foot_pos_centered,
                state.RL.foot_pos_centered,
                state.RR.foot_pos_centered,
                self.integral_errors
            )
        ).flatten()
        self.acados_ocp_solver.set(0, "lbx", state_acados)
        self.acados_ocp_solver.set(0, "ubx", state_acados)
        
        # use warm start??
        if self.use_warm_start:
            self.warm_start(
                state_acados=state_acados,
                reference=reference,
                FL_contact=FL_contact_sequence,
                FR_contact=FR_contact_sequence,
                RL_contact=RL_contact_sequence,
                RR_contact=RR_contact_sequence
            )
        
        # constraint stage
        self.set_stage_constraint(
            constraint=constraint,
            state=state,
            reference=reference,
            contact_sequence_FL=FL_contact_sequence,
            contact_sequence_FR=FR_contact_sequence,
            contact_sequence_RL=RL_contact_sequence,
            contact_sequence_RR=RR_contact_sequence
        )
        status = self.acados_ocp_solver.solve()
        if self.verbose:
            print("ocp time: ", self.acados_ocp_solver.get_stats('time_tot'))
        
        control = self.acados_ocp_solver.get(0, "u")
        optimal_GRF = control[12:]
        optimal_foothold = np.zeros((4, 3))
        optimal_footholds_assigned = [False, False, False, False]
        
        # We need to provide the next touchdown foothold position.
        # We first take the foothold in stance now (they are not optimized!)
        # and flag them as True (aka "assigned")
        
        contact_sequences = [
            FL_contact_sequence, 
            FR_contact_sequence, 
            RL_contact_sequence, 
            RR_contact_sequence
        ]
        # 将当前足端位置放入列表
        current_foot_pos = [
            state.FL.foot_pos_centered, 
            state.FR.foot_pos_centered, 
            state.RL.foot_pos_centered, 
            state.RR.foot_pos_centered
        ]
        ref_list = [
            reference.ref_foot_FL,
            reference.ref_foot_FR,
            reference.ref_foot_RL,
            reference.ref_foot_RR
        ]
        # 2. 处理当前处于支撑相 (Stance) 的腿
        # 支撑腿的落脚点就是当前所在的位置（不进行优化，直接锁定）
        for i in range(4):
            if contact_sequences[i][0] == 1:
                optimal_foothold[i] = current_foot_pos[i]
                optimal_footholds_assigned[i] = True
        
        yaw = state.base.euler[2]
        R_wb = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        base_pos_xy = state.base.pos[0:2]
        for i in range(4): # 遍历四条腿
            if not optimal_footholds_assigned[i]:
                # 在预测时域内寻找 Touchdown（从 0 变 1 的瞬间）
                for j in range(1, self.horizon):
                    if contact_sequences[i][j] == 1 and contact_sequences[i][j-1] == 0:
                        # 从状态向量 x 中提取该腿对应的落脚点 (索引需根据你的模型定义，这里假设为 12+i*3)
                        # 注意：此处索引需与你的 Acados Model 状态定义一致
                        start_idx = 12 + i * 3
                        raw_foothold = self.acados_ocp_solver.get(j, "x")[start_idx : start_idx+3]
                        
                        # --- 安全裁剪 (Saturation) ---
                        # 将落脚点转换到身体局部水平系进行裁剪
                        local_xy = R_wb @ (raw_foothold[:2] - base_pos_xy)
                        
                        if constraint is None:
                            # 如果没有外部约束（如视觉），使用参考点附近的默认盒子
                            ref_xy = R_wb @ (ref_list[i][:2] - base_pos_xy)
                            up_limit = ref_xy + 0.10
                            low_limit = ref_xy - 0.10
                        else:
                            # 使用视觉提供的 VFA 约束
                            # up_limit = R_wb @ (constraint[0:2, i] - base_pos_xy)
                            # low_limit = R_wb @ (constraint[9:11, i] - base_pos_xy)
                            pass

                        # 执行裁剪并转换回全局世界坐标
                        local_xy_clipped = np.clip(local_xy, low_limit, up_limit)
                        optimal_foothold[i][:2] = R_wb.T @ local_xy_clipped + base_pos_xy
                        optimal_foothold[i][2] = raw_foothold[2] # Z轴通常跟随地形
                        
                        optimal_footholds_assigned[i] = True
                        break # 找到第一个落地时刻即停止

        # 4. 兜底逻辑：如果在整个 Horizon 内该腿都不落地
        # 为了不让 Swing Controller 迷茫，直接给参考落脚点
        for i in range(4):
            if not optimal_footholds_assigned[i]:
                optimal_foothold[i] = ref_list[i]
        
        # 5. 确定下一时刻的目标状态 (用于下一帧的初始值)
        # 考虑到控制延迟，如果 dt 很小，通常取 index=2
        next_idx = 2 if self.dt <= 0.02 else 1
        optimal_next_state = self.acados_ocp_solver.get(next_idx, "x")
        
        # --- 6. 求解失败的兜底逻辑 (Status != 0) ---
        # Status 1: 达到最大迭代次数未收敛; Status 4: QP 求解失败
        if status != 0:
            if self.verbose:
                print(f"MPC Solver failed with status {status}. Using fallback strategy.")
            
            # A. 摆动腿回退：直接使用 FRG 提供的参考落脚点
            for i in range(4):
                if contact_sequences[i][0] == 0:
                    optimal_foothold[i] = ref_list[i]

            # B. 支撑腿力矩回退：使用上一时刻的GRF平滑过渡（避免突变）
            if self.previous_status == 0 and np.linalg.norm(self.previous_optimal_GRF) > 0:
                # 如果上一次成功，使用上一次的GRF
                optimal_GRF = self.previous_optimal_GRF.copy()
            else:
                # 如果连续失败，使用静态平衡力（基于计划接触）
                num_stance = int(sum(seq[0] for seq in contact_sequences))
                optimal_GRF = np.zeros(12)
                if num_stance <= 0:
                    num_stance = 4
                f_z_avg = self.grf_max / num_stance
                for i in range(4):
                    if contact_sequences[i][0] == 1:
                        optimal_GRF[i*3 + 2] = f_z_avg
            
            # 不要reset，保留求解器状态以便下次warm start
        
        self.previous_optimal_GRF = optimal_GRF
        self.previous_status = status
        self.previous_contact_sequence = contact_sequence
        # 核心逻辑：将相对于 (0,0) 的局部坐标，还原到世界的绝对坐标系中
        # 这样底层的 PD 控制器才能在正确的地图位置执行
        world_base_pos = self.initial_base_pos # 这是你初始保存的真实世界位置

        for i in range(4):
            optimal_foothold[i] += world_base_pos

        # 对预测的下一个状态进行还原，用于监控和调试以及状态观测
        optimal_next_state[0:3] += world_base_pos # Base Position
        # 更新状态向量中的足端位置（12-24维）
        for i in range(4):
            idx = 12 + i * 3
            optimal_next_state[idx : idx+3] = optimal_foothold[i]
        
        return optimal_GRF, optimal_foothold, optimal_next_state, status
    
    
    def warm_start(self,
        state_acados,
        reference,
        FL_contact,
        FR_contact,
        RL_contact,
        RR_contact):
        for j in range(self.horizon):
            # 1. 先拿到当前的猜测（Acados 默认会保留上一时刻的解）
            x_guess = copy.deepcopy(self.acados_ocp_solver.get(j, "x"))

            # 2. 更新足端位置猜测
            # 逻辑：如果预测到第 j 步脚是支撑状态(1)，猜它在当前位置；
            # 如果是摆动(0)，猜它去参考落脚点
            x_guess[12:15] = state_acados[12:15] if FL_contact[j] == 1 else reference.ref_foot_FL
            
            # FR 腿 (15:18)
            x_guess[15:18] = state_acados[15:18] if FR_contact[j] == 1 else reference.ref_foot_FR
            
            # RL 腿 (18:21)
            x_guess[18:21] = state_acados[18:21] if RL_contact[j] == 1 else reference.ref_foot_RL
            
            # RR 腿 (21:24)
            x_guess[21:24] = state_acados[21:24] if RR_contact[j] == 1 else reference.ref_foot_RR
            
            self.acados_ocp_solver.set(j, "x", x_guess)
            

        
    # Method to perform the centering of the states and the reference around (0, 0, 0)    
    def _perform_state_centering(self, state: QuadrupedState, reference: ReferenceState):
        self.initial_base_pos = copy.deepcopy(state.base.pos)
        reference = copy.deepcopy(reference)
        state = copy.deepcopy(state)
        
        # 保存当前高度用于参考位置
        current_xy = state.base.pos[:2].copy()
        
        # 参考位置：XY平移到0，Z保持期望高度（相对于当前）
        reference.ref_position = reference.ref_position - state.base.pos
        reference.ref_position[2] = reference.ref_position[2]  # Z高度已经在reference_interface中正确设置
        
        reference.ref_foot_FL = reference.ref_foot_FL - state.base.pos
        reference.ref_foot_FR = reference.ref_foot_FR - state.base.pos
        reference.ref_foot_RL = reference.ref_foot_RL - state.base.pos
        reference.ref_foot_RR = reference.ref_foot_RR - state.base.pos
        
        # TODO: 已经在环境层面进行状态中心化，其实不一定需要在这里重复进行状态中心化了
        state.FL.foot_pos_centered = state.FL.foot_pos - state.base.pos
        state.FR.foot_pos_centered = state.FR.foot_pos - state.base.pos
        state.RL.foot_pos_centered = state.RL.foot_pos - state.base.pos
        state.RR.foot_pos_centered = state.RR.foot_pos - state.base.pos
        state.base.pos = np.array([0, 0, 0])
        return state, reference
    
    def set_stage_constraint(self,
                             constraint,   # None or np.ndarray givn by outside
                             state: QuadrupedState,
                             reference: ReferenceState,
                             contact_sequence_FL,
                             contact_sequence_FR,
                             contact_sequence_RL,
                             contact_sequence_RR):
        yaw = state.base.euler[2]
        base_xy = state.base.pos[:2]
        R_wb = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        
        # 2. 定义局部辅助函数：快速生成一个“约束盒子”
        def create_box(foot_pos, offset_xy, offset_z):
            # 将世界坐标转为相对身体的局部坐标
            rel_xy = R_wb @ (foot_pos[:2] - base_xy)
            up = np.array([rel_xy[0] + offset_xy, rel_xy[1] + offset_xy, foot_pos[2] + offset_z])
            low = np.array([rel_xy[0] - offset_xy, rel_xy[1] - offset_xy, foot_pos[2] - offset_z])
            return up, low
        
        # 3. 计算【当前步】的支撑约束 (Stance Constraints)
        # 支撑腿必须死死踩在原地，所以给的 offset 非常小 (0.005m)
        up_st_FL, low_st_FL = create_box(state.FL.foot_pos_centered, 0.005, 0.002)
        up_st_FR, low_st_FR = create_box(state.FR.foot_pos_centered, 0.005, 0.002)
        up_st_RL, low_st_RL = create_box(state.RL.foot_pos_centered, 0.005, 0.002)
        up_st_RR, low_st_RR = create_box(state.RR.foot_pos_centered, 0.005, 0.002)
        
        # 4. 计算【未来步】的落地约束 (Swing/Touchdown Constraints)
        if constraint is None:
            up_sw_FL, low_sw_FL = create_box(reference.ref_foot_FL, 0.15, 0.005)
            up_sw_FR, low_sw_FR = create_box(reference.ref_foot_FR, 0.15, 0.005)
            up_sw_RL, low_sw_RL = create_box(reference.ref_foot_RL, 0.15, 0.005)
            up_sw_RR, low_sw_RR = create_box(reference.ref_foot_RR, 0.15, 0.005)
        else:
            # TODO: 解析 VFA 矩阵，为每一条腿设置视觉安全区
            pass
        
        # 5. 核心循环：遍历整个预测时域 (Horizon)
        # 根据步态序列 (Contact Sequence) 动态决定此时该用哪套“盒子”
        for j in range(self.horizon):
            # 获取摩擦力约束的基础值
            ub_total = self.constr_uh_friction.copy()
            lb_total = self.constr_lh_friction.copy()
            # 逻辑：脚在地上(1)用 Stance 盒子；脚在空中(0)用 Swing 盒子
            # 我们把四条腿的约束水平拼接起来
            cur_up_FL, cur_low_FL = (up_st_FL, low_st_FL) if contact_sequence_FL[j] == 1 else (up_sw_FL, low_sw_FL)
            cur_up_FR, cur_low_FR = (up_st_FR, low_st_FR) if contact_sequence_FR[j] == 1 else (up_sw_FR, low_sw_FR)
            cur_up_RL, cur_low_RL = (up_st_RL, low_st_RL) if contact_sequence_RL[j] == 1 else (up_sw_RL, low_sw_RL)
            cur_up_RR, cur_low_RR = (up_st_RR, low_st_RR) if contact_sequence_RR[j] == 1 else (up_sw_RR, low_sw_RR)

            # 拼装成 Acados 需要的 12 维向量 (4条腿 * xyz)
            foot_up = np.concatenate([cur_up_FL, cur_up_FR, cur_up_RL, cur_up_RR])
            foot_low = np.concatenate([cur_low_FL, cur_low_FR, cur_low_RL, cur_low_RR])

            # 6. 将足端位置约束叠加到非线性约束 (h) 中
            # 最终的约束向量通常是 [摩擦力约束, 足端位置约束]
            if self.use_foothold_constraint:
                ub_total = np.concatenate([ub_total, foot_up])
                lb_total = np.concatenate([lb_total, foot_low])
            else:
                ub_total = ub_total
                lb_total = lb_total
            
            # 7. 插入稳定性约束 (Support Polygon / ZMP)
            if self.use_stability_constraint:
                # 初始化为全放开状态
                ub_stab = np.array([ACADOS_INFTY] * 6)
                lb_stab = np.array([-ACADOS_INFTY] * 6)

                # --- 1. FULL STANCE (四腿支撑) ---
                if(contact_sequence_FL[j] == 1 and contact_sequence_FR[j] == 1 and
                   contact_sequence_RL[j] == 1 and contact_sequence_RR[j] == 1):
                    margin = self.sim_config.get('gait', {}).get('full_stance', {}).get('margin', 0.05)
                    # 此时四条边全部激活，重心收缩进矩形内部
                    ub_stab[0], lb_stab[0] = -margin, -ACADOS_INFTY # 原本要求 <= 0
                    ub_stab[1], lb_stab[1] = ACADOS_INFTY, 0 + margin  # 原本要求 >= 0
                    ub_stab[2], lb_stab[2] = ACADOS_INFTY, 0 + margin  # 原本要求 >= 0
                    ub_stab[3], lb_stab[3] = -margin, -ACADOS_INFTY # 原本要求 <= 0

                # --- 2. TROT (对角步态) ---
                elif(contact_sequence_FL[j] == contact_sequence_RR[j] and
                     contact_sequence_FR[j] == contact_sequence_RL[j] and 
                     contact_sequence_FL[j] != contact_sequence_FR[j]):
                    margin = self.sim_config.get('gait', {}).get('trot', {}).get('margin', 0.04)
                    if contact_sequence_FL[j] == 1: # FL-RR 支撑
                        ub_stab[4], lb_stab[4] = margin, -margin # 对角线通常是对称约束
                    else: # FR-RL 支撑
                        ub_stab[5], lb_stab[5] = margin, -margin

                # --- 3. PACE (同侧步态) ---
                elif (contact_sequence_FL[j] == contact_sequence_RL[j] and
                      contact_sequence_FR[j] == contact_sequence_RR[j]):
                    margin = self.sim_config.get('gait', {}).get('pace', {}).get('margin', 0.03)
                    if contact_sequence_FL[j] == 1: # 左侧支撑 (FL-RL)
                        ub_stab[3], lb_stab[3] = -margin, -ACADOS_INFTY # 约束左边线 (Index 3 原本 <= 0)
                    else: # 右侧支撑 (FR-RR)
                        ub_stab[1], lb_stab[1] = ACADOS_INFTY, 0 + margin  # 约束右边线 (Index 1 原本 >= 0)

                # --- 4. CRAWL (三腿支撑) ---
                else:
                    margin = self.sim_config.get('gait', {}).get('crawl', {}).get('margin', 0.05)
                    if contact_sequence_FL[j] == 0: # FL 摆动，FR-RR-RL 支撑
                        ub_stab[1], lb_stab[1] = ACADOS_INFTY, 0 + margin # 右边 (Index 1 >= 0)
                        ub_stab[2], lb_stab[2] = ACADOS_INFTY, 0 + margin # 后边 (Index 2 >= 0)
                        ub_stab[5], lb_stab[5] = margin, -margin          # 对角线
                    
                    elif contact_sequence_FR[j] == 0: # FR 摆动，FL-RL-RR 支撑
                        ub_stab[0], lb_stab[0] = -margin, -ACADOS_INFTY # 前边 (Index 0 <= 0)
                        ub_stab[3], lb_stab[3] = -margin, -ACADOS_INFTY # 左边 (Index 3 <= 0)
                        ub_stab[4], lb_stab[4] = margin, -margin

                    elif contact_sequence_RL[j] == 0: # RL 摆动，FL-FR-RR 支撑
                        ub_stab[0], lb_stab[0] = -margin, -ACADOS_INFTY
                        ub_stab[1], lb_stab[1] = ACADOS_INFTY, 0 + margin
                        ub_stab[4], lb_stab[4] = margin, -margin
                    
                    elif contact_sequence_RR[j] == 0: # RR 摆动，FL-FR-RL 支撑
                        ub_stab[2], lb_stab[2] = ACADOS_INFTY, 0 + margin
                        ub_stab[3], lb_stab[3] = -margin, -ACADOS_INFTY
                        ub_stab[5], lb_stab[5] = margin, -margin

                ub_total = np.concatenate([ub_total, ub_stab])
                lb_total = np.concatenate([lb_total, lb_stab])
                
            # No friction constraint at the end! we don't have u_N
            if j == self.horizon:
                if self.use_foothold_constraints:
                    if self.use_stability_constraints:
                        ub_total = np.concatenate((foot_up, ub_stab))
                        lb_total = np.concatenate((foot_low, lb_stab))
                    else:
                        ub_total = foot_up
                        lb_total = foot_low
                else:
                    if self.use_stability_constraints:
                        ub_total = ub_stab
                        lb_total = lb_stab
                    else:
                        continue
            # 7. 特殊处理：第 0 步通常只保留力约束，防止初始状态冲突导致无解
            if j == 0:
                self.acados_ocp_solver.constraints_set(j, "uh", self.constr_uh_friction)
                self.acados_ocp_solver.constraints_set(j, "lh", self.constr_lh_friction)
            else:
                self.acados_ocp_solver.constraints_set(j, "uh", ub_total)
                self.acados_ocp_solver.constraints_set(j, "lh", lb_total)
            
            # save the constraint for logging
            self.upper_bound[j] = ub_total.tolist()
            self.lower_bound[j] = lb_total.tolist()
        

    def reset(self):
        self.acados_ocp_solver.reset()
        self.acados_ocp_solver = AcadosOcpSolver(
            self.ocp,
            json_file=self.ocp.code_export_directory + "acados_ocp.json",
            build=False,
            generate=False,
        )
     
    def _setup_ocp_cost(self):
        nx = self.state_dim
        nu = self.control_dim
        ny = nx + nu
        
        Q_mat, R_mat = self._set_weight_by_config()
        self.ocp.solver_options.N_horizon = self.horizon
        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"
        self.ocp.cost.W_e = Q_mat
        self.ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vx[:nx,:nx] = np.eye(nx)
        self.ocp.cost.Vu[nx:nx+nu,:nu] = np.eye(nu)
        self.ocp.cost.Vx_e = np.eye(nx)
        # 初始化参考值
        self.ocp.cost.yref = np.zeros((ny,))
        self.ocp.cost.yref_e = np.zeros((nx,))

    def _setup_ocp_constraints(self, quadruped_constraints: QuadrupedConstraints):
        # friction cone
        Jbu, lb, ub = quadruped_constraints.get_friction_cone_bounds(self.grf_min, self.grf_max)
        self.constr_uh_friction = copy.deepcopy(ub)
        self.constr_lh_friction = copy.deepcopy(lb)
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
        X0 = np.zeros(shape=(self.state_dim,))
        self.ocp.constraints.x0 = X0

    def _setup_ocp_initialize_params(self):
        init_contact_sequence = np.array([1, 1, 1, 1])
        init_mu = np.array([self.sim_config.get("physics", {}).get("mu", 0.5)])
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

    def _setup_ocp_options(self):
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.hessian_approx = "GAUSS_NEWTON"
        self.ocp.integrator_type = "ERK"
        self.ocp.solver_options.qp_solver_cond_N = self.horizon
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
        
        # 关闭 Acados 的打印输出
        self.ocp.solver_options.print_level = 0

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


    
