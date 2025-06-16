import mujoco
import do_mpc
import numpy as np
import casadi
class MPC:
    def __init__(self, data):
        self. data = data
        mujoco.mj_forward(data.model, data)
        self.model = self.init_model()
        self.controller = self.init_controller()


    def rotation_matrix_to_quat_numpy(self,R):
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s1 = np.sqrt(trace + 1.0) * 2  
            qw = 0.25 * s1
            qx = (R[2, 1] - R[1, 2]) / s1
            qy = (R[0, 2] - R[2, 0]) / s1
            qz = (R[1, 0] - R[0, 1]) / s1
        
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s2 = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2 
            qw = (R[2, 1] - R[1, 2]) / s2
            qx = 0.25 * s2
            qy = (R[0, 1] + R[1, 0]) / s2
            qz = (R[0, 2] + R[2, 0]) / s2
        
        elif R[1, 1] > R[2, 2]:
            s3 = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2 
            qw = (R[0, 2] - R[2, 0]) / s3
            qx = (R[0, 1] + R[1, 0]) / s3
            qy = 0.25 * s3
            qz = (R[1, 2] + R[2, 1]) / s3
        
        else:
            s4 = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2 
            qw = (R[1, 0] - R[0, 1]) / s4
            qx = (R[0, 2] + R[2, 0]) / s4
            qy = (R[1, 2] + R[2, 1]) / s4
            qz = 0.25 * s4
        
        quat = np.array([qw, qx, qy, qz])
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        quat_normalized = quat / norm
        
        return quat_normalized

    def rotation_matrix_to_quat(self, R):
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        case1_condition = trace > 0
        s1 = casadi.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw1 = 0.25 * s1
        qx1 = (R[2, 1] - R[1, 2]) / s1
        qy1 = (R[0, 2] - R[2, 0]) / s1
        qz1 = (R[1, 0] - R[0, 1]) / s1
        
        case2_condition = casadi.logic_and(R[0, 0] > R[1, 1], R[0, 0] > R[2, 2])
        case2_condition = casadi.logic_and(case2_condition, casadi.logic_not(case1_condition))
        s2 = casadi.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw2 = (R[2, 1] - R[1, 2]) / s2
        qx2 = 0.25 * s2
        qy2 = (R[0, 1] + R[1, 0]) / s2
        qz2 = (R[0, 2] + R[2, 0]) / s2
        
        case3_condition = R[1, 1] > R[2, 2]
        case3_condition = casadi.logic_and(case3_condition, casadi.logic_not(case1_condition))
        case3_condition = casadi.logic_and(case3_condition, casadi.logic_not(case2_condition))
        s3 = casadi.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw3 = (R[0, 2] - R[2, 0]) / s3
        qx3 = (R[0, 1] + R[1, 0]) / s3
        qy3 = 0.25 * s3
        qz3 = (R[1, 2] + R[2, 1]) / s3
        
        s4 = casadi.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw4 = (R[1, 0] - R[0, 1]) / s4
        qx4 = (R[0, 2] + R[2, 0]) / s4
        qy4 = (R[1, 2] + R[2, 1]) / s4
        qz4 = 0.25 * s4
        
        qw = casadi.if_else(case1_condition, qw1,
                        casadi.if_else(case2_condition, qw2,
                                        casadi.if_else(case3_condition, qw3, qw4)))
        qx = casadi.if_else(case1_condition, qx1,
                        casadi.if_else(case2_condition, qx2,
                                        casadi.if_else(case3_condition, qx3, qx4)))
        qy = casadi.if_else(case1_condition, qy1,
                        casadi.if_else(case2_condition, qy2,
                                        casadi.if_else(case3_condition, qy3, qy4)))
        qz = casadi.if_else(case1_condition, qz1,
                        casadi.if_else(case2_condition, qz2,
                                        casadi.if_else(case3_condition, qz3, qz4)))
        
        quat = casadi.vertcat(qw, qx, qy, qz)
        
        norm = casadi.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        quat_normalized = quat / norm
        
        return quat_normalized

    def get_trajectory(self, t_now):
        base_z = 0.4
        frequency = 0.2
        if isinstance(t_now, float):
            t = t_now
        else:
            t = t_now[0]
        phase = 2 * np.pi * frequency * t
        target_x = 0.5
        target_y = 0.3 * np.sin(phase)
        target_pos = np.array([target_x, target_y, base_z])
        target_direction = np.array([0.0, 0.0, 0.0])
        return target_pos, target_direction

    def quat_to_rotation_matrix(self, qw, qx, qy, qz):
        norm = casadi.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        
        R = casadi.vertcat(
            casadi.horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
            casadi.horzcat(2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)),
            casadi.horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2))
        )
        return R

    def calculate_end_effector_kinematics(self, q):        
        T1 = casadi.SX.eye(4)
        T1[0, 3] = 0
        T1[1, 3] = 0
        T1[2, 3] = 0.333
        
        c1, s1 = casadi.cos(q[0]), casadi.sin(q[0])
        R1 = casadi.vertcat(
            casadi.horzcat(c1, -s1, 0),
            casadi.horzcat(s1, c1, 0),
            casadi.horzcat(0, 0, 1)
        )
        T1[:3, :3] = R1
        
        T2 = casadi.SX.eye(4)
        T2[0, 3] = 0  
        T2[1, 3] = 0
        T2[2, 3] = 0
        
        R2_static = self.quat_to_rotation_matrix(1, -1, 0, 0)
        
        c2, s2 = casadi.cos(q[1]), casadi.sin(q[1])
        R2_joint = casadi.vertcat(
            casadi.horzcat(c2, -s2, 0),
            casadi.horzcat(s2, c2, 0),
            casadi.horzcat(0, 0, 1)
        )
        
        T2[:3, :3] = casadi.mtimes(R2_static, R2_joint)
        
        T3 = casadi.SX.eye(4)
        T3[0, 3] = 0
        T3[1, 3] = -0.316
        T3[2, 3] = 0
        
        R3_static = self.quat_to_rotation_matrix(1, 1, 0, 0)
        
        c3, s3 = casadi.cos(q[2]), casadi.sin(q[2])
        R3_joint = casadi.vertcat(
            casadi.horzcat(c3, -s3, 0),
            casadi.horzcat(s3, c3, 0),
            casadi.horzcat(0, 0, 1)
        )
        
        T3[:3, :3] = casadi.mtimes(R3_static, R3_joint)
        
        T4 = casadi.SX.eye(4)
        T4[0, 3] = 0.0825
        T4[1, 3] = 0
        T4[2, 3] = 0
        
        R4_static = self.quat_to_rotation_matrix(1, 1, 0, 0)
        
        c4, s4 = casadi.cos(q[3]), casadi.sin(q[3])
        R4_joint = casadi.vertcat(
            casadi.horzcat(c4, -s4, 0),
            casadi.horzcat(s4, c4, 0),
            casadi.horzcat(0, 0, 1)
        )
        
        T4[:3, :3] = casadi.mtimes(R4_static, R4_joint)
        
        T5 = casadi.SX.eye(4)
        T5[0, 3] = -0.0825
        T5[1, 3] = 0.384
        T5[2, 3] = 0
        
        R5_static = self.quat_to_rotation_matrix(1, -1, 0, 0)
        
        c5, s5 = casadi.cos(q[4]), casadi.sin(q[4])
        R5_joint = casadi.vertcat(
            casadi.horzcat(c5, -s5, 0),
            casadi.horzcat(s5, c5, 0),
            casadi.horzcat(0, 0, 1)
        )
        
        T5[:3, :3] = casadi.mtimes(R5_static, R5_joint)
        
        T6 = casadi.SX.eye(4)
        T6[0, 3] = 0
        T6[1, 3] = 0
        T6[2, 3] = 0
        
        R6_static = self.quat_to_rotation_matrix(1, 1, 0, 0)
        
        c6, s6 = casadi.cos(q[5]), casadi.sin(q[5])
        R6_joint = casadi.vertcat(
            casadi.horzcat(c6, -s6, 0),
            casadi.horzcat(s6, c6, 0),
            casadi.horzcat(0, 0, 1)
        )
        
        T6[:3, :3] = casadi.mtimes(R6_static, R6_joint)
        
        T7 = casadi.SX.eye(4)
        T7[0, 3] = 0.088
        T7[1, 3] = 0
        T7[2, 3] = 0
        
        R7_static = self.quat_to_rotation_matrix(1, 1, 0, 0)
        
        c7, s7 = casadi.cos(q[6]), casadi.sin(q[6])
        R7_joint = casadi.vertcat(
            casadi.horzcat(c7, -s7, 0),
            casadi.horzcat(s7, c7, 0),
            casadi.horzcat(0, 0, 1)
        )
        
        T7[:3, :3] = casadi.mtimes(R7_static, R7_joint)
        
        T_hand = casadi.SX.eye(4)
        T_hand[0, 3] = 0
        T_hand[1, 3] = 0
        T_hand[2, 3] = 0.107
        
        R_hand = self.quat_to_rotation_matrix(0.9238795, 0, 0, -0.3826834)
        T_hand[:3, :3] = R_hand
        
        T_attachment = casadi.SX.eye(4)
        T_attachment[0, 3] = 0        
        T_attachment[1, 3] = 0
        T_attachment[2, 3] = 0.107
        T_total = T1

        for T in [T2, T3, T4, T5, T6, T7, T_hand, T_attachment]:
            T_total = casadi.mtimes(T_total, T)
        

        pos = T_total[:3, 3]
        R_final = T_total[:3, :3]
        
        return pos, R_final

    def calculate_dynamics(self, u, qd, q):
        m = [0.629769,    # link0
            4.970684,    # link1
            0.646926,    # link2  
            3.228604,    # link3
            3.587895,    # link4
            1.225946,    # link5
            1.666555,    # link6
            0.73522,     # link7
            0.73]        # hand
        
        com = [
            [-0.041018, -0.00014, 0.049974],      # link0
            [0.003875, 0.002081, -0.04762],       # link1
            [-0.003141, -0.02872, 0.003495],      # link2
            [0.027518, 0.039252, -0.066502],      # link3
            [-0.05317, 0.104419, 0.027454],       # link4
            [-0.011953, 0.041065, -0.038437],     # link5
            [0.060149, -0.014117, -0.010517],     # link6
            [0.010517, -0.004252, 0.061597],      # link7
            [-0.01, 0, 0.03]                      # hand
        ]
        
        inertia = [
            [0.00315, 0.00388, 0.004285, 8.2904e-7, 8.2299e-6, 0.00015],    # link0
            [0.70337, 0.70661, 0.0091170, -0.00013900, 0.019169, 0.0067720], # link1
            [0.0079620, 2.8110e-2, 2.5995e-2, -3.925e-3, 7.04e-4, 1.0254e-2], # link2
            [3.7242e-2, 3.6155e-2, 1.083e-2, -4.761e-3, -1.2805e-2, -1.1396e-2], # link3
            [2.5853e-2, 1.9552e-2, 2.8323e-2, 7.796e-3, 8.641e-3, -1.332e-3], # link4
            [3.5549e-2, 2.9474e-2, 8.627e-3, -2.117e-3, 2.29e-4, -4.037e-3], # link5
            [1.964e-3, 4.354e-3, 5.433e-3, 1.09e-4, 3.41e-4, -1.158e-3],     # link6
            [1.2516e-2, 1.0027e-2, 4.815e-3, -4.28e-4, -7.41e-4, -1.196e-3], # link7
            [0.001, 0.0025, 0.0017, 0, 0, 0]                                 # hand
        ]
        
        dh_params = [
            [0, 0, 0.333, 0],          # Joint 1
            [0, -casadi.pi/2, 0, 0],   # Joint 2
            [0, casadi.pi/2, 0.316, 0], # Joint 3
            [0.0825, casadi.pi/2, 0, 0], # Joint 4
            [-0.0825, -casadi.pi/2, 0.384, 0], # Joint 5
            [0, casadi.pi/2, 0, 0],    # Joint 6
            [0.088, casadi.pi/2, 0, 0], # Joint 7
            [0, 0, 0.107, 0],          # Hand
            [0, 0, 0.107, 0]           # Attachment
        ]
        
        damping = casadi.SX.ones(7) * 1.0   
        armature = casadi.SX.ones(7) * 0.1  
        
        cos_q = [casadi.cos(q[i]) for i in range(7)]
        sin_q = [casadi.sin(q[i]) for i in range(7)]
        
        M_diag = [casadi.SX(0.0)] * 7  # ZMIANA: używamy SX
        
        M_diag[0] = sum(m) + armature[0]
        for i in range(1, 9):
            M_diag[0] += inertia[i][0] + inertia[i][1]
        
        link2_to_hand_mass = sum(m[2:]) 
        M_diag[1] = (link2_to_hand_mass * (dh_params[2][2]**2 + dh_params[4][2]**2) + 
                    inertia[1][1] + inertia[1][2] + armature[1])
        
        M_diag[1] += link2_to_hand_mass * dh_params[2][2] * dh_params[4][2] * cos_q[1]**2
        
        M_diag[2] = (sum(m[3:]) * (dh_params[3][0]**2 + dh_params[7][2]**2) + 
                    inertia[2][0] + inertia[2][2] + armature[2])
        M_diag[2] += sum(m[3:]) * dh_params[3][0] * dh_params[7][2] * cos_q[2]**2
        
        M_diag[3] = (sum(m[4:]) * (dh_params[4][0]**2 + dh_params[4][2]**2) + 
                    inertia[3][0] + inertia[3][2] + armature[3])
        M_diag[3] += sum(m[4:]) * abs(dh_params[4][0]) * dh_params[4][2] * cos_q[3]**2
        
        M_diag[4] = (sum(m[5:]) * dh_params[5][2]**2 + 
                    inertia[4][0] + inertia[4][2] + armature[4])
        
        M_diag[5] = (sum(m[6:]) * dh_params[6][0]**2 + 
                    inertia[5][0] + inertia[5][2] + armature[5])
        M_diag[5] += sum(m[6:]) * dh_params[6][0] * dh_params[7][2] * cos_q[5]**2
        
        M_diag[6] = (m[7] + m[8]) * dh_params[7][2]**2 + inertia[6][1] + inertia[6][2] + armature[6]
        
        M_off_diag = casadi.SX.zeros(7, 7)
        
        v1 = sum(m[3:]) * dh_params[2][2] * dh_params[3][0] * cos_q[2]
        M_off_diag[1, 2] = v1
        M_off_diag[2, 1] = v1  
        v2 = sum(m[4:]) * dh_params[3][0] * dh_params[4][0] * sin_q[3]
        M_off_diag[2, 3] = v2
        M_off_diag[3, 2] = v2  
        

        v3 = sum(m[5:]) * dh_params[4][0] * sin_q[4]
        M_off_diag[3, 4] = v3
        M_off_diag[4, 3] = v3  
        
        v4 = sum(m[6:]) * dh_params[5][2] * dh_params[6][0] * sin_q[5]
        M_off_diag[4, 5] = v4
        M_off_diag[5, 4] = v4  
        
        M = casadi.SX.zeros(7, 7)
        for i in range(7):
            M[i, i] = M_diag[i]
            for j in range(7):
                if i != j:
                    M[i, j] = M_off_diag[i, j]
        
        C_centrifugal = casadi.SX.zeros(7, 1)
        
        C_centrifugal[1] = -sum(m[3:]) * dh_params[2][2] * sin_q[1] * qd[1]**2
        
        C_centrifugal[2] = -sum(m[4:]) * dh_params[3][0] * sin_q[2] * qd[2]**2
        
        C_centrifugal[3] = -sum(m[5:]) * dh_params[4][2] * sin_q[3] * qd[3]**2
        
        C_centrifugal[5] = -sum(m[7:]) * dh_params[6][0] * sin_q[5] * qd[5]**2
        
        C_coriolis = casadi.SX.zeros(7, 1)
        
        C_coriolis[0] += -sum(m[2:]) * dh_params[2][2] * sin_q[1] * qd[0] * qd[1]
        C_coriolis[1] += sum(m[2:]) * dh_params[2][2] * sin_q[1] * qd[0] * qd[1]
        
        C_coriolis[1] += -sum(m[3:]) * dh_params[2][2] * dh_params[3][0] * sin_q[2] * qd[1] * qd[2]
        C_coriolis[2] += sum(m[3:]) * dh_params[2][2] * dh_params[3][0] * sin_q[2] * qd[1] * qd[2]
        
        C_coriolis[2] += -sum(m[4:]) * dh_params[3][0] * dh_params[4][0] * cos_q[3] * qd[2] * qd[3]
        C_coriolis[3] += sum(m[4:]) * dh_params[3][0] * dh_params[4][0] * cos_q[3] * qd[2] * qd[3]
        
        C_damping = damping * qd
        
        C = C_centrifugal + C_coriolis + C_damping
        
        g = 9.81
        G = casadi.SX.zeros(7, 1)
        
        G[1] = g * (m[1] * com[1][2] * cos_q[1] +                # link1 COM
                sum(m[2:]) * (dh_params[2][2] * cos_q[1] +    # link2
                            dh_params[4][2] * sin_q[1] * sin_q[3]))  # dalsze linki
        
        G[2] = g * (m[2] * com[2][2] * cos_q[2] +                # link2 COM
                sum(m[3:]) * dh_params[3][0] * sin_q[2])      # dalsze linki
        
        G[3] = g * (m[3] * com[3][2] * cos_q[3] +                # link3 COM
                sum(m[4:]) * dh_params[4][2] * cos_q[3])      # dalsze linki
        
        G[4] = g * m[4] * com[4][2] * sin_q[4]                   # link4 COM
        
        G[5] = g * (m[5] * com[5][2] * cos_q[5] +                # link5 COM
                    sum(m[6:]) * dh_params[6][0] * sin_q[5])     # dalsze linki
        
        G[6] = g * m[6] * com[6][2] * sin_q[6]             
        

        qdd = casadi.solve(M, (u - C - G))
        
        return qdd

    def init_model(self):
        model = do_mpc.model.Model('continuous')
        q = model.set_variable(var_type='_x', var_name='q', shape=(7, 1))
        qd = model.set_variable(var_type='_x', var_name='qd', shape=(7, 1))
        u = model.set_variable(var_type='_u', var_name='u', shape=(7, 1))
        taget = model.set_variable(var_type='_tvp', var_name='target', shape=(7, 1))
        mujoco.mj_forward(self.data.model, self.data)

        target_pos = model.set_variable(var_type='_tvp', var_name='target_pos', shape=(3, 1))
        target_rot = model.set_variable(var_type='_tvp', var_name='target_rot', shape=(3, 1))  # 3x3 matriz jako wektor
        
       
        ee_pos, ee_rot = self.calculate_end_effector_kinematics(q)
        
        


        qdd = self.calculate_dynamics(u, qd, q)
        model.set_rhs('q', qd)
        model.set_rhs('qd', qdd)
        model.set_expression('ee_pos', ee_pos)
        model.set_expression('ee_rot', ee_rot)
        model.setup()
        return model
    
    def calculate_orientation_error(self, ee_rot, target_direction):

        
        current_x_axis = ee_rot[:, 2] 
    
        target_x_direction = casadi.vertcat(0.0, 0.0, -1.0)
        
        orientation_error = casadi.vertcat(
            current_x_axis[1] * target_x_direction[2] - current_x_axis[2] * target_x_direction[1],
            current_x_axis[2] * target_x_direction[0] - current_x_axis[0] * target_x_direction[2],
            current_x_axis[0] * target_x_direction[1] - current_x_axis[1] * target_x_direction[0]
        )
        
        return orientation_error   
    
    def init_controller(self):
        mpc = do_mpc.controller.MPC(self.model)
        n_horizon = 5
        setup_mpc = {
            'n_horizon': n_horizon,
            't_step': 0.002, 
            'store_full_solution': True,
            'nlpsol_opts': {
                'ipopt.max_iter': 200,      
                'ipopt.print_level': 0,
                'ipopt.tol': 1e-4,          
                'ipopt.acceptable_tol': 1e-3,
                'ipopt.constr_viol_tol': 1e-4,
                'ipopt.warm_start_init_point': 'yes', 
                'ipopt.mu_strategy': 'adaptive',
                'print_time': 0
            }
        }
        mpc.settings.set_linear_solver(solver_name = "MA27")
        mpc.set_param(**setup_mpc)
        pos_error = self.model.aux['ee_pos'] - self.model.tvp['target_pos']
        weights = casadi.vertcat(200.0, 200.0, 200.0)
        weighted_pos_cost = casadi.sum1(weights * (pos_error**2))
        rot_weights = casadi.vertcat(200.0, 200.0, 200.0)     
        orientation_error = self.calculate_orientation_error(
            self.model.aux['ee_rot'], 
            self.model.tvp['target_rot']
        )
        weighted_rot_cost = casadi.sum1(rot_weights * (orientation_error**2))

    
        mterm = weighted_pos_cost
        lterm = casadi.DM.zeros()


        ### Tutaj można ustawić mu contrait na to, aby trzymał orientację ręki, mimo zaleceń aby ustawić robota w pobliżu celu, ten dalej nie potrafi utrzymać pozycji i nie wylecieć
        # mpc.set_nl_cons('obstacles', weighted_pos_cost, 0)
        # mpc.set_nl_cons('obstacles2', weighted_rot_cost, 0.01)

        tvp_template = mpc.get_tvp_template()
        def tvp_fun(t_now):
            for k in range(setup_mpc['n_horizon']+1):
                future_time = t_now + k * setup_mpc['t_step']
                target_pos, target_rot = self.get_trajectory(future_time)
                tvp_template["_tvp", k, "target_pos"] = target_pos.reshape(-1, 1)
                tvp_template["_tvp", k, "target_rot"] = target_rot.flatten().reshape(-1, 1)
            
            return tvp_template
        mpc.set_tvp_fun(tvp_fun)
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=0.1)

        mpc.bounds['lower', '_u', 'u'] = -3
        mpc.bounds['upper', '_u', 'u'] = 3

        joint_pos_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        joint_pos_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        mpc.bounds['lower', '_x', 'q'] = joint_pos_lower
        mpc.bounds['upper', '_x', 'q'] = joint_pos_upper
        mpc.setup()
        return mpc