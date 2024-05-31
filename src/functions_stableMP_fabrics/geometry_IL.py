import numpy as np
import casadi as ca
import quaternionic
# from fabrics.diffGeometry.spec import Spec
# from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.helpers.variables import Variables
from fabrics.diffGeometry.geometry import Geometry
from scipy.spatial.transform import Rotation as R
import math
import spatial_casadi as sc

class construct_IL_geometry(object):
    def __init__(self, planner, dof: int, dimension_task: int, forwardkinematics: ca.SX, variables: dict, first_dim = 0, BOOL_pos_orient = False):
        # dimensions
        self._dof = dof
        self._dimension_task = dimension_task
        self.first_dim = first_dim
        self.BOOL_pos_orient = BOOL_pos_orient

        # store variables
        self._variables = variables
        self._q = variables._state_variables["q"]
        self._qdot = variables._state_variables["qdot"]

        if forwardkinematics.shape == (3, 1):
            self.forward_kinematics = forwardkinematics
            self._dimension_task_pos = dimension_task
        elif forwardkinematics.shape == (4, 4):
            self.forward_kinematics = forwardkinematics[0:3, 3]
            self.rotation_matrix = forwardkinematics[0:3, 0:3]
            self._dimension_task_pos = 3
            self._dimension_task_orien = 4
            self.construct_Jacobians_orientation(rot_matrix=self.rotation_matrix)
        else:
            print("No known shape of the forward kinematics provided")

        # symbolic geometries
        # self.M = np.identity(self._dimension_task_pos)
        if BOOL_pos_orient == True:
            self._h = ca.SX.sym("h_x", self._dimension_task_pos+3, 1)
        else:
            self._h = ca.SX.sym("h_x", self._dimension_task_pos, 1)

        # self.initialize_joint_variables()

        # operations to construct symbolic and function geometry for safe MP
        self.construct_Jacobians(phi=self.forward_kinematics)
        # self.symbolic_geometry_IL()
        # # self.h_pulled_from_planner(planner=planner)
        # self.create_function_h_pulled()

        # no limits, if not set yet.
        self.v_min = np.array([-10e6, -10e6])
        self.v_max = np.array([10e6, 10e6])
        self.acc_min = np.array([-10e6, -10e6])
        self.acc_max = np.array([10e6, 10e6])

    # def initialize_joint_variables(self):
    #     # copied from parametrized_planner.py
    #     self._q = ca.SX.sym("q", self._dof)
    #     self._qdot = ca.SX.sym("qdot", self._dof)
    #     self._variables = Variables(state_variables={"q": self._q, "qdot": self._qdot})

    def construct_Jacobians(self, phi):
        Jdot_sign = -1
        self._phi = phi
        self._J = ca.jacobian(phi, self._q)[self.first_dim:self._dimension_task_pos+self.first_dim, :]
        self._Jdot = Jdot_sign * ca.jacobian(ca.mtimes(self._J, self._qdot), self._q)
        self.select_jacobian()

    def construct_Jacobians_orientation(self, rot_matrix):
        """Jacobian of orientation vector """
        Jdot_sign = -1
        self._rot_matrix = rot_matrix
        self._eul_angles = self.symbolic_rot_matrix_to_euler(self._rot_matrix) #self.rotationMatrixToEulerAngles(self._rot_matrix)
        self._J_euler = ca.jacobian(self._eul_angles, self._q)
        self._Jdot_euler = Jdot_sign * ca.jacobian(ca.mtimes(self._J_euler, self._qdot), self._q)
        self._quaternions = self.symbolic_rot_matrix_to_quaternions(rot_matrix)
        self._J_quat = ca.jacobian(self._quaternions, self._q)
        self._Jdot_quat = Jdot_sign * ca.jacobian(ca.mtimes(self._J_quat, self._qdot), self._q)

    def select_jacobian(self):
        """ Make sure you can switch between position and pos+orientation jacobian.
        #https://robotacademy.net.au/lesson/the-analytic-jacobian/
        """
        if self.BOOL_pos_orient == True:
            self._J_angular = self.analytical_jacobian()
            self.J_analytic = ca.vcat([self._J, self._J_euler])
            self.J_analytic_dot = ca.vcat([self._Jdot, self._Jdot_euler])
        else:
            self.J_analytic = self._J
            self.J_analytic_dot = self._Jdot

    def symbolic_rot_matrix_to_quaternions(self, rot_matrix):
        r = sc.Rotation.from_matrix(rot_matrix)
        quatern = r.as_quat()
        return quatern

    def symbolic_rot_matrix_to_euler(self, rot_matrix):
        r = sc.Rotation.from_matrix(rot_matrix)
        euler = r.as_euler(seq="xyz")
        return euler

    def fk_functions(self):
        #x via forward kinematics
        self.x_function = ca.Function("x_fk", [self._q], [self.forward_kinematics], ["q"], ["x"])

        #xdot via Jacobian*qdot
        xdot = ca.mtimes(self._J, self._qdot)
        self.xdot_function = ca.Function("xdot_fk", [self._q, self._qdot], [xdot], ["q", "qdot"], ["x_dot"])

        return self.x_function, self.xdot_function

    def fk_functions_with_orientation(self):
        """
        xdot_function including angular velocities in derivative quaternions.
        """
        J_quat = self.analytical_jacobian()
        J_analytical = ca.vcat([self._J, J_quat])
        xdot = ca.mtimes(J_analytical, self._qdot)
        self.xdot_function_analytical = ca.Function("xdot_analytic", [self._q, self._qdot], [xdot], ["q", "qdot"], ["xdot"])
        return self.xdot_function_analytical

    def rotation_matrix_function(self):
        self.rot_matrix_function = ca.Function("rotation_matrix_fk", [self._q], [self.rotation_matrix])
        return self.rot_matrix_function

    def get_orientation_quaternion(self, q):
        rotation_matrix = self.rot_matrix_function(q)
        r = quaternionic.array.from_rotation_matrix(rotation_matrix)
        orientation = np.array(r)
        return orientation

    def get_rotation_matrix_from_quaternion(self, x_quat):
        r = R.from_quat(x_quat)
        return r.as_matrix()


    def symbolic_geometry_IL(self):
        #I think it doesn't work bcause the variables are not the same....
        self.differentialmap = DifferentialMap(self.forward_kinematics, self._variables)
        self._h_pulled = self.pull(J=self.J_analytic, J_dot=self.J_analytic_dot, h=self._h, q_dot=self._qdot)
        return self._h_pulled


    def pull(self, J, J_dot, h, q_dot):
        # gotten from "geometry.py" in fabrics
        J_dot_q_dot = ca.mtimes(J_dot, q_dot)
        h_pulled = ca.mtimes(ca.pinv(J), h + J_dot_q_dot)
        return h_pulled

    def create_function_vel_pulled(self):
        Jac = self._J
        # Jacobian =
        # if self.BOOL_pos_orient == True:
        #     self._J_angular = self.analytical_jacobian()
        #     J_analytic = ca.vcat([self._J, self._J_euler])
        # else:
        #     J_analytic = self._J
        # symbolic function:
        self._vel_pulled = self.pull_vel(J=self.J_analytic, h=self._h)

        # casadi function:
        self._vel_pulled_function = ca.Function("vel_pulled", [self._q, self._h], [self._vel_pulled],
                                                ["q", "vel_task"], ["qdot"])

        self._inv_Jacobian_function = ca.Function("inv_Jac", [self._q], [ca.pinv(self.J_analytic)], ["q"], ["inv_J"])
        return self._vel_pulled_function

    def pull_vel(self, J, h):
        vel_pulled = ca.mtimes(ca.pinv(J), h)
        return vel_pulled

    # def h_pulled_from_planner(self, planner):
    #     # if you would like to not use the fk but get it via the planner (be aware that goal parameters are still in there:
    #     mapping = planner.leaves["goal_0_leaf"]._map
    #     phi = mapping._phi
    #     J = mapping._phi
    #     Jdot = mapping._Jdot
    #     var = mapping._vars
    #     h_pulled = self.pull(J=J, J_dot=Jdot, h=self._h, q_dot=self._qdot)
    #     self._h_pulled = h_pulled
    #     return h_pulled

    def create_function_h_pulled(self):
        self.symbolic_geometry_IL()
        self._h_pulled_function = ca.Function("h_pulled", [self._q, self._qdot, self._h], [self._h_pulled],
                                              ["q", "qdot", "h_task"], ["h_config"])

        return self._h_pulled_function

    def get_numerical_vel_pulled(self, q_num, h_NN):
        h_num = self._vel_pulled_function(q_num, h_NN)
        if any(np.isnan(h_num) == True):
            h_num = 0*h_num
            print("nan detected in pulled acceleration!")
        return h_num.full().transpose()[0]

    def get_numerical_invJ(self, q_num):
        invJ_num = self._inv_Jacobian_function(q_num)
        return invJ_num


    def get_numerical_h_pulled(self, q_num, qdot_num, h_NN):
        # get h_pulled numerically from (q, qdot)
        h_num = self._h_pulled_function(q_num, qdot_num, h_NN)
        if any(np.isnan(h_num) == True):
            h_num = 0*h_num
            print("nan detected in pulled acceleration!")
        return h_num.full().transpose()[0]

    def get_action_safeMP_pulled(self, qdot, qddot, mode="acc", dt=0.01):
        if mode == "acc":
            action = qddot
        elif mode == "vel":
            action = qdot + dt*qddot
        else:
            action = 0
            print("mode could not be found!")

        #also limit the action:
        action = self.get_action_in_limits(action, mode=mode)
        return action

    def set_limits(self, v_min, v_max, acc_min, acc_max):
        self.v_min = v_min
        self.v_max = v_max
        self.acc_min = acc_min
        self.acc_max = acc_max

    def get_action_in_limits(self, action_old, mode="acc"):
        if mode == "vel":
            action = np.clip(action_old, self.v_min, self.v_max)
        else:
            action = np.clip(action_old, self.acc_min, self.acc_max)
        return action

    def joint_state_to_action_state(self, q, qdot):
        # --- End-effector state ---#
        x_ee = self.x_function(q).full().transpose()[0][0:self._dimension_task]
        xdot_ee = self.xdot_function(q, qdot).full().transpose()[0]
        xee_orientation = self.get_orientation_quaternion(q)
        if self.BOOL_pos_orient == True:
            x_t = np.array([np.append(x_ee, xee_orientation)])
        else:
            x_t = x_ee
        return x_t

    def q(self):
        return self._variables.position_variable()

    def qdot(self):
        return self._variables.velocity_variable()

    # def isRotationMatrix(self, R):
    #     """
    #     https://learnopencv.com/rotation-matrix-to-euler-angles/
    #     Checks if a matrix is a valid rotation matrix.
    #     """
    #     Rt = np.transpose(R)
    #     shouldBeIdentity = np.dot(Rt, R)
    #     I = np.identity(3, dtype=R.dtype)
    #     n = np.linalg.norm(I - shouldBeIdentity)
    #     return n < 1e-6
    #
    # def rotationMatrixToEulerAngles(self, RR):
    #     """
    #     https://learnopencv.com/rotation-matrix-to-euler-angles/
    #     Calculates rotation matrix to euler angles
    #     # The result is the same as MATLAB except the order
    #     # of the euler angles ( x and z are swapped ).
    #     """
    #     # assert (self.isRotationMatrix(R))
    #
    #     sy = ca.sqrt(RR[0, 0] * RR[0, 0] + RR[1, 0] * RR[1, 0])
    #
    #     # singular = sy < 1e-6
    #
    #     # if not singular:
    #     x = ca.atan2(RR[2, 1], RR[2, 2])
    #     y = ca.atan2(-RR[2, 0], sy)
    #     z = ca.atan2(RR[1, 0], RR[0, 0])
    #     # else:
    #     #     x = ca.atan2(-RR[1, 2], RR[1, 1])
    #     #     y = ca.atan2(-RR[2, 0], sy)
    #     #     z = 0
    #     return ca.hcat([x, y, z])

    # def quat_vel_to_euler_vel(self, quat_vel_norm, quat_norm, dt=0.01):
    #     """"
    #     Via the inefficient route:
    #     quat_dot --> quat --> euler --> euler_vel
    #     todo: smoothen!!
    #     """
    #     # previous euler angle
    #     rr = R.from_quat(quat_norm)
    #     euler_angles_prev = rr.as_euler("xyz")
    #
    #     #new euler angles
    #     quat_norm_ = quat_norm + quat_vel_norm.transpose()
    #     r = R.from_quat(quat_norm_)
    #     euler_angles = r.as_euler("xyz")
    #
    #     #euler velocity
    #     euler_vel = (euler_angles - euler_angles_prev)
    #     return euler_vel

    def analytical_jacobian(self):
        angle_quaternion = self.symbolic_rot_matrix_to_quaternions(self._rot_matrix)
        q0 = angle_quaternion[0]
        q1 = angle_quaternion[1]
        q2 = angle_quaternion[2]
        q3 = angle_quaternion[3]

        row1 = ca.hcat([-q1, q0, -q3, q2])
        row2 = ca.hcat([-q2, q3, q0, -q1])
        row3 = ca.hcat([-q3, -q2, q1, q0])
        H = ca.vcat([row1, row2, row3])
        H_transpose = H.T
        # H = ca.SX([[-q1, q0,  q3, -q2],
        #            [-q2, -q3, q0,  q1],
        #            [-q3,  q2, -q1, q0]
        #            ])
        Jac_angles_quat = 0.5*H_transpose @ self._J_euler
        return Jac_angles_quat

    def quat_vel_to_angular_vel(self, angle_quaternion, vel_quaternion):
        # angle_quaternion = self.symbolic_rot_matrix_to_quaternions(rot_matrix)
        q0 = angle_quaternion[0]
        q1 = angle_quaternion[1]
        q2 = angle_quaternion[2]
        q3 = angle_quaternion[3]
        H = np.array([[-q1, q0, -q3, q2],
                   [-q2, q3, q0, -q1],
                   [-q3, -q2, q1, q0]
                   ])
        # H = ca.SX([[-q1, q0,  q3, -q2],
        #            [-q2, -q3, q0,  q1],
        #            [-q3,  q2, -q1, q0]
        #            ])
        angular_vel = 2*np.dot(H, vel_quaternion)
        return angular_vel
