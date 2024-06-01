"""
Authors:
    Micah Prendergast <j.m.prendergast@tudelft.nl>
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
    Saray Bakker <s.bakker@tudelft.nl>
"""

import numpy as np
from spatialmath import SO3, UnitQuaternion

class CartesianImpedanceController:
    def __init__(self, robot=[], ee_translational_stiffness=1.5, ee_rotational_stiffness=0.5):
        # Parameters
        ee_translational_stiffness = ee_translational_stiffness #10 #original: 1.5 #30 #30 #300. #1
        ee_rotational_stiffness = ee_rotational_stiffness #4 #original:0.5 #100 #60 #100. #1
        ee_translational_damping_factor = 0 # 0.78 #0.3# 0.1
        ee_rotational_damping_factor = 0 #1.0#0.5 #0.3
        elbow_stiffness_factor = 0.3
        elbow_translational_damping_factor = 3
        elbow_rotational_damping_factor = 2
        self.n_joints = 7
        self.elbow_position_d = np.array([0, 0, 1.5])

        # Set stiffness and damping
        self.ee_stiffness = self.set_stiffness(xyz=ee_translational_stiffness,
                                               rot=ee_rotational_stiffness)
        self.ee_damping = self.set_damping(np.eye(6), #originally dependent on: self.ee_stiffness
                                           xyz_factor=ee_translational_damping_factor,
                                           rot_factor=ee_rotational_damping_factor)

        self.elbow_stiffness = self.ee_stiffness * elbow_stiffness_factor

        self.elbow_damping = self.set_damping(self.elbow_stiffness, 
                                              xyz_factor=elbow_translational_damping_factor,
                                              rot_factor=elbow_rotational_damping_factor)

        # Init variables
        self.error_prev = None
        self.position_prev = None
        self.orientation_prev = None
        self.robot = robot

    def set_stiffness(self, xyz, rot):
        K = np.eye(6, 6)
        K[0, 0] = xyz
        K[1, 1] = xyz
        K[2, 2] = xyz
        K[3, 3] = rot
        K[4, 4] = rot
        K[5, 5] = rot
        return K

    def set_damping(self, stiffness, xyz_factor=3.0, rot_factor=1.0):
        D = np.sqrt(stiffness)
        D[0, 0] = xyz_factor * D[0, 0]
        D[1, 1] = xyz_factor * D[1, 1]
        D[2, 2] = xyz_factor * D[2, 2]
        D[3, 3] = rot_factor * D[3, 3]
        D[4, 4] = rot_factor * D[4, 4]
        D[5, 5] = rot_factor * D[5, 5]
        return D
        
    def control_law(self, position_d, orientation_d, ee_pose, J, ee_velocity=[], ee_velocity_d=np.zeros(6), ee_vel_quat = [], ee_vel_quat_d = [], dt=0.01):
        # Get ee's positiona and orientation's from ee's pose
        ee_position = ee_pose[:3]
        ee_orientation = UnitQuaternion(ee_pose[3:7]) #in quaternions

        # Get pose error
        error_position = self.get_pose_error(ee_position, ee_orientation, position_d, UnitQuaternion(orientation_d))/dt
        # error_position = self.get_pose_error(ee_position, ee_orientation, position_d,
        #                                      UnitQuaternion(np.array([1., 0., 0., 0.]))) / dt

        #get velocity error:
        # error_quat_vel = ee_vel_quat_d - ee_vel_quat #ee_vel_quat_d - ee_vel_quat
        # error_euler_vel = self.quat_vel_to_angular_vel(angle_quaternion=ee_orientation.A, vel_quaternion=error_quat_vel)
        # error_pos_vel = ee_velocity_d[0:3] - ee_velocity[0:3]

        xdot_PD = np.matmul(self.ee_stiffness, error_position) # + np.matmul(self.ee_damping, np.append(error_pos_vel, error_euler_vel))

        qdot = np.matmul(np.linalg.pinv(J[0][:, :]), xdot_PD)

        # Get ee's jacobian
        J = self.robot.jacob0(q, end='iiwa_link_7', start='iiwa_link_0')

        # Map ee's force to joint torques
        tau_ee = np.matmul(J.transpose(), F_ext)

        # Get nullspace torque
        tau_nullspace = self._nullspace_control(J, elbow_pose, elbow_velocity, q)

        # Add ee's tau with nullspace tau
        tau = tau_ee + tau_nullspace
        
        return tau
        return qdot

    def get_pose_error(self, position, orientation, position_d, orientation_d):
        error = np.zeros(6)
        error[:3] = position_d - position
        error[3:] = (orientation_d / orientation).rpy()
        return error

    def control_law_vel(self, position_d, orientation_d, ee_pose, ee_pose_t_1, dt=0.01, J=[]):
        """
        position_d = position desired
        orientation_d = orientation_desired
        ee_pose = current pose (position + orientation)
        ee_pose_t_1 = pose at t-1 (position + orientation)
        """
        alpha = 0 #100 #100 #100
        beta = np.eye(6)
        beta[:3, :3] = 0.001*np.eye(3)
        beta[3:, 3:] = 0. *np.eye(3)

        # ---------------- initialize ----------------- #
        # desired orientation in quat format:
        orient_d = UnitQuaternion(orientation_d) #in quaternions

        # Get ee's current position and orientation's from ee's pose
        x_t_pos = ee_pose[:3]
        x_t_orient = UnitQuaternion(ee_pose[3:]) #in quaternions

        # get ee's position and orientation from previous time step
        x_t_1_pos = ee_pose_t_1[:3]
        x_t_1_orient = UnitQuaternion(ee_pose_t_1[3:]) #in quaternions

        # ------------- pure position error control ----------------#
        error_position = self.get_pose_error(x_t_pos, x_t_orient, position_d, UnitQuaternion(orientation_d))

        # ------------------------ velocity error control -------------#
        # Get position error for velocity:
        dv_linear = (1/dt)*(position_d - 2*x_t_pos - x_t_1_pos)

        # Get angular velocity error in quaternions:
        dx_0 = orient_d/x_t_orient      #first term
        dx_1 = x_t_orient/x_t_1_orient  #second term
        dx_tot = dx_0/dx_1              #first term - second term in quat
        dv_angular = (1/dt)*dx_tot.rpy()
        #dv_angular = self.quat_vel_to_angular_vel(angle_quaternion=ee_pose[3:], vel_quaternion=dv_quat) #todo: avoid this conversion!
        error_vel = np.append(dv_linear, dv_angular)

        action = alpha*error_position + np.matmul(beta, error_vel)
        print("alpha*error_position: ", alpha*error_position, "  , beta*error_vel: ",np.matmul(beta, error_vel))
        qdot = np.matmul(np.linalg.pinv(J[0][:, :]), action)
        return qdot





    def _elbow_cartesian_impedance_controller(self, elbow_pose, elbow_velocity, q):
        # Get elbow position and orientation from its pose
        position_elbow = elbow_pose[:3]
        orientation_elbow = elbow_pose[3:]

        # Map elbow euler orientation to matrix
        orientation_elbow = SO3.RPY(orientation_elbow)

        # Get pose error (orientation elbow is disregarded, so the error against itself is computed)
        error_elbow = self.get_pose_error(position_elbow, orientation_elbow, self.elbow_position_d, orientation_elbow)

        # Compute elbow's cartesian force with PD control
        force_elbow = np.matmul(self.elbow_stiffness, error_elbow) - np.matmul(self.elbow_damping, elbow_velocity)

        # Get elbow's jacobian
        J_elbow = self.robot.jacob0(q, end='iiwa_link_3', start='iiwa_link_0')

        # Map elbow's cartesian force to joint torques
        torque_elbow = np.matmul(J_elbow.T, force_elbow)

        # Create torque vector with zeros and fill torque that can control elbow
        torque_arm = np.zeros(7)
        torque_arm[:3] = torque_elbow
        return torque_arm

    def _nullspace_control(self, J, elbow_pose, elbow_velocity, q):
        # Get torque elbow's control
        torque = self._elbow_cartesian_impedance_controller(elbow_pose, elbow_velocity, q)

        # Get nullspace matrix
        nullspace = (np.identity(self.n_joints) - np.matmul(J.T, np.linalg.pinv(J).T))

        # Map elbow's torque to ee's nullspace
        nullspace_torque = np.matmul(nullspace, torque)
        return nullspace_torque


    def quat_vel_to_angular_vel(self, angle_quaternion, vel_quaternion):
        """
        Quaternion velocities to angular velocities,
        Slight modification of book:
        https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/RD2016script.pdf
        """
        q0 = angle_quaternion[0]
        q1 = angle_quaternion[1]
        q2 = angle_quaternion[2]
        q3 = angle_quaternion[3]
        H = np.array([[-q1, q0, -q3, q2],
                   [-q2, q3, q0, -q1],
                   [-q3, -q2, q1, q0]
                   ])
        # H = np.array([[-q1, q0,  q3, -q2],
        #            [-q2, -q3, q0,  q1],
        #            [-q3,  q2, -q1, q0]
        #            ])
        angular_vel = 2*np.dot(H, vel_quaternion)
        return angular_vel