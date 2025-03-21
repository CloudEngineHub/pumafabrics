"""
Authors:
    Micah Prendergast <j.m.prendergast@tudelft.nl>
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
"""

import numpy as np
import os
import pytorch_kinematics as pk
import torch
from pumafabrics.tamed_puma.nullspace_control.iiwa_robotics_toolbox import iiwa
from spatialmath import UnitQuaternion

class CartesianImpedanceController:
    def __init__(self, robot_name="iiwa14"):
        # Parameters
        ee_translational_stiffness = 800
        ee_rotational_stiffness = 25
        ee_translational_damping_factor = 3
        ee_rotational_damping_factor = 1
        elbow_stiffness_factor = 0.1
        elbow_translational_damping_factor = 3
        elbow_rotational_damping_factor = 2
        self.n_joints = 7
        self.elbow_position_d = np.array([0.2, 0, 0.45])

        # Set stiffness and damping
        self.ee_stiffness = self.set_stiffness(xyz=ee_translational_stiffness,
                                               rot=ee_rotational_stiffness)

        self.ee_damping = self.set_damping(self.ee_stiffness,
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

        self.robot_name = robot_name
        self.construct_chain_elbow_endeff()

        self.robot = iiwa(model=robot_name)

    def construct_chain_elbow_endeff(self):
        path_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        path_child = path_parent + '/config/urdfs/%s.urdf' % self.robot_name
        self.chain_elbow = pk.build_serial_chain_from_urdf(open(path_child).read(), "iiwa_link_3", root_link_name="iiwa_link_0")
        self.chain_endeff = pk.build_serial_chain_from_urdf(open(path_child).read(), "iiwa_link_7")

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

    def _elbow_cartesian_impedance_controller(self, q, qdot):
        # Get elbow position and orientation from its pose
        elbow_T = self.chain_elbow.forward_kinematics(q, end_only=False)["iiwa_link_3"].get_matrix()
        position_elbow = torch.Tensor.numpy(elbow_T[:, :3, 3])[0]
        orientation_elbow = torch.Tensor.numpy(pk.matrix_to_quaternion(elbow_T[:, :3, :3]))[0]

        #differentiable kinematics
        J_elbow = self.robot.jacob0(q, end='iiwa_link_3', start='iiwa_link_0')
        elbow_velocity = J_elbow @ qdot[0:3]

        # Get pose error (orientation elbow is disregarded, so the error against itself is computed)
        error_elbow = self.get_pose_error(position_elbow, orientation_elbow, self.elbow_position_d, orientation_elbow)

        # Compute elbow's cartesian force with PD control
        force_elbow = np.matmul(self.elbow_stiffness, error_elbow) - np.matmul(self.elbow_damping, elbow_velocity)

        # Map elbow's cartesian force to joint torques
        torque_elbow = np.matmul(J_elbow.T, force_elbow)

        # Create torque vector with zeros and fill torque that can control elbow
        torque_arm = np.zeros(7)
        torque_arm[:3] = torque_elbow
        return torque_arm

    def _nullspace_control(self, q, qdot):
        # Get torque elbow's control
        torque = self._elbow_cartesian_impedance_controller(q, qdot)

        # Get nullspace matrix
        J = self.chain_endeff.jacobian(q, ret_eef_pose=False)[0].numpy()
        #J = self.robot.jacob0(q, end='iiwa_link_7', start='iiwa_link_0')
        nullspace = (np.identity(self.n_joints) - np.matmul(J.T, np.linalg.pinv(J).T))

        # Map elbow's torque to ee's nullspace
        nullspace_torque = np.matmul(nullspace, torque)
        return nullspace_torque

    def get_pose_error(self, position, orientation, position_d, orientation_d):
        error = np.zeros(6)
        error[:3] = position_d - position
        error[3:] = np.zeros((3,))
        return error

    def control_law_vel(self, position_d, orientation_d, ee_pose, J=[]):
        alpha = 30 #100 #100 #100
        beta = np.eye(6)
        beta[:3, :3] = 0.001*np.eye(3)
        beta[3:, 3:] = 0. *np.eye(3)

        # ---------------- initialize ----------------- #
        # Get ee's current position and orientation's from ee's pose
        x_t_pos = ee_pose[:3]
        x_t_orient = UnitQuaternion(ee_pose[3:]) #in quaternions

        # ------------- pure position error control ----------------#
        error_position = self.get_pose_error(x_t_pos, x_t_orient, position_d, UnitQuaternion(orientation_d))

        action = alpha*error_position
        print("alpha*error_position: ", alpha*error_position)
        qdot = np.matmul(np.linalg.pinv(J[0][:, :]), action)
        return qdot