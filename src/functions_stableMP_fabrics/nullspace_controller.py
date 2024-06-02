"""
Authors:
    Micah Prendergast <j.m.prendergast@tudelft.nl>
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
"""

import numpy as np
from spatialmath import SO3
import os
import pytorch_kinematics as pk
import torch
from roboticstoolbox.robot.ERobot import ERobot
from functions_stableMP_fabrics.iiwa_robotics_toolbox import iiwa

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
        path_child = path_parent + '/examples/urdfs/%s.urdf' % self.robot_name
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
        # q_desired = [-0.01850674,  1.18335387, -0.59194822, -1.70635513, -0.49675492, -1.24385178, -0.89709277]
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

    # def control_law(self, position_d, orientation_d, ee_pose, ee_velocity, elbow_pose, elbow_velocity, q):
    #     # Get ee's positiona and orientation's from ee's pose
    #     ee_position = ee_pose[:3]
    #     ee_orientation = ee_pose[3:]
    #
    #     # Map ee's RPY orientation to matrix
    #     ee_orientation = SO3.RPY(ee_orientation)
    #
    #     # Get pose error
    #     error = self.get_pose_error(ee_position, ee_orientation, position_d, orientation_d)
    #
    #     # Compute ee's force with PD control
    #     F_ext = np.matmul(self.ee_stiffness, error) - np.matmul(self.ee_damping, ee_velocity)
    #
    #     # Get ee's jacobian
    #     J = self.robot.jacob0(q, end='iiwa_link_7', start='iiwa_link_0')
    #
    #     # Map ee's force to joint torques
    #     tau_ee = np.matmul(J.transpose(), F_ext)
    #
    #     # Get nullspace torque
    #     tau_nullspace = self._nullspace_control(J, elbow_pose, elbow_velocity, q)
    #
    #     # Add ee's tau with nullspace tau
    #     tau = tau_ee + tau_nullspace

# def _nullspace_control(self, q=np.zeros((7,)), orientation=np.array([1., 0., 0., 0.]), q_neutral=np.zeros((7,)),
#                        order="1st"):
#     order_float = float(order[0])
#     J = self.diff_kinematics_quat(q, orientation) + 0.00000001
#     dof = len(q)
#     action = (q_neutral - q) / (self.dt ** order_float)
#
#     # Get nullspace matrix
#     nullspace = (np.identity(dof) - np.matmul(J.T, np.linalg.pinv(J).T))
#
#     nullspace_action = np.matmul(nullspace, action)
#     return nullspace_action