import torch
import pytorch_kinematics as pk
import os
import numpy as np
import copy
import math
from scipy.linalg import block_diag
import spatial_casadi as sc
import casadi as ca
from pumafabrics.tamed_puma.kinematics.quaternion_operations import QuaternionOperations
from pumafabrics.tamed_puma.kinematics.kinematics_basics import KinematicsBasics

class KinematicsKuka(KinematicsBasics):
    def __init__(self, end_link_name="iiwa_link_7", robot_name="iiwa14", dt=0.01):
        self.end_link_name = end_link_name
        super().__init__(end_link_name, robot_name, dt)

    def get_initial_pose(self, q_init, offset_orientation):
        # initial state pose:
        x_t_init_pose = self.forward_kinematics(q_init, end_link_name=self.end_link_name)
        x_t_init_pose[3:7] = self.quaternion_operations.check_quaternion_initial(x_orientation=x_t_init_pose[3:], quat_offset=offset_orientation)
        xee_orientation = x_t_init_pose[3:7]
        return x_t_init_pose, xee_orientation

    def get_pose(self, q, quat_prev):
        # --- End-effector state ---#
        x_t_pose = self.forward_kinematics(q, end_link_name=self.end_link_name)
        x_t_pose[3:7] = self.quaternion_operations.check_quaternion_flipped(quat=x_t_pose[3:7], quat_prev=quat_prev)
        xee_orientation = x_t_pose[3:7]
        return x_t_pose, xee_orientation

    def get_initial_state_task(self, q_init, offset_orientation, mode_NN="1st", qdot_init=None):
        # initial state pose:
        x_t_init_pose, xee_orientation = self.get_initial_pose(q_init, offset_orientation)
        if mode_NN == '1st': #"1st order"
            return x_t_init_pose
        else: #"2nd order"
            # end-effector quaternion velocity:
            xdot_t, _ = self.get_state_velocity(q_init, qdot_init)
            xdot_t_pose = np.append(xdot_t[0:3], self.quaternion_operations.angular_vel_to_quat_vel(angle_quaternion=xee_orientation, vel_angular=xdot_t[3:]))
            x_t_init = np.append(x_t_init_pose, xdot_t_pose)
            return x_t_init

    def get_state_task(self, q, quat_prev, mode_NN="1st", qdot=None):
        # --- End-effector state ---#
        x_t_pose, xee_orientation = self.get_pose(q, quat_prev)
        if mode_NN == "1st":
            x_t = np.array([x_t_pose])
            xdot_t = []
        else:
            # end-effector quaternion velocity:
            xdot_t,  _ = self.get_state_velocity(q, qdot)
            xdot_t[3:7] = np.clip(xdot_t[3:7], np.array([-2, -2, -2]), np.array([2, 2, 2]))
            xdot_t_pose = np.append(xdot_t[0:3], self.quaternion_operations.angular_vel_to_quat_vel(angle_quaternion=xee_orientation, vel_angular=xdot_t[3:7]))
            x_t = np.array([np.append(x_t_pose, xdot_t_pose)])
        return x_t, xee_orientation, xdot_t

if __name__ == "__main__":
    kuka_kinematics = KinematicsKuka()
