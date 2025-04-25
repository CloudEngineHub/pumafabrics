import torch
import pytorch_kinematics as pk
import numpy as np
import spatial_casadi as sc
from scipy.linalg import block_diag
import copy
import casadi as ca

class QuaternionOperations():
    def __init__(self):
        pass

    def quat_product(self, p, q):
        p_w = p[0]
        q_w = q[0]
        p_v = p[1:4]
        q_v = q[1:4]

        if isinstance(p, np.ndarray):
            pq_w = p_w*q_w - np.matmul(p_v, q_v)
            pq_v = p_w*q_v + q_w*p_v + np.cross(p_v, q_v)
            pq = np.append(pq_w, pq_v)
        elif isinstance(p, ca.SX):
            pq_w = p_w*q_w - ca.dot(p_v, q_v)
            pq_v = p_w*q_v + q_w*p_v + ca.cross(p_v, q_v)
            pq = ca.vcat((pq_w, pq_v))
        else:
            print("no matching type found in quat_product")
        return pq

    def quat_to_rot_matrix(self, quat):
        if type(quat) == list:
            quat = torch.FloatTensor(quat).cuda()
        rot_matrix = pk.quaternion_to_matrix(quat)
        return rot_matrix.cpu().detach().numpy()

    def rot_matrix_to_quat(self, rot_matrix):
        if type(rot_matrix) == list or type(rot_matrix) == np.ndarray:
            rot_matrix = torch.FloatTensor(rot_matrix).cuda()
        quat = pk.matrix_to_quaternion(rot_matrix)
        return quat.T.cpu().detach().numpy()

    def quat_vel_with_offset(self, quat_vel_NN, quat_offset):
        a1 = quat_vel_NN[0]
        v1 = quat_vel_NN[1:]
        a2 = quat_offset[0]
        v2 = quat_offset[1:]

        qv_a = a1*a2 - np.dot(v1, v2)
        qv_v = a1*v2 + a2*v1 + np.cross(v1, v2)
        qv = np.append(qv_a, qv_v)
        return qv

    def map_angular_quat(self, angle_quaternion):
        q0 = angle_quaternion[0]
        q1 = angle_quaternion[1]
        q2 = angle_quaternion[2]
        q3 = angle_quaternion[3]
        H = np.array([[-q1, q0, -q3, q2],
                   [-q2, q3, q0, -q1],
                   [-q3, -q2, q1, q0]
                   ])
        return H

    def quat_vel_to_angular_vel(self, angle_quaternion, vel_quaternion):
        """
        Quaternion velocities to angular velocities,
        Slight modification of book:
        https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/RD2016script.pdf
        """
        H = self.map_angular_quat(angle_quaternion=angle_quaternion)
        angular_vel = 2*np.dot(H, vel_quaternion)
        return angular_vel

    def angular_vel_to_quat_vel(self, angle_quaternion, vel_angular):
        H = self.map_angular_quat(angle_quaternion=angle_quaternion)
        quat_vel = 0.5*np.dot(H.transpose(), vel_angular)
        return quat_vel

    def symbolic_rot_matrix_to_quaternions(self, rot_matrix):
        r = sc.Rotation.from_matrix(rot_matrix)
        quatern = r.as_quat()
        quat = [quatern[3], quatern[0], quatern[1], quatern[2]]
        return quatern

    def symbolic_rot_matrix_to_euler(self, rot_matrix):
        r = sc.Rotation.from_matrix(rot_matrix)
        euler = r.as_euler("xyz", degrees=False)
        return euler

    # --------------------------- check flips of quaternions -------------------------------#
    def check_quaternion_flipped(self, quat, quat_prev):
        dist_quat = np.linalg.norm(quat - quat_prev)
        if dist_quat > 1.0:
            quat_new = -1*copy.deepcopy(quat)
            "flip quaternion!"
        else:
            quat_new = copy.deepcopy(quat)
        return quat_new

    def check_quaternion_initial(self, x_orientation, quat_offset):
        """Check that we start in hemisphere nearest to goal"""

        #orientations (flipped and unflipped)
        orientation_1 = copy.deepcopy(x_orientation)
        orientation_2 = -copy.deepcopy(orientation_1)

        # Calculate the dot product
        dot_product_1 = np.dot(orientation_1, quat_offset)
        dot_product_2 = np.dot(orientation_2, quat_offset)

        #check angular distance:
        angular_distance_1 = np.arccos(dot_product_1)
        angular_distance_2 = np.arccos(dot_product_2)

        if angular_distance_1 < angular_distance_2:
            quat_new = copy.deepcopy(orientation_1)
            print("Not flipped:  ", angular_distance_1)
        else:
            quat_new = copy.deepcopy(orientation_2)
            print("Flipped: ", angular_distance_2)

        return quat_new