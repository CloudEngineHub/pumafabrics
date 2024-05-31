import torch
import pytorch_kinematics as pk
import os
import numpy as np
import copy
import math
from scipy.linalg import block_diag
import spatial_casadi as sc
import casadi as ca
from functions_stableMP_fabrics.filters import ema_filter_deriv

class KinematicsKuka(object):
    def __init__(self, end_link_name="iiwa_link_ee", robot_name="iiwa14", dt=0.01):
        self.robot_name = robot_name
        self.construct_chain(end_link_name=end_link_name)
        self.dt=dt
        self.Jacobian_vec = torch.empty(0, requires_grad=True)
        self.len_max_list=10
        self.Jac_dot_list = []

    def construct_chain(self, end_link_name="iiwa_link_ee"):
        """
        To be able to process the urdf, I had to comment out the first line of iiwa7.urdf:
        <!-- ?xml version="1.0" encoding="utf-8"?> -->
        """
        path_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        path_child = path_parent + '/examples/urdfs/%s.urdf' % self.robot_name
        self.chain = pk.build_serial_chain_from_urdf(open(path_child).read(), end_link_name)

    def call_jacobian(self, q):
        self.Jacobian, self.pose = self.chain.jacobian(q, ret_eef_pose=True)

        self.Jacobian_vec = self.update_J_vec()
        if len(self.Jacobian_vec)>1:
            self.Jacobian_grad_tot = torch.gradient(self.Jacobian_vec, axis=0)
            self.Jacobian_grad = self.Jacobian_grad_tot[-1][-1]
        else:
            self.Jacobian_grad = torch.zeros((6, 7))
        return self.Jacobian

    def update_J_vec(self):
        self.Jacobian_vec = torch.cat((self.Jacobian_vec, self.Jacobian), 0)
        if len(self.Jacobian_vec)>self.len_max_list:
            self.Jacobian_vec = self.Jacobian_vec[-self.len_max_list:]
        return self.Jacobian_vec

    def forward_kinematics(self, q, end_link_name="iiwa_link_ee"):
        x_fk = self.chain.forward_kinematics(q, end_only=False)[end_link_name]
        m = x_fk.get_matrix()
        pos = torch.Tensor.numpy(m[:, :3, 3])[0]
        rot = torch.Tensor.numpy(pk.matrix_to_quaternion(m[:, :3, :3]))[0]
        x_pose = np.append(pos, rot)
        return x_pose

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

    def get_state_velocity(self, q, qdot):
        Jac_current = self.call_jacobian(q)
        ee_velocity = np.matmul(Jac_current, qdot)[0].cpu().detach().numpy()
        return ee_velocity, self.Jacobian

    def map_angular_quat(self, angle_quaternion):
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

    def inverse_diff_kinematics(self, xdot):
        """"
        Uses position + velocity
        y vectors to compute inverse differentiable kinematics
        """
        invJ = torch.linalg.pinv(self.Jacobian)
        qdot = invJ @ xdot

        #check if unstable:
        self.check_instability_jacobian(invJ=invJ)
        return qdot

    def diff_kinematics_quat(self, q, angle_quaternion):
        self.Jacobian = self.call_jacobian(q=q)
        H = self.map_angular_quat(angle_quaternion=angle_quaternion)
        E_inv = block_diag(np.eye(3), 0.5 * H.transpose())
        J_quat = E_inv @ self.Jacobian.numpy()[0]
        self.J_quat = J_quat
        return J_quat

    def inverse_diff_kinematics_quat(self, xdot, angle_quaternion):
        """
        xdot must be [vel_linear, vel_quaternion]
        """
        H = self.map_angular_quat(angle_quaternion=angle_quaternion)
        E_tot = block_diag(np.eye(3), 2*H)

        invJ_quat = torch.linalg.pinv(self.Jacobian) @ E_tot
        qdot =  invJ_quat @ xdot

        self.check_instability_jacobian(invJ=invJ_quat)
        return qdot

    def inverse_2nd_kinematics_quat(self, q, qdot, xddot, angle_quaternion, Jac_prev):
        """
        xdot must be [vel_linear, vel_quaternion]
        """
        H = self.map_angular_quat(angle_quaternion=angle_quaternion)
        E_tot = block_diag(np.eye(3), 2*H)
        self.Jac_quat = self.diff_kinematics_quat(q, angle_quaternion)

        invJ_quat = torch.linalg.pinv(self.Jacobian) @ E_tot

        # self.Jac_dot_list.append(self.Jac_quat)
        # # if len(self.Jac_dot_list) > 10:
        #     self.Jac_dot_list.pop(0)
        # self.Jdot = ema_filter_deriv(np.array(self.Jac_dot_list), alpha=0.5, dt=self.dt)
        # alpha = 0.9
        # self.J_dot = (self.J_quat - Jac_dot_prev)/self.dt
        # self.Jdot = (1-alpha)*Jac_dot_prev + alpha*self.J_dot
        # self.Jdot = ((1-alpha)*Jac_prev + alpha*self.Jac_quat)/self.dt
        self.Jdot = (self.J_quat - Jac_prev)/self.dt
        qdot =  invJ_quat @ xddot - invJ_quat @ (self.Jdot @ qdot)

        self.check_instability_jacobian(invJ=invJ_quat)
        return qdot, self.Jac_quat, self.Jdot

    def order2_inverse_diff_kinematics_quat(self, xddot, angle_quaternion, qdot):
        H = self.map_angular_quat(angle_quaternion=angle_quaternion)
        E_tot = block_diag(np.eye(3), 2*H)
        E_inv = block_diag(np.eye(3), 0.5*H.transpose())

        invJ_quat = torch.linalg.pinv(self.Jacobian) @ E_tot
        J_quat = E_inv @ self.Jacobian.numpy()[0]
        qddot =  invJ_quat @ (xddot - J_quat @ qdot)
        return qddot

    def check_instability_jacobian(self, invJ):
        norm_invJ = np.linalg.norm(invJ)
        if norm_invJ > 30:
            print("invJ has a norm that is too large: norm_invJ={}".format(norm_invJ))


    def inverse_linear_vel_kinematics(self, xdot):
        """
        Uses only position vectors to compute inverse differentiable kinematics
        """
        invJ = torch.linalg.pinv(self.Jacobian[:, :3, :])
        qdot = invJ @ xdot
        return qdot

    def get_qdot_from_linear_angular_vel(self, q, xdot, x_pose):
        """
        xdot = [linear velocity, angle velocity in quaternions], [7x1]
        """
        # forward kinematics:
        self.Jacobian = self.call_jacobian(q=q)

        # # differentiable inverse kinematics
        xdot_euler = self.quat_vel_to_angular_vel(angle_quaternion=x_pose[3:], vel_quaternion=xdot[3:])
        xdot = np.append(xdot[:3], xdot_euler/self.dt)

        # ---- position + orientation ---#
        qdot = self.inverse_diff_kinematics(xdot)

        return qdot.numpy()[0]

    def get_qdot_from_linear_velocity(self, q, xdot):
        # forward kinematics:
        self.call_jacobian(q=q)
        # ---- position ----#
        qdot = self.inverse_linear_vel_kinematics(xdot[:3])
        return qdot.numpy()[0]

    def get_initial_pose(self, q_init, offset_orientation):
        # initial state pose:
        x_t_init_pose = self.forward_kinematics(q_init)
        x_t_init_pose[3:7] = self.check_quaternion_initial(x_orientation=x_t_init_pose[3:], quat_offset=offset_orientation)
        xee_orientation = x_t_init_pose[3:7]
        return x_t_init_pose, xee_orientation

    def get_pose(self, q, quat_prev):
        # --- End-effector state ---#
        x_t_pose = self.forward_kinematics(q)
        x_t_pose[3:7] = self.check_quaternion_flipped(quat=x_t_pose[3:7], quat_prev=quat_prev)
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
            xdot_t_pose = np.append(xdot_t[0:3], self.angular_vel_to_quat_vel(angle_quaternion=xee_orientation, vel_angular=xdot_t[3:]))
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
            xdot_t_pose = np.append(xdot_t[0:3], self.angular_vel_to_quat_vel(angle_quaternion=xee_orientation, vel_angular=xdot_t[3:7]))
            x_t = np.array([np.append(x_t_pose, xdot_t_pose)])
        return x_t, xee_orientation, xdot_t


    # ---------------------- check flips of quaternions ---------------------------------- #
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

    def forward_kinematics_symbolic(self, end_link_name="iiwa_link_ee", fk=None):
        #x_fk = fk.fk(q=q, parent_link="iiwa_link_0", child_link=end_link_name, positionOnly=False)
        q = fk._q_ca
        x_fk= fk.casadi(q=q,
                        parent_link="iiwa_link_0",
                        child_link=end_link_name)
        pos = x_fk[:3, 3]
        rot_matrix = x_fk[:3, :3]
        quat = self.symbolic_rot_matrix_to_quaternions(rot_matrix=rot_matrix)
        x_pose = ca.vcat((pos, quat))
        return x_pose

    def symbolic_rot_matrix_to_quaternions(self, rot_matrix):
        r = sc.Rotation.from_matrix(rot_matrix)
        quatern = r.as_quat()
        quat = [quatern[3], quatern[0], quatern[1], quatern[2]]
        return quatern

if __name__ == "__main__":
    # input parameters:
    q = torch.tensor([0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0])
    xdot = np.array([0.2, 0.5, 0.2, 1., 0., 0., 0.])

    kuka_kinematics = KinematicsKuka()
    qdot = kuka_kinematics.get_qdot_from_linear_angular_vel(q=q, xdot=xdot)
    qdot_from_pos = kuka_kinematics.get_qdot_from_linear_velocity(q=q, xdot=xdot[3:])

    print("qdot:", qdot)
    print("qdot_pos:", qdot_from_pos)
