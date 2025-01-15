import torch
import os
import numpy as np
import pytorch_kinematics as pk
from scipy.linalg import block_diag
from pumafabrics.tamed_puma.kinematics.quaternion_operations import QuaternionOperations
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
import casadi as ca

class KinematicsBasics():
    def __init__(self, end_link_name="iiwa_link_7", robot_name="iiwa14", dt=0.01):
        self.robot_name = robot_name
        self.construct_chain(end_link_name=end_link_name)
        self.dt=dt
        self.Jacobian_vec = torch.empty(0, requires_grad=True)
        self.len_max_list=10
        self.Jac_dot_list = []
        self.quaternion_operations = QuaternionOperations()

    def construct_chain(self, end_link_name="iiwa_link_7"):
        """
        To be able to process the urdf, I had to comment out the first line of iiwa7.urdf:
        <!-- ?xml version="1.0" encoding="utf-8"?> -->
        """
        path_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        path_child = path_parent + '/../tamed_puma/config/urdfs/%s.urdf' % self.robot_name
        self.chain = pk.build_serial_chain_from_urdf(open(path_child).read(), end_link_name)

    ### ---------------- Jacobians ------------------###
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

    def diff_kinematics_quat(self, q, angle_quaternion):
        self.Jacobian = self.call_jacobian(q=q)
        H = self.quaternion_operations.map_angular_quat(angle_quaternion=angle_quaternion)
        E_inv = block_diag(np.eye(3), 0.5 * H.transpose())
        J_quat = E_inv @ self.Jacobian.numpy()[0]
        self.J_quat = J_quat
        return J_quat

    def check_instability_jacobian(self, invJ):
        norm_invJ = np.linalg.norm(invJ)
        if norm_invJ > 30:
            print("invJ has a norm that is too large: norm_invJ={}".format(norm_invJ))

    ### ----------------- Forward (differentiable) kinematics -------------------###
    def forward_kinematics(self, q, end_link_name="iiwa_link_7"):
        x_fk = self.chain.forward_kinematics(q, end_only=False)[end_link_name]
        m = x_fk.get_matrix()
        pos = torch.Tensor.numpy(m[:, :3, 3])[0]
        rot = torch.Tensor.numpy(pk.matrix_to_quaternion(m[:, :3, :3]))[0]
        x_pose = np.append(pos, rot)
        return x_pose

    def get_state_velocity(self, q, qdot):
        Jac_current = self.call_jacobian(q)
        ee_velocity = np.matmul(Jac_current, qdot)[0].cpu().detach().numpy()
        return ee_velocity, self.Jacobian

    ### ----------------- Inverse differentiable kinematics -------------------###
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

    def get_qdot_from_linear_velocity(self, q, xdot):
        # forward kinematics:
        self.call_jacobian(q=q)
        # ---- position ----#
        qdot = self.inverse_linear_vel_kinematics(xdot[:3])
        return qdot.numpy()[0]

    def inverse_linear_vel_kinematics(self, xdot):
        """
        Uses only position vectors to compute inverse differentiable kinematics
        """
        invJ = torch.linalg.pinv(self.Jacobian[:, :3, :])
        qdot = invJ @ xdot
        return qdot

    def inverse_diff_kinematics_quat(self, xdot, angle_quaternion):
        """
        xdot must be [vel_linear, vel_quaternion]
        """
        H = self.quaternion_operations.map_angular_quat(angle_quaternion=angle_quaternion)
        E_tot = block_diag(np.eye(3), 2*H)

        invJ_quat = torch.linalg.pinv(self.Jacobian) @ E_tot
        qdot =  invJ_quat @ xdot

        self.check_instability_jacobian(invJ=invJ_quat)
        return qdot

    ### ----------------- Inverse 2nd-order differentiable kinematics -------------------###
    def inverse_2nd_kinematics_quat(self, q, qdot, xddot, angle_quaternion, Jac_prev):
        """
        xdot must be [vel_linear, vel_quaternion]
        """
        H = self.quaternion_operations.map_angular_quat(angle_quaternion=angle_quaternion)
        E_tot = block_diag(np.eye(3), 2*H)
        self.Jac_quat = self.diff_kinematics_quat(q, angle_quaternion)

        invJ_quat = torch.linalg.pinv(self.Jacobian) @ E_tot
        self.Jdot = (self.J_quat - Jac_prev)/self.dt
        qdot =  invJ_quat @ xddot - invJ_quat @ (self.Jdot @ qdot)

        self.check_instability_jacobian(invJ=invJ_quat)
        return qdot, self.Jac_quat, self.Jdot

    def order2_inverse_diff_kinematics_quat(self, xddot, angle_quaternion, qdot):
        H = self.quaternion_operations.map_angular_quat(angle_quaternion=angle_quaternion)
        E_tot = block_diag(np.eye(3), 2*H)
        E_inv = block_diag(np.eye(3), 0.5*H.transpose())

        invJ_quat = torch.linalg.pinv(self.Jacobian) @ E_tot
        J_quat = E_inv @ self.Jacobian.numpy()[0]
        qddot =  invJ_quat @ (xddot - J_quat @ qdot)
        return qddot

    # ---------------------- Symbolic forward kinematics ---------------------------------- #
    def forward_kinematics_symbolic(self, end_link_name="iiwa_link_7", fk=None):
        #x_fk = fk.fk(q=q, parent_link="iiwa_link_0", child_link=end_link_name, positionOnly=False)
        q = fk._q_ca
        x_fk= fk.casadi(q=q,
                        parent_link="iiwa_link_0",
                        child_link=end_link_name)
        pos = x_fk[:3, 3]
        rot_matrix = x_fk[:3, :3]
        quat = self.quaternion_operations.symbolic_rot_matrix_to_quaternions(rot_matrix=rot_matrix)
        x_pose = ca.vcat((pos, quat))
        return x_pose