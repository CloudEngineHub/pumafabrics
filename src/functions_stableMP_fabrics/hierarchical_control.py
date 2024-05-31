import casadi as ca
import numpy as np
import copy
class hierarchical_controller():
    def __init__(self, Jacobian_tasks=None):
        self.dof = 7
        self.max_order = len(Jacobian_tasks)
        self.M = np.eye(self.dof)
        self.n_tasks = [arr.shape[1] for arr in Jacobian_tasks]
        self.Jacobian_tasks = Jacobian_tasks

    def construct_hierarchical_jacobian(self, Jacobian_tasks=None):
        if Jacobian_tasks is not None:
            self.Jacobian_tasks = Jacobian_tasks

        J_list = []
        J_prev = []
        for i in range(self.max_order):
            if i>0:
                J_prev = copy.deepcopy(self.Jacobian_tasks[0])
            #equation 3.11 in NTNU project thesis
            J_i = self.Jacobian_tasks[i]
            N_i = self.null_space_projector(J_prev=J_prev, order=i)
            J_task_i = np.dot(J_i, N_i.transpose())

            # lists and next step:
            J_list.append(J_i)
        J_hierarchical = ca.vcat(J_list)
        return J_hierarchical

    def null_space_projector(self, J_prev, order=0):
        if order == 0:
            dof_i = self.n_tasks[order]
            N_i = np.eye(dof_i)
        else:
            dof_i = self.n_tasks[order]
            J_minv = self.dyn_consistent_pseudoinverse(J_prev)
            N_i = np.eye(dof_i) - np.dot(J_prev.transpose(), J_minv.transpose())
        return N_i

    def dyn_consistent_pseudoinverse(self, Jac):
        invM = np.linalg.inv(self.M)
        J_T = Jac.transpose()
        invMJt = np.dot(invM, J_T)
        J_Minv = np.dot(invM, np.dot(J_T, np.linalg.inv(np.dot(Jac, invMJt))))
        return J_Minv


if __name__ == '__main__':
    # --- test example ----
    J_0 = np.random.rand(6, 7)
    J_1 = np.eye(7)
    Jac_tasks = [J_0, J_1]
    h_controller = hierarchical_controller(Jac_tasks)
    J_hierarchical = h_controller.construct_hierarchical_jacobian()
    print("J_Hierarchical: ", J_hierarchical)

