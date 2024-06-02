import numpy as np
import torch
import casadi as ca

class energy_regulation():
    def __init__(self, dim_task=2, dof=7, mode_NN="2nd", dynamical_system=None):
        self.dim_task=dim_task
        self.dof = dof
        self.mode_NN = mode_NN
        self.potential_NN = dynamical_system.model.potential_from_encoder
        self.dxdq_fun = None

    def gpu_to_cpu(self, x_gpu):
        x_cpu = x_gpu.cpu().detach().numpy()
        return x_cpu

    def relationship_dq_dx(self, offset_orientation, translation_cpu, kuka_kinematics, normalizations, fk):
        # qq = ca.SX.sym("q", 7, 1)
        qq = fk._q_ca
        x_pose = kuka_kinematics.forward_kinematics_symbolic(fk=fk)
        x_NN = normalizations.normalize_pose_to_NN([x_pose], translation_cpu, offset_orientation)
        dxdq = ca.jacobian(x_NN[0], qq)
        self.dxdq_fun = ca.Function("q_to_x", [qq], [dxdq], ["q"], ["dxdq"])
        return self.dxdq_fun

    def compute_potential_and_gradient(self, x_t, mode_NN="2nd", dxdq=None, q=None):
        """
        Compute the gradient of the potential function at position x_t
        """
        if torch.is_tensor(x_t):
            if mode_NN=="2nd":
                x_t_zero_vel = x_t
                x_t_zero_vel[0][self.dim_task:] = torch.zeros((1, self.dim_task))
            else:
                x_t_zero_vel = x_t
        else:
            print("x_t must be torch tensor")
            x_t_zero_vel = torch.FloatTensor([x_t]).cuda() #todo adapt!!

        #---  gradient using Jacobian function of Pytorch ---#
        with torch.no_grad():
            potential_torch = self.potential_NN(x_t_zero_vel)
            grad_potential_torch = torch.autograd.functional.jacobian(self.potential_NN, x_t_zero_vel)
            grad_potential = self.gpu_to_cpu(grad_potential_torch[0][0][0])
        if self.dxdq_fun is not None:
            dxdq = self.read_dqdx(q=q)
            grad_potential_q = grad_potential[0:self.dim_task] @ dxdq
            return potential_torch, grad_potential_q
        else:
            return potential_torch, grad_potential

    def compute_energy_regulator(self, x_t_NN, qdot, M, beta=0., alpha=0.05, mode_NN="2nd", dxdq=None, q=None):
        """get energy regulator by NN via theorem III.5 in https://arxiv.org/pdf/2309.07368.pdf"""
        _, grad_potential = self.compute_potential_and_gradient(x_t_NN, mode_NN=mode_NN, dxdq=dxdq, q=q)

        fraction_denominator = qdot @ M @ qdot + 0.000001
        if fraction_denominator == 0:
            fraction_term = (np.outer(qdot, qdot)) / 1
        else:
            fraction_term = (np.outer(qdot, qdot)) / fraction_denominator
        energy_regulator = -alpha * fraction_term @ grad_potential[0:self.dim_task] - beta * qdot
        return energy_regulator


    def energized_system(self, qdot, M_lambda, h_tilde, f, beta=0, gamma=0):
        """
        Energize a random system with (xddot + f() = 0) where f can change every timestep.
        Using proposition III.3 in https://arxiv.org/pdf/2309.07368.pdf
        """
        denumerator = qdot @ M_lambda @ qdot + 0.00001
        if denumerator == 0:
            alpha_f = 0
        else:
            alpha_f = - (qdot @ M_lambda @ f) / denumerator
        lambdda = gamma * alpha_f
        f_energized = f + lambdda * qdot - beta * qdot
        system_energized = h_tilde + f_energized
        return system_energized

    def read_dqdx(self, q):
        dxdq_dict = self.dxdq_fun(q=q)
        dxdq = dxdq_dict["dxdq"].full()
        return dxdq

    def compute_action_theorem_III5(self, q, qdot, qddot_attractor, action_avoidance, M_avoidance, transition_info, weight_attractor=0.25):
        energy_regulator = self.compute_energy_regulator(transition_info["desired state"], qdot,
                                                         M_avoidance,
                                                         mode_NN=self.mode_NN,
                                                         q=q)
        h_f_energized = self.energized_system(qdot=qdot, M_lambda=M_avoidance,
                                              h_tilde=action_avoidance,
                                              f=weight_attractor * qddot_attractor[0:self.dof])
        action_fabrics_safeMP = h_f_energized + energy_regulator
        return action_fabrics_safeMP