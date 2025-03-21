import numpy as np
import copy
import torch
from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import normalize_state, denormalize_state
from spatialmath import SO3, UnitQuaternion, Quaternion
import casadi as ca
from pumafabrics.tamed_puma.utils.denormalizations import denormalizations

class normalization_functions(denormalizations):
    '''
    Functions to normalize and denormalize the states and actions (positions/quaternions/velocities).
    '''
    def __init__(self, x_min, x_max, dof_task=0, dim_pos=3, dt=0.01, mode_NN="1st", min_vel=[], max_vel=[], learner=None):
        super().__init__(x_min, x_max, dof_task=dof_task, dim_pos=dim_pos, dt=dt, mode_NN=mode_NN, min_vel=min_vel,
                         max_vel=max_vel, learner=learner)

    def call_normalize_state(self, state):
        state_normalized = normalize_state(state, x_min=self.min_state[:len(state)], x_max=self.max_state[:len(state)])
        return state_normalized

    def normalize_to_NN(self, state_value, x_min, x_max):
        state_scaled = copy.deepcopy(state_value)
        state_scaled[0:self.dim_pos] = normalize_state(state_value[0:self.dim_pos], x_min=x_min[:self.dim_pos], x_max=x_max[:self.dim_pos])
        return state_scaled

    def get_translation(self, goal_pos, goal_pos_NN):
        """
        Store the translation between the goal in real-coordinates and the goal in the neural network
        """
        translation = goal_pos - np.array(goal_pos_NN)
        self.translation = translation
        return translation

    def translation_to_NN(self, xy_sim, translation):
        """
        Translate to states of the neural network
        """
        xy_translated = xy_sim - translation
        return xy_translated

    def transformation_to_NN(self, x_t, translation_cpu):
        """
        Transform system states to normalized states in the neural network
        """
        x_t_NN = copy.deepcopy(x_t)

        #scale wrt room size:
        x_t_NN[0] = self.normalize_to_NN(x_t[0], x_min=self.min_state, x_max=self.max_state)

        # translation wrt goal
        x_t_NN[0][0:self.dof_task] = self.translation_to_NN(x_t_NN[0][0:self.dof_task], translation=translation_cpu)
        return x_t_NN

    def transformation_to_NN_vel(self, v_t):
        vel_transformed = copy.deepcopy(v_t)

        # scale wrt room size
        vel_transformed[0] = v_t[0] / self.scaling_factor[0] #v_x
        vel_transformed[1] = v_t[1] / self.scaling_factor[1] #v_y

        #scale wrt time-step, scale wrt min and max vel of NN:
        vel_transformed = normalize_state(vel_transformed * self.dt, self.min_vel, self.max_vel)
        return vel_transformed

    def reverse_transformation_position(self, position_gpu):
        """
        Transform normalized position to system positions.
        """
        position_cpu = self.gpu_to_cpu(position_gpu).transpose()
        pos_detranslated = self.reverse_translation(position_cpu)
        pos = denormalize_state(pos_detranslated, x_min=self.min_state, x_max=self.max_state)
        return pos

    # ------------------------------- in case of quaternions ---------------------------------- #
    def get_offset_quaternion(self, goal_quat_desired, goal_NN_quat):
        """
        Quaternion orientation offset of the goal wrt the goal of the NN (e.g. [1, 0, 0, 0]).
        """
        offset = UnitQuaternion(goal_quat_desired) / UnitQuaternion(goal_NN_quat)

        # ---- checking the norms ---#
        self.check_norm_quaternion(quat_list = [goal_NN_quat, goal_quat_desired, offset])
        return offset.A

    def system_quat_to_NN(self, quat, offset):
        offset_inverse = UnitQuaternion(offset).inv()
        if torch.is_tensor(quat):
            quat2 = self.gpu_to_cpu(quat)
            quat_NN = self.quaternion_operations.quat_product(quat2, offset_inverse.A)
        else:
            quat_NN = self.quaternion_operations.quat_product(quat, offset_inverse.A)

        # ---- checking the norms ---#
        self.check_norm_quaternion(quat_list = [quat_NN, offset])
        return quat_NN

    # -------------------- High level functions -----------------------------#
    def translation_goal(self, state_goal, goal_NN):
        # Translation of goal:
        goal_normalized = self.call_normalize_state(state=state_goal)
        translation = self.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN)
        if len(translation)>3:
            translation[3:] = np.zeros(4)
        translation_gpu = torch.FloatTensor(translation).cuda()
        return translation_gpu, translation

    def normalize_pose_to_NN(self, x_t, translation_cpu, offset_orientation):
        # transformations
        x_cpu = self.transformation_to_NN(x_t=x_t, translation_cpu=translation_cpu)
        # quaternion offset
        x_cpu[0][3:7] = self.system_quat_to_NN(quat=x_cpu[0][3:7], offset=offset_orientation)
        return x_cpu

    def normalize_vel_to_NN(self, x_t, offset_orientation):
        offset_inverse = UnitQuaternion(offset_orientation).inv()
        x_t[0][-4:] = self.quaternion_operations.quat_product(x_t[0][-4:], offset_inverse.A)
        x_vel_cpu = self.transformation_to_NN_vel(v_t=x_t[0][self.dof_task:self.dof_task * 2])
        return x_vel_cpu

    def normalize_state_to_NN(self, x_t, translation_cpu, offset_orientation):
        # normalize pose (position+orientation)
        x_cpu = self.normalize_pose_to_NN(x_t, translation_cpu, offset_orientation)
        x_gpu = self.cpu_to_gpu(x_cpu)

        # --- if state=(pose, vel) also normalize the velocities) ---#
        if self.mode_NN == "2nd":
            x_vel_cpu = self.normalize_vel_to_NN(x_t, offset_orientation)
            x_gpu[0][self.dof_task:self.dof_task*2] = torch.cuda.FloatTensor(x_vel_cpu)
            return x_gpu
        else:
            return x_gpu

    def normalize_state_position_to_NN(self, x_t, translation_cpu):
        # normalize pose (position+orientation)
        x_cpu = self.transformation_to_NN(x_t=x_t, translation_cpu=translation_cpu)
        x_gpu = self.cpu_to_gpu(x_cpu)

        # --- if state=(pose, vel) also normalize the velocities) ---#
        if self.mode_NN == "2nd":
            x_vel_cpu = self.transformation_to_NN_vel(v_t=x_t[0][self.dof_task:self.dof_task * 2])
            x_gpu[0][self.dof_task:self.dof_task*2] = torch.cuda.FloatTensor(x_vel_cpu)
            return x_cpu, x_gpu
        else:
            return x_cpu, x_gpu

    def check_quaternion_flipped(self, quat, quat_prev):
        dist_quat = np.linalg.norm(quat - quat_prev)
        if dist_quat > 1.0:
            quat_new = -1*copy.deepcopy(quat)
        else:
            quat_new = copy.deepcopy(quat)
        return quat_new

