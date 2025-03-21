import numpy as np
import copy
import torch
from pumafabrics.puma_adapted.agent.utils.dynamical_system_operations import normalize_state, denormalize_state
from pumafabrics.tamed_puma.kinematics.quaternion_operations import QuaternionOperations
from spatialmath import UnitQuaternion

class denormalizations():
    def __init__(self, x_min, x_max, dof_task=0, dim_pos=3, dt=0.01, mode_NN="1st", min_vel=[], max_vel=[], learner=None):
        self.min_state = x_min
        self.max_state = x_max
        if dof_task == 0:
            self.dof_task = len(self.min_state)
            self.dim_pos = self.dof_task
        else:
            self.dof_task = dof_task
            self.dim_pos = dim_pos
        self.scaling_factor = self.max_state  - self.min_state
        self.dt = dt
        # self.mode = mode
        self.mode_NN = mode_NN
        if learner is not None:
            self.min_vel = self.gpu_to_cpu(learner.min_vel).transpose()[0]
            self.max_vel = self.gpu_to_cpu(learner.max_vel).transpose()[0]
        else:
            self.min_vel = min_vel
            self.max_vel = max_vel
        self.quaternion_operations = QuaternionOperations()

    # ---- GPU CPU conversions ---#
    def cpu_to_gpu(self, x_cpu):
        if type(x_cpu) == list:
            x_cpu = np.stack(x_cpu)
        x_gpu = torch.FloatTensor(x_cpu).cuda()
        return x_gpu

    def gpu_to_cpu(self, x_gpu):
        x_cpu = x_gpu.T.cpu().detach().numpy()
        return x_cpu

    # ---- denormalizations ------#
    def denormalize_action(self, action_value):
        """
        todo: make this function more general. why not action_scaled[2]???
        """
        action_scaled = copy.deepcopy(action_value)
        action_scaled[0] = action_value[0]*self.scaling_factor[0]
        action_scaled[1] = action_value[1]*self.scaling_factor[1]
        return action_scaled

    def reverse_translation(self, xy_NN):
        """
        Reversed translation.

        """
        xy_detranslated = xy_NN + self.translation
        return xy_detranslated

    def reverse_transformation(self, action_gpu, mode_NN=None):
        """
        Transform from normalized actions --> system actions. Actions can be velocities or accelerations.
        """
        if mode_NN is None:
            mode_NN = self.mode_NN

        action_cpu = np.zeros((self.dof_task, ))
        action_t = self.gpu_to_cpu(action_gpu)
        action_cpu[0:self.dof_task] = action_t[0:self.dof_task].T

        # unscale wrt time-step:
        if mode_NN == "2nd":
            action_Transformed = action_cpu / (self.dt*self.dt)
        elif mode_NN == "1st":
            action_Transformed = action_cpu / self.dt

        # unscale wrt room size
        action_Transformed[0:self.dof_task] = self.denormalize_action(action_Transformed[0:self.dof_task])
        return action_Transformed

    def reverse_transformation_position(self, position_gpu):
        """
        Transform normalized position to system positions.
        """
        position_cpu = self.gpu_to_cpu(position_gpu).transpose()
        pos_detranslated = self.reverse_translation(position_cpu)
        pos = denormalize_state(pos_detranslated, x_min=self.min_state, x_max=self.max_state)
        return pos

    # ------------------------------- in case of quaternions ---------------------------------- #
    def NN_quat_to_system(self, quat, offset):
        quat2 = UnitQuaternion(quat, check=False)
        if self.quaternion_flipped(quat=quat2.A, quat_prev=quat):
            quat2 = quat2 * -1
            # print("flipped 2!!")

        quat_system = quat2 * UnitQuaternion(offset)
        quat_system = self.quaternion_operations.quat_product(quat, offset)

        # ---- checking the norms ---#
        self.check_norm_quaternion(quat_list = [quat_system, quat2, offset])
        return quat_system #.A

    def reverse_transformation_pos_quat(self, state_gpu, offset_orientation):
        """
        Transform normalized poses (position + quaternion) to system poses.
        """
        state_cpu = self.gpu_to_cpu(state_gpu).transpose()[0]

        # position
        pose_detranslated = self.reverse_translation(state_cpu[0:self.dof_task])
        position_denormalized = denormalize_state(pose_detranslated[:3], x_min=self.min_state[:3], x_max=self.max_state[:3])

        #quaternion:
        quat_denormalized = self.NN_quat_to_system(state_cpu[3:7], offset_orientation)

        pose = np.append(position_denormalized, quat_denormalized)
        return pose

    # --------------------------- check norm quaternion -----------------------------------#
    def check_norm_quaternion(self, quat=None, quat_list=None):
        """
        Check if the norm of the quaternion is equal to 1, otherwise print a warning message
        """
        if quat_list is not None:
            for quat in quat_list:
                if isinstance(quat, np.ndarray):
                    len_quat = np.linalg.norm(quat)
                    if np.linalg.norm(len_quat - 1.) >= 1e-2:
                        print("warning: length of quaternion checked is not 1.!")
        elif quat is not None and isinstance(quat, np.ndarray):
            len_quat = np.linalg.norm(quat)
            if np.linalg.norm(len_quat - 1.) >= 1e-2:
                print("warning: length of quaternion checked is not 1.!")
        else:
            print("warning: provide a valid argument for check_norm_quaternion, either np_array or list of np_arrays")

    def quaternion_flipped(self, quat, quat_prev):
        dist_quat = np.linalg.norm(quat - quat_prev)
        if dist_quat > 1.0:
            return True
        else:
            return False