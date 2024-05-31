import numpy as np
import copy
import torch
from agent.utils.dynamical_system_operations import normalize_state, denormalize_state
from spatialmath import SO3, UnitQuaternion, Quaternion
import casadi as ca

class normalization_functions():
    '''
    Functions to normalize and denormalize the states and actions (positions/quaternions/velocities).
    '''
    def __init__(self, x_min, x_max, dof_task=0, dim_pos=3, dt=0.01, mode_NN="1st", learner=None):
        self.min_state = x_min
        self.max_state = x_max
        if dof_task == 0:
            self.dof_task = len(self.min_state)
            self.dim_pos = self.dof_task
        else:
            self.dof_task = dof_task
            self.dim_pos = dim_pos
        self.scaling_factor = self.get_scaling_factor()
        self.dt = dt
        # self.mode = mode
        self.mode_NN = mode_NN
        if learner is not None:
            self.min_vel = self.gpu_to_cpu(learner.min_vel).transpose()[0]
            self.max_vel = self.gpu_to_cpu(learner.max_vel).transpose()[0]

    def get_scaling_factor(self):
        """
        Scaling wrt x_min and x_max of the original state space by the NN.
        """
        self.scaling_factor = self.max_state  - self.min_state
        return self.scaling_factor

    def call_normalize_state(self, state):
        state_normalized = normalize_state(state, x_min=self.min_state[:len(state)], x_max=self.max_state[:len(state)])
        return state_normalized

    def denormalize_action(self, action_value):
        """
        todo: make this function more general. why not action_scaled[2]???
        """
        action_scaled = copy.deepcopy(action_value)
        action_scaled[0] = action_value[0]*self.scaling_factor[0]
        action_scaled[1] = action_value[1]*self.scaling_factor[1]
        return action_scaled

    def normalize_to_NN(self, state_value, x_min, x_max):
        state_scaled = copy.deepcopy(state_value)
        state_scaled[0:self.dim_pos] = normalize_state(state_value[0:self.dim_pos], x_min=x_min[:self.dim_pos], x_max=x_max[:self.dim_pos])
        # if self.mode_NN == "2nd": #self.mode == "acc" and  #state is pos + vel
        #     state_scaled[self.dof_task] = state_value[self.dof_task] / self.scaling_factor[0] #v_x
        #     state_scaled[self.dof_task+1] = state_value[self.dof_task+1] / self.scaling_factor[1] #v_y
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

    def reverse_translation(self, xy_NN):
        """
        Reversed translation.

        """
        xy_detranslated = xy_NN + self.translation
        return xy_detranslated

    def transformation_to_NN(self, x_t, translation_cpu, min_vel=list, max_vel=list):
        """
        Transform system states to normalized states in the neural network
        """
        x_t_NN = copy.deepcopy(x_t)

        #scale wrt room size:
        x_t_NN[0] = self.normalize_to_NN(x_t[0], x_min=self.min_state, x_max=self.max_state)

        #scale wrt velocities
        # if self.mode_NN == "2nd": #self.mode == "acc" and
        #     x_t_gpu[0][self.dof_task:self.dof_task*2] = normalize_state(x_t_gpu[0][self.dof_task:self.dof_task*2] * self.dt, min_vel, max_vel)

        # translation wrt goal
        x_t_NN[0][0:self.dof_task] = self.translation_to_NN(x_t_NN[0][0:self.dof_task], translation=translation_cpu)
        return x_t_NN

    def cpu_to_gpu(self, x_cpu):
        if type(x_cpu) == list:
            x_cpu = np.stack(x_cpu)
        x_gpu = torch.FloatTensor(x_cpu).cuda()
        return x_gpu

    def gpu_to_cpu(self, x_gpu):
        x_cpu = x_gpu.T.cpu().detach().numpy()
        return x_cpu

    def transformation_to_NN_vel(self, v_t):
        vel_transformed = copy.deepcopy(v_t)

        # scale wrt room size
        vel_transformed[0] = v_t[0] / self.scaling_factor[0] #v_x
        vel_transformed[1] = v_t[1] / self.scaling_factor[1] #v_y

        #scale wrt time-step, scale wrt min and max vel of NN:
        vel_transformed = normalize_state(vel_transformed * self.dt, self.min_vel, self.max_vel)
        return vel_transformed

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
        #action_Transformed2 = denormalize_state(action_Transformed, dynamical_system.min_vel.T.cpu().detach().numpy().transpose()[0], dynamical_system.max_vel.T.cpu().detach().numpy().transpose()[0])
        action_Transformed[0:self.dof_task] = self.denormalize_action(action_Transformed[0:self.dof_task])
        return action_Transformed

    def reverse_transformation_pos(self, position_gpu):
        """
        Transform normalized position to system positions.
        """
        position_cpu = self.gpu_to_cpu(position_gpu)
        pos_detranslated = self.reverse_translation(position_cpu)
        pos = denormalize_state(pos_detranslated, x_min=self.min_state, x_max=self.max_state)
        return pos

    # ------------------------------- in case of quaternions ---------------------------------- #
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

    def get_offset_quaternion(self, goal_quat_desired, goal_NN_quat):
        """
        Quaternion orientation offset of the goal wrt the goal of the NN (e.g. [1, 0, 0, 0]).
        """
        offset = UnitQuaternion(goal_quat_desired) / UnitQuaternion(goal_NN_quat)

        # ---- checking the norms ---#
        self.check_norm_quaternion(quat_list = [goal_NN_quat, goal_quat_desired, offset])
        return offset.A

    def NN_quat_to_system(self, quat, offset):
        quat2 = UnitQuaternion(quat, check=False)
        if self.quaternion_flipped(quat=quat2.A, quat_prev=quat):
            quat2 = quat2 * -1
            # print("flipped 2!!")

        quat_system = quat2 * UnitQuaternion(offset)
        quat_system = self.quat_product(quat, offset)

        # ---- checking the norms ---#
        self.check_norm_quaternion(quat_list = [quat_system, quat2, offset])
        return quat_system #.A

    def system_quat_to_NN(self, quat, offset):
        offset_inverse = UnitQuaternion(offset).inv()
        if torch.is_tensor(quat):
            quat2 = self.gpu_to_cpu(quat) #.cpu().numpy()
            # quat_NN = UnitQuaternion(quat2) / UnitQuaternion(offset)
            quat_NN = self.quat_product(quat2, offset_inverse.A) #UnitQuaternion(quat2) * offset_inverse
        else:
            # # quat_np = quat.transpose()[0] #todo: check if necessary!
            # quat2 = UnitQuaternion(s=quat[0], v=quat[1:], norm=True)
            # if self.quaternion_flipped(quat=quat2.A, quat_prev=quat): #todo, add!!
            #     # print("flipped!!")
            #     quat2 = quat2 * -1
            # quat_NN = quat2 * offset_inverse
            quat_NN = self.quat_product(quat, offset_inverse.A)

        # ---- checking the norms ---#
        self.check_norm_quaternion(quat_list = [quat_NN, offset])
        return quat_NN #torch.FloatTensor(quat_NN).cuda() #torch.FloatTensor(quat_NN.A).cuda()

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

    # -------------------- High level functions -----------------------------#
    def translation_goal(self, state_goal, goal_NN):
        # Translation of goal:
        goal_normalized = self.call_normalize_state(state=state_goal)
        translation = self.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN)
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
        x_t[0][-4:] = self.quat_product(x_t[0][-4:], offset_inverse.A)
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

    def check_quaternion_flipped(self, quat, quat_prev):
        dist_quat = np.linalg.norm(quat - quat_prev)
        if dist_quat > 1.0:
            quat_new = -1*copy.deepcopy(quat)
        else:
            quat_new = copy.deepcopy(quat)
        return quat_new

    def quaternion_flipped(self, quat, quat_prev):
        dist_quat = np.linalg.norm(quat - quat_prev)
        if dist_quat > 1.0:
            return True
        else:
            return False

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

