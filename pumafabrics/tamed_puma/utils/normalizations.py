import numpy as np
import copy
import torch
from src.agent.utils.dynamical_system_operations import normalize_state, denormalize_state
from spatialmath import UnitQuaternion
from spatialmath import SO3, UnitQuaternion

class normalizaton_sim_NN():

    def __init__(self, scaling_room, dof_task=0):
        self.scaling_room = scaling_room
        try:
            self.min_state = np.array([scaling_room["x"][0], scaling_room["y"][0], scaling_room["z"][0]])
            self.max_state = np.array([scaling_room["x"][1], scaling_room["y"][1], scaling_room["z"][1]])
        except:
            self.min_state = np.array([scaling_room["x"][0], scaling_room["y"][0]])
            self.max_state = np.array([scaling_room["x"][1], scaling_room["y"][1]])
        if dof_task == 0:
            self.dof_task = len(self.min_state)
            self.dim_pos = self.dof_task
        else:
            self.dof_task = dof_task
            self.dim_pos = 3
        self.scaling_factor = self.get_scaling_factor()

    def get_scaling_factor(self):
        self.scaling_factor = self.max_state  - self.min_state
        return self.scaling_factor

    def call_normalize_state(self, state):
        if len(state) == 3 or len(state) == 2:
            state_normalized = normalize_state(state, x_min=self.min_state, x_max=self.max_state)
        elif len(state) == 7:
            pos_normalized = normalize_state(state[0:self.dim_pos], x_min=self.min_state, x_max=self.max_state)
            state_normalized = np.append(pos_normalized, state[self.dim_pos:])
        else:
            state_normalized = state
            print("length for normalization not known!!")
        return state_normalized

    def denormalize_action(self, action_value):
        action_scaled = copy.deepcopy(action_value)
        action_scaled[0] = action_value[0]*self.scaling_factor[0]
        action_scaled[1] = action_value[1]*self.scaling_factor[1]
        #todo: why not action_scaled[2]???
        return action_scaled

    def normalize_to_NN(self, state_value, x_min, x_max):
        state_scaled = copy.deepcopy(state_value)
        state_scaled[0:self.dim_pos] = normalize_state(state_value[0:self.dim_pos], x_min=x_min, x_max=x_max)
        if len(state_value)==self.dof_task*2: #(pos + vel)
            state_scaled[self.dof_task] = state_value[self.dof_task] / self.scaling_factor[0]
            state_scaled[self.dof_task+1] = state_value[self.dof_task+1] / self.scaling_factor[1]
        return state_scaled

    def get_translation(self, goal_pos, goal_pos_NN):
        translation = goal_pos - np.array(goal_pos_NN)
        self.translation = translation
        return translation

    def get_offset_quaternion(self, goal_quat_desired, goal_NN_quat):
        offset = UnitQuaternion(goal_quat_desired) / UnitQuaternion(goal_NN_quat)
        len_quat = np.linalg.norm(offset)
        if np.linalg.norm(len_quat-1.)>=1e-6 :
            print("len  quaternion is not equal to zero!!")
        return offset.A

    def NN_quat_to_system(self, quat, offset):
        quat_system = UnitQuaternion(quat) * UnitQuaternion(offset)
        len_quat = np.linalg.norm(quat_system)
        if np.linalg.norm(len_quat-1.)>=1e-6 :
            print("len  quaternion is not equal to zero!!")
        return quat_system

    def system_quat_to_NN(self, quat, offset):
        quat_NN = UnitQuaternion(quat.cpu().numpy()) / UnitQuaternion(offset)
        len_quat = np.linalg.norm(quat_NN)
        if np.linalg.norm(len_quat-1.)>=1e-6 :
            print("len  quaternion is not equal to zero!!")
        return torch.FloatTensor(quat_NN.A).cuda()

    def transformation_to_NN(self, x_t, translation_gpu, dt=0.01, min_vel=[], max_vel=[]):
        if self.dof_task == len(x_t): #todo: fix!!!
            x_t = [x_t.transpose()]

        x_t_NN = copy.deepcopy(x_t)

        #scale wrt room size:
        x_t_NN[0] = self.normalize_to_NN(x_t[0], x_min=self.min_state, x_max=self.max_state)

        #scale wrt velocities
        x_t_gpu = torch.FloatTensor(x_t_NN).cuda()
        if len(x_t[0]) == self.dof_task * 2:
            x_t_gpu[0][self.dof_task:self.dof_task*2] = normalize_state(x_t_gpu[0][self.dof_task:self.dof_task*2] * dt, min_vel, max_vel)

        # translate:
        # if self.dof_task == 7: #todo: replace!!
        #     x_t_gpu[0][0:3] = self.translation_to_NN(x_t_gpu[0][0:3],
        #                                                          translation=translation_gpu[0:3])
        #     x_t_gpu[0][3:7] = self.translation_quaternions_to_NN(x_t_gpu[0][3:7], offset_quat = offset_quat)
        # else:
        x_t_gpu[0][0:self.dof_task] = self.translation_to_NN(x_t_gpu[0][0:self.dof_task],
                                                                 translation=translation_gpu)
        x_t_cpu = x_t_gpu.T.cpu().detach().numpy()
        return x_t_gpu, x_t_cpu

    def reverse_transformation(self, action_gpu, dt=0.01, mode_NN="1st"):
        action_cpu = np.zeros((self.dof_task, ))
        action_t = action_gpu.T.cpu().detach().numpy()
        action_cpu[0:self.dof_task] = action_t[0:self.dof_task].T

        # unscale action:
        if mode_NN == "2nd":
            action_Transformed = action_cpu / (dt*dt)
        elif mode_NN == "1st":
            action_Transformed = action_cpu / dt
        action_Transformed[0:self.dof_task] = self.denormalize_action(action_Transformed[0:self.dof_task])
        return action_Transformed

    def translation_to_NN(self, xy_sim, translation):
        xy_translated = xy_sim - translation
        return xy_translated

    def translation_quaternions_to_NN(self, x_quat, offset_quat = [1., 0., 0., 0.]):
        x_quat = UnitQuaternion(x_quat.cpu().detach().numpy())
        offset = UnitQuaternion(offset_quat)
        quat_diff = x_quat/offset
        return torch.FloatTensor(quat_diff.A).cuda()

    def reverse_transformation_pos(self, position_gpu, dt=0.01):
        position_cpu = position_gpu.T.cpu().detach().numpy()
        pos_detranslated = self.detranslation_to_NN(position_cpu)
        pos = denormalize_state(pos_detranslated, x_min=self.min_state, x_max=self.max_state)
        return pos

    def reverse_transformation_pos_quat(self, state_gpu, offset_orientation):
        state_cpu = state_gpu.T.cpu().detach().numpy()

        # position
        pose_detranslated = self.detranslation_to_NN(state_cpu.transpose())
        position_denormalized = denormalize_state(pose_detranslated[0][:3], x_min=self.min_state, x_max=self.max_state)

        #quaternion:
        quat = UnitQuaternion(state_cpu[3:].transpose()[0])
        offset = UnitQuaternion(offset_orientation)
        quat_denormalized = (quat * offset).A

        pose = np.append(position_denormalized, quat_denormalized)
        return pose

    def detranslation_to_NN(self, xy_NN):
        xy_detranslated = xy_NN + self.translation
        return xy_detranslated