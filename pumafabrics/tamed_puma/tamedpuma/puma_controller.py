import numpy as np
import importlib
from pumafabrics.tamed_puma.nullspace_control.nullspace_controller import CartesianImpedanceController
from pumafabrics.tamed_puma.utils.normalizations_2 import normalization_functions
from pumafabrics.puma_extension.initializer import initialize_framework

class PUMAControl():
    def __init__(self, params, kinematics, NULLSPACE=True):
        self.update_params(params)
        self.kuka_kinematics = kinematics
        self.time_list = []
        self.NULLSPACE = NULLSPACE
        if self.NULLSPACE:
            self.controller_nullspace = CartesianImpedanceController(robot_name=self.params["robot_name"])
        self.Jac_dot_prev = np.zeros((self.params["dim_task"], self.params["dof"]))
        self.Jac_prev = np.zeros((self.params["dim_task"], self.params["dof"]))

    def update_params(self, params):
        self.params = params

    def vel_NN_rescale(self, transition_info, offset_orientation, xee_orientation, normalizations, kuka_kinematics):
        action_t_gpu = transition_info["desired velocity"]
        action_PUMA = normalizations.reverse_transformation(action_gpu=action_t_gpu, mode_NN="1st") #because we use velocity action!
        action_quat_vel = action_PUMA[3:]
        action_quat_vel_sys = kuka_kinematics.quaternion_operations.quat_vel_with_offset(quat_vel_NN=action_quat_vel,
                                                                   quat_offset=offset_orientation)
        xdot_pos_quat = np.append(action_PUMA[:3], action_quat_vel_sys)

        # --- if necessary, also get rpy velocities corresponding to quat vel ---#
        vel_rpy = kuka_kinematics.quaternion_operations.quat_vel_to_angular_vel(angle_quaternion=xee_orientation,
                                                            vel_quaternion=xdot_pos_quat[3:7]) / self.params["dt"]  # action_quat_vel
        return xdot_pos_quat, vel_rpy

    def acc_NN_rescale(self, transition_info, offset_orientation, xee_orientation, normalizations, kuka_kinematics):
        action_t_gpu = transition_info["desired acceleration"]
        action_PUMA = normalizations.reverse_transformation(action_gpu=action_t_gpu, mode_NN="2nd") #because we use velocity action!
        action_quat_acc = action_PUMA[3:]
        action_quat_acc_sys = kuka_kinematics.quaternion_operations.quat_vel_with_offset(quat_vel_NN=action_quat_acc,
                                                                   quat_offset=offset_orientation)
        xddot_pos_quat = np.append(action_PUMA[:3], action_quat_acc_sys)
        return xddot_pos_quat

    def initialize_PUMA(self, q_init, goal_pos, offset_orientation, results_base_directory=None):
        # # Construct classes:
        if results_base_directory is None:
            results_base_directory = '../pumafabrics/puma_extension/'

        # Parameters
        if self.params["mode_NN"] == "1st":
            self.params_name = self.params["params_name_1st"]
        else:
            self.params_name = self.params["params_name_2nd"]
        self.mode_NN_int = int(self.params["mode_NN"][0])
        print("self.params_name:", self.params_name)

        # Load parameters
        Params = getattr(importlib.import_module('pumafabrics.puma_extension.params.' + self.params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, self.params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Normalization class
        self.normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"], learner=learner)

        # Translation of goal:
        if self.params["dim_task"] == 7:
            state_goal = np.append(goal_pos, offset_orientation)
        else:
            state_goal = goal_pos
        translation_gpu, translation_cpu = self.normalizations.translation_goal(state_goal = state_goal, goal_NN=goal_NN)

        # initial state:
        x_t_init = self.kuka_kinematics.get_initial_state_task(q_init=q_init, qdot_init=np.zeros((self.params["dof"], 1)), offset_orientation=offset_orientation, mode_NN=self.params["mode_NN"])
        x_init_gpu = self.normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=translation_cpu, offset_orientation=offset_orientation)
        self.dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu[:, :self.params["dim_task"]*self.mode_NN_int].clone(), delta_t=1)
        return x_t_init, x_init_gpu, translation_cpu, goal_NN

    def return_classes(self):
        return self.dynamical_system, self.normalizations

    def request_PUMA(self, q, qdot, x_t, xee_orientation, offset_orientation, translation_cpu, POS_OUTPUT=False):
        # normalization
        x_t_gpu = self.normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=translation_cpu,
                                                       offset_orientation=offset_orientation)

        # compute action network
        transition_info = self.dynamical_system.transition(space='task', x_t=x_t_gpu[:, :self.params["dim_task"]*self.mode_NN_int].clone())

        # # -- normalize pose --#
        if self.params["dim_task"] == 7:
            x_t_action = self.normalizations.reverse_transformation_pos_quat(state_gpu=transition_info["desired state"], offset_orientation=offset_orientation)
        else:
            x_t_action = self.normalizations.reverse_transformation_position(position_gpu=transition_info["desired state"])  # , offset_orientation=offset_orientation)
        if POS_OUTPUT:
            return x_t_action, transition_info

        # --- rescale velocities (correct offset and normalization) ---#
        xdot_pos_quat, euler_vel = self.vel_NN_rescale(transition_info, offset_orientation,
                                                                       xee_orientation, self.normalizations,
                                                                       self.kuka_kinematics)
        xddot_pos_quat = self.acc_NN_rescale(transition_info, offset_orientation, xee_orientation,
                                                             self.normalizations, self.kuka_kinematics)

        # ---- velocity action_PUMA: option 1 ---- #
        qdot_PUMA_pulled = self.kuka_kinematics.inverse_diff_kinematics_quat(xdot=xdot_pos_quat,
                                                                                 angle_quaternion=xee_orientation).numpy()[0]

        #### --------------- directly from acceleration!! -----#
        qddot_PUMA, self.Jac_prev, Jac_dot_prev = self.kuka_kinematics.inverse_2nd_kinematics_quat(q=q,
                                                                                                  qdot=qdot_PUMA_pulled,
                                                                                                  xddot=xddot_pos_quat,
                                                                                                  angle_quaternion=xee_orientation,
                                                                                                  Jac_prev=self.Jac_prev)
        qddot_PUMA = qddot_PUMA.numpy()[0]
        if self.NULLSPACE:
            action_nullspace = self.controller_nullspace._nullspace_control(q=q, qdot=qdot)
            qddot_PUMA = qddot_PUMA + action_nullspace
        return qddot_PUMA, transition_info