import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from functions_stableMP_fabrics.filters import ema_filter_deriv, PDController
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from src.functions_stableMP_fabrics.geometry_IL import construct_IL_geometry
from src.functions_stableMP_fabrics.plotting_functions import plotting_functions
from src.functions_stableMP_fabrics.plotting_functions2 import plotting_functions2
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from functions_stableMP_fabrics.cartesian_impedance_control import CartesianImpedanceController
from agent.utils.normalizations_2 import normalization_functions
from functions_stableMP_fabrics.environments import trial_environments
from functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka
from tools.animation import TrajectoryPlotter
import pybullet as p
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
import copy
from spatialmath import SO3, UnitQuaternion
from functions_stableMP_fabrics.drake_ik.ik_drake import iiwa_example_drake

class example_kuka_stableMP_R3S3():
    def __init__(self):
        dt = 0.01
        self.GOAL_REACHED = False

    def check_goal_reached(self, x_ee, x_goal):
        dist = np.linalg.norm(x_ee - x_goal)
        if dist<0.02:
            self.GOAL_REACHED = True
            return True
        else:
            return False

    def run_kuka_example(self, n_steps=2000, env=None, goal_pos=[-0.24355761, -0.75252747, 0.5], mode="acc", mode_NN = "1st", dt=0.01):
        # --- parameters --- #
        dof = 7
        dim_pos = 3
        dim_task = 7
        vel_limits = np.array([86, 85, 100, 75, 130, 135, 135])*np.pi/180
        orientation_goal = np.array([ 0.61566569, -0.37995015,  0.67837375, -0.12807299])
        #np.array([0.64611803, 0.20264371, 0.40471341, -0.61455193])  #np.array([1., 0., 0., 0]) # start orientation: np.array([0.49533057,  0.3304898 , 0.5092156 , -0.62138844])
        offset_orientation = copy.deepcopy(orientation_goal)

        action = np.zeros(dof)
        ob, *_ = env.step(action)

        # Construct classes:
        results_base_directory = './'
        kuka_kinematics = KinematicsKuka(dt=dt)
        pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=dt)
        cartesian_controller = CartesianImpedanceController(robot=[])
        drake_class = iiwa_example_drake()

        # Parameters
        if mode_NN == "1st":
            self.params_name = '1st_order_R3S3_converge'
        else:
            print("not implemented!!")
            self.params_name = '2nd_order_R3S3_saray'
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]

        # Load parameters
        Params = getattr(importlib.import_module('params.' + self.params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, self.params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Normalization class
        normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=dim_task, dt=dt, mode_NN=mode_NN)

        # Translation of goal:
        translation_gpu, translation_cpu = normalizations.translation_goal(state_goal = np.append(goal_pos, orientation_goal), goal_NN=goal_NN)

        # initial state:
        x_t_init = kuka_kinematics.get_initial_state_task(q_init=q_init, offset_orientation=offset_orientation)
        x_init_gpu = normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=translation_cpu, offset_orientation=offset_orientation)
        dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

        # Initialize lists
        list_diff = []
        list_fabr_goal = []
        list_fabr_avoidance = []
        list_safeMP = []
        q_list = np.zeros((dof, n_steps))
        qdot_list = np.zeros((dof, n_steps))
        action_list = np.zeros((dof, n_steps))
        quat_vel_list = np.zeros((4, n_steps))
        angular_vel_list = np.zeros((3, n_steps))
        x_list = np.zeros((dim_pos, n_steps))
        quat_list = np.zeros((4, n_steps))
        quat_prev = copy.deepcopy(x_t_init[3:])
        x_t = copy.deepcopy([x_t_init])

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            q_list[:, w] = q
            qdot_list[:, w] = qdot

            # --- end-effector states and normalized states --- #
            x_t_prev = copy.deepcopy(x_t)
            x_t, xee_orientation, _ = kuka_kinematics.get_state_task(q, quat_prev)
            #print("xee_orientation:", xee_orientation)
            quat_prev = copy.deepcopy(xee_orientation)
            vel_ee, Jac_current = kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
            x_t_gpu = normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=translation_cpu, offset_orientation=offset_orientation)

            # --- action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]
            if mode_NN == "2nd":
                print("not implemented!!")
            else:
                action_t_gpu = transition_info["desired velocity"]
                action_cpu = action_t_gpu.T.cpu().detach().numpy()
            x_t_action = normalizations.reverse_transformation_pos_quat(state_gpu=x_t_NN, offset_orientation=offset_orientation)
            # action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu)

            ## Use drake for IK to configuration space (with obstacle avoidance)
            # RPY_angles = [-1.82, 0, 1.58]
            # rot_matrixx = kuka_kinematics.rpy_to_rot_matrix(RPY_angles)
            # print("rot_matrix:", rot_matrixx)
            # desired_rot_matrix = kuka_kinematics.quat_to_rot_matrix(x_t_action[3:7])
            desired_rot_matrix = kuka_kinematics.quat_to_rot_matrix(orientation_goal)
            q_d = drake_class.main(translation=x_t_action[0:3], rot_matrix=desired_rot_matrix, q0=q)
            # q_d = drake_class.main(translation=goal_pos, rot_matrix=desired_rot_matrix, q0=q)
            print("q_d:", q_d)
            print("q:", q)
            action = 0.1*pdcontroller.control(desired_velocity=q_d, current_velocity=q)
            # # # -- transform to configuration space --#
            # if mode_NN == "1st":
            #     # ---- option 1 -----#
            #     action_quat_vel = action_safeMP[3:]
            #     action_quat_vel_sys = kuka_kinematics.quat_vel_with_offset(quat_vel_NN=action_quat_vel,
            #                                                                quat_offset=offset_orientation)
            #     action_safeMP_pulled = kuka_kinematics.inverse_diff_kinematics_quat(xdot=np.append(action_safeMP[:3], action_quat_vel_sys), angle_quaternion=xee_orientation).numpy()[0]
            #
            #     # ---- option 2: with PD controller ------ #
            #     euler_vel = kuka_kinematics.quat_vel_to_angular_vel(angle_quaternion=xee_orientation,
            #                                                               vel_quaternion=action_quat_vel_sys)
            #     """
            #     ee_velocity_NN = np.append(action_safeMP[0:3], euler_vel)
            #     ee_vel_quat_d = kuka_kinematics.angular_vel_to_quat_vel(angle_quaternion=xee_orientation, vel_angular=vel_ee[3:])
            #     action_safeMP_pulled, action_quat_vel, euler_vel = cartesian_controller.control_law(position_d=x_t_action[:3], #state_goal,
            #                                                                                         orientation_d=x_t_action[3:], #orientation_goal,
            #                                                                                         ee_pose=x_t[0],
            #                                                                                         ee_velocity=vel_ee,
            #                                                                                         ee_velocity_d=ee_velocity_NN,
            #                                                                                         ee_vel_quat = action_quat_vel_sys,
            #                                                                                         ee_vel_quat_d = ee_vel_quat_d,
            #                                                                                         J=Jac_current,
            #                                                                                         dt=dt)
            #     """
            # else:
            #     print("not implemented!!")
            #
            # if mode == "acc" and mode_NN == "1st":
            #     # ---- get a decent acceleration based on the velocity command ---#
            #     action = pdcontroller.control(desired_velocity=action_safeMP_pulled, current_velocity=qdot)
            # else:
            #     action = action_safeMP_pulled

            self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=goal_pos)
            # if self.GOAL_REACHED == True:
            #     action = np.zeros(dof)
            print("action:", action)
            ob, *_ = env.step(action)

            # save for plotting:
            x_list[:, w] = x_t[0][:3]
            # action_quat_vel = np.zeros((7,))
            # quat_vel_list[:, w] = action_quat_vel.transpose()
            # angular_vel_list[:, w] = euler_vel.transpose()
            # action_list[:, w] = np.clip(action_safeMP_pulled, -vel_limits, vel_limits)
            quat_list[:, w] = xee_orientation

        plt.savefig(params.results_path+"images/kuka_robot_plot_R3S3")
        plotting_class = plotting_functions2(results_path=params.results_path)
        plotting_class.velocities_over_time(quat_vel_list=quat_vel_list,
                                            ang_vel_list=angular_vel_list,
                                            joint_vel_list=qdot_list,
                                            action_list=action_list,
                                            dt=dt)
        plotting_class.pose_over_time(quat_list=quat_list)
        env.close()
        return {}


if __name__ == "__main__":
    render = True
    dof = 7
    mode = "vel"
    mode_NN = "1st"
    dt = 0.01
    nr_obst = 0
    init_pos = np.array([0.9, 0.79, -0.22, -1.33, 1.20, -1.76, -1.06])
    #np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)) #np.zeros((dof,))
    goal_pos = [0.60829608, 0.04368581, 0.452421  ] #[-0.24355761, -0.55252747, 0.5] #[-0.07311596, -0.6, 0.4]

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initialize_environment_kuka(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos, nr_obst=nr_obst)
    example_class = example_kuka_stableMP_R3S3()
    res = example_class.run_kuka_example(n_steps=1000, env=env, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN)
