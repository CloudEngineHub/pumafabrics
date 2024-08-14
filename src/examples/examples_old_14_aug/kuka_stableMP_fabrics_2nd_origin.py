import os
import gymnasium as gym
import yaml
import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from functions_stableMP_fabrics.filters import PDController
from mpscenes.goals.goal_composition import GoalComposition
from src.functions_stableMP_fabrics.plotting_functions2 import plotting_functions2
from functions_stableMP_fabrics.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from agent.utils.normalizations_2 import normalization_functions
from functions_stableMP_fabrics.environments import trial_environments
from functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka
from functions_stableMP_fabrics.energy_regulator import energy_regulation
from functions_stableMP_fabrics.cartesian_impedance_control import CartesianImpedanceController
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
import copy
import torch

class example_kuka_stableMP_R3S3():
    def __init__(self, bool_energy_regulator=False, bool_combined=True, robot_name="iiwa14"):
        self.bool_energy_regulator = bool_energy_regulator
        self.bool_combined = bool_combined
        self.GOAL_REACHED = False
        self.robot_name=robot_name

    def check_goal_reached(self, x_ee, x_goal):
        dist = np.linalg.norm(x_ee - x_goal)
        if dist<0.02:
            self.GOAL_REACHED = True
            return True
        else:
            return False
    def set_planner(self, goal: GoalComposition, degrees_of_freedom: int = 7, mode="acc", dt=0.01, bool_speed_control=True):
        """
        Initializes the fabric planner for the panda robot.
        """
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            root_link="iiwa_link_0",
            end_links=["iiwa_link_7"],
        )
        planner = ParameterizedFabricPlannerExtended(
            degrees_of_freedom,
            forward_kinematics,
            time_step=dt,
        )
        collision_links = ["iiwa_link_3", "iiwa_link_4", "iiwa_link_5", "iiwa_link_6", "iiwa_link_7"]
        iiwa_limits = [
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-3.05432619, 3.05432619],
        ]
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            collision_links=collision_links,
            goal=goal,
            number_obstacles=2,
            number_plane_constraints=0,
            limits=iiwa_limits,
        )
        planner.concretize_extensive(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
        # planner.concretize(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
        return planner, forward_kinematics

    def compute_action_fabrics(self, q, ob_robot):
        arguments_dict = dict(
            q=q,
            qdot=ob_robot["joint_state"]["velocity"],
            x_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size'],
            radius_body_links=self.collision_radii,
            constraint_0=np.array([0, 0, 1, 0.0]))

        M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = self.planner_avoidance.compute_M_f_action_avoidance(
            **arguments_dict)
        # print("action_avoidance:", action_avoidance)
        # print("speed avoidance:", xddot_speed_avoidance)
        qddot_speed = np.zeros((dof,))  # todo: think about what to do with speed regulation term!!
        # diff = action_avoidance-np.dot(self.planner_avoidance.Minv(M_avoidance), f_avoidance)
        # print("difference, ", diff)
        return action_avoidance, M_avoidance, f_avoidance, qddot_speed

    def combine_action(self, M_avoidance, M_attractor, f_avoidance, f_attractor, qddot_speed, qdot = []):
        #xddot_combined = -0.5*np.dot(self.planner_avoidance.Minv(M_avoidance), f_avoidance) - 0.5*np.dot(self.planner_avoidance.Minv(M_attractor), f_attractor) # + qddot_speed
        xddot_combined = -np.dot(self.planner_avoidance.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + qddot_speed
        #print("diff:", np.linalg.norm(xddot_combined - xddot_combined2))
        if self.planner_avoidance._mode == "vel":
            action_combined = qdot + self.planner_avoidance._time_step * xddot_combined
        else:
            action_combined = xddot_combined
        return action_combined

    def integrate_to_vel(self, qdot, action_acc, dt):
        qdot_action = action_acc *dt +qdot
        return qdot_action

    def vel_NN_rescale(self, transition_info, offset_orientation, xee_orientation, normalizations, kuka_kinematics):
        action_t_gpu = transition_info["desired velocity"]
        action_stableMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, mode_NN="1st") #because we use velocity action!
        action_quat_vel = action_stableMP[3:]
        action_quat_vel_sys = kuka_kinematics.quat_vel_with_offset(quat_vel_NN=action_quat_vel,
                                                                   quat_offset=offset_orientation)
        xdot_pos_quat = np.append(action_stableMP[:3], action_quat_vel_sys)

        # --- if necessary, also get rpy velocities corresponding to quat vel ---#
        vel_rpy = kuka_kinematics.quat_vel_to_angular_vel(angle_quaternion=xee_orientation,
                                                            vel_quaternion=xdot_pos_quat[3:7]) / dt  # action_quat_vel
        return xdot_pos_quat, vel_rpy

    def acc_NN_rescale(self, transition_info, offset_orientation, xee_orientation, normalizations, kuka_kinematics):
        action_t_gpu = transition_info["desired acceleration"]
        action_stableMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, mode_NN="2nd") #because we use velocity action!
        action_quat_acc = action_stableMP[3:]
        action_quat_acc_sys = kuka_kinematics.quat_vel_with_offset(quat_vel_NN=action_quat_acc,
                                                                   quat_offset=offset_orientation)
        xddot_pos_quat = np.append(action_stableMP[:3], action_quat_acc_sys)
        return xddot_pos_quat

    def relationship_dq_dx(self, offset_orientation, translation_cpu, kuka_kinematics, normalizations, fk):
        qq = ca.SX.sym("q", 7, 1)
        x_pose = kuka_kinematics.forward_kinematics_symbolic(end_link_name="iiwa_link_7", fk=fk)
        x_NN = normalizations.normalize_pose_to_NN([x_pose], translation_cpu, offset_orientation)
        dxdq = ca.jacobian(x_NN[0], qq)
        self.dxdq_fun = ca.Function("q_to_x", [qq], [dxdq], ["q"], ["dxdq"])
        return self.dxdq_fun

    def run_kuka_example(self, n_steps=2000, env=None, goal_pos=[-0.24355761, -0.75252747, 0.5], mode="acc", mode_NN = "1st", dt=0.01, mode_env=None):
        # --- parameters --- #
        dof = 7
        dim_pos = 3
        dim_task = 7
        vel_limits = np.array([86, 85, 100, 75, 130, 135, 135])*np.pi/180
        orientation_goal = np.array([ 0.61566569, -0.37995015,  0.67837375, -0.12807299]) # start orientation: np.array([0.49533057,  0.3304898 , 0.5092156 , -0.62138844])
        offset_orientation = copy.deepcopy(orientation_goal)
        self.collision_radii = {3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
        Jac_prev = np.zeros((7, 7))

        action = np.zeros(dof)
        ob, *_ = env.step(action)

        # Construct classes:
        results_base_directory = './'
        kuka_kinematics = KinematicsKuka(dt=dt, end_link_name="iiwa_link_7", robot_name="iiwa14")
        pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=dt)
        if mode_NN == "2nd":
            ee_translational_stiffness = 6
            ee_rotational_stiffness = 2
            cartesian_controller = CartesianImpedanceController([], ee_translational_stiffness, ee_rotational_stiffness)
        else:
            cartesian_controller = CartesianImpedanceController()

        # fabrics planner:
        self.planner_avoidance, fk = self.set_planner(goal=None, bool_speed_control=False, mode=mode, dt=dt)

        # Parameters
        if mode_NN == "1st":
            self.params_name = '1st_order_R3S3_converge'
        else:
            self.params_name = '2nd_order_R3S3_saray'
        print("self.params_name:", self.params_name)
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
        normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=dim_task, dt=dt, mode_NN=mode_NN, learner=learner)

        # Translation of goal:
        translation_gpu, translation_cpu = normalizations.translation_goal(state_goal = np.append(goal_pos, orientation_goal), goal_NN=goal_NN)

        # initial state:
        x_t_init = kuka_kinematics.get_initial_state_task(q_init=q_init, qdot_init=np.zeros((dof, 1)), offset_orientation=offset_orientation, params_name=self.params_name)
        x_init_gpu = normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=translation_cpu, offset_orientation=offset_orientation)
        dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

        #create function dxdq:
        energy_regulation_class = energy_regulation(dim_task=7, mode_NN=mode_NN, dof=dof, dynamical_system=dynamical_system)
        energy_regulation_class.relationship_dq_dx(offset_orientation, translation_cpu, kuka_kinematics, normalizations, fk)

        # Initialize lists
        q_list = np.zeros((dof, n_steps))
        qdot_list = np.zeros((dof, n_steps))
        action_list = np.zeros((dof, n_steps))
        quat_vel_list = np.zeros((4, n_steps))
        angular_vel_list = np.zeros((3, n_steps))
        x_list = np.zeros((dim_pos, n_steps))
        quat_list = np.zeros((4, n_steps))
        quat_prev = copy.deepcopy(x_t_init[3:7])

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            q_list[:, w] = q
            qdot_list[:, w] = qdot

            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = kuka_kinematics.get_state_task(q, quat_prev, params_name=self.params_name, qdot=qdot)
            quat_prev = copy.deepcopy(xee_orientation)
            vel_ee, Jac_current = kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
            x_t_gpu = normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=translation_cpu, offset_orientation=offset_orientation)

            # --- action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)

            # # -- transform to configuration space --#
            # --- rescale velocities and pose (correct offset and normalization) ---#
            xdot_pos_quat, euler_vel = self.vel_NN_rescale(transition_info, offset_orientation, xee_orientation, normalizations, kuka_kinematics)
            xddot_pos_quat = self.acc_NN_rescale(transition_info, offset_orientation, xee_orientation, normalizations, kuka_kinematics)
            x_t_action = normalizations.reverse_transformation_pos_quat(state_gpu=transition_info["desired state"], offset_orientation=offset_orientation)

            # ---- velocity action_stableMP: option 1 ---- #
            qdot_stableMP_pulled = kuka_kinematics.inverse_diff_kinematics_quat(xdot=xdot_pos_quat,
                                                                                    angle_quaternion=xee_orientation).numpy()[0]
            # # ---- get a decent acceleration based on the velocity command ---#
            # qddot_stableMP = pdcontroller.control(desired_velocity=qdot_stableMP_pulled, current_velocity=qdot)
            # # Jac_quat = np.zeros((7,7))

            #### --------------- directly from acceleration!! -----#
            qddot_stableMP, Jac_quat = kuka_kinematics.inverse_2nd_kinematics_quat(q=q, qdot=qdot_stableMP_pulled, xddot=xddot_pos_quat, angle_quaternion=xee_orientation, Jac_prev=Jac_prev)
            qddot_stableMP = qddot_stableMP.numpy()[0]

            if self.bool_combined == True:
                # ----- Fabrics action ----#
                action_avoidance, M_avoidance, f_avoidance, qddot_speed = self.compute_action_fabrics(q=q, ob_robot=ob_robot)

                if self.bool_energy_regulator == True:
                    # ---- get action by NN via theorem III.5 in https://arxiv.org/pdf/2309.07368.pdf ---#
                    action_combined = energy_regulation_class.compute_action_theorem_III5(q=q, qdot=qdot,
                                                                                          qddot_attractor = qddot_stableMP,
                                                                                          action_avoidance=action_avoidance,
                                                                                          M_avoidance=M_avoidance,
                                                                                          transition_info=transition_info)
                else:
                    # --- get action by a simpler combination, sum of dissipative systems ---#
                    action_combined = 0.3*qddot_stableMP + action_avoidance
                    # action_combined = self.combine_action(M_avoidance=M_avoidance,
                    #                                             M_attractor=M_stableMP,
                    #                                             f_avoidance=f_avoidance,
                    #                                             f_attractor=-weight_stableMP*qddot_stableMP[0:dof],
                    #                                             qddot_speed=qddot_speed,
                    #                                             qdot=ob_robot["joint_state"]["velocity"][0:dof])
            else: #otherwise only apply action by stable MP
                action_combined = qddot_stableMP

            if mode_env is not None:
                if mode_env == "vel": # and bool_combined == True:# todo: fix nicely or mode == "acc"): #mode_NN=="2nd":
                    action = self.integrate_to_vel(qdot=qdot, action_acc=action_combined, dt=dt)
                else:
                    action = action_combined
            else:
                action = action_combined
            self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=goal_pos)
            if self.GOAL_REACHED == True and mode_env=="vel": #todo: fix this!!
                action = np.zeros(dof)
            ob, *_ = env.step(action)

            print("quat desired: ", orientation_goal)
            print("quat: ", xee_orientation)

            # save for plotting:
            x_list[:, w] = x_t[0][:3]
            quat_vel_list[:, w] = xdot_pos_quat[3:7].transpose()
            angular_vel_list[:, w] = euler_vel.transpose()
            action_list[:, w] = np.clip(qdot_stableMP_pulled, -vel_limits, vel_limits)
            quat_list[:, w] = xee_orientation
            Jac_prev = Jac_quat

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
    robot_name = "iiwa14"
    dof = 7
    bool_energy_regulator = False
    bool_combined = False
    mode_NN = "2nd"     # mode of the NN (1st or 2nd)
    mode = "acc"        # mode of fabrics (vel or acc)
    mode_env = "vel"    # mode fed to the environment (vel or acc)
    dt = 0.00769
    nr_obst = 0
    init_pos = np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010))
    goal_pos = [0.60829608, 0.04368581, 0.252421]

    # --- generate environment --- #
    envir_trial = trial_environments()
    params = {"render": render, "mode_env": mode_env, "dt":dt, "init_pos": init_pos, "goal_pos": goal_pos, "nr_obst": nr_obst, "robot_name": robot_name,
              "end_links": ["iiwa_link_7"], "nr_obst_dyn": nr_obst, "positions_obstacles": [[0.5, -0.25, 0.5], [0.24355761, 0.45252747, 0.2]],
              "collision_links": []}
    (env, goal) = envir_trial.initialize_environment_kuka(params)
    example_class = example_kuka_stableMP_R3S3(bool_energy_regulator=bool_energy_regulator, bool_combined=bool_combined, robot_name=robot_name)
    res = example_class.run_kuka_example(n_steps=3000, env=env, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN, mode_env=mode_env)