import os
import gymnasium as gym
import yaml
import numpy as np
import casadi as ca
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from functions_stableMP_fabrics.filters import PDController
from mpscenes.goals.goal_composition import GoalComposition
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
import time
from functions_stableMP_fabrics.analysis_utils import UtilsAnalysis

class example_kuka_stableMP_fabrics():
    def __init__(self):
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = -1
        self.solver_times = []
        with open("../../config/kuka_stableMP_fabrics.yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.robot_name = self.params["robot_name"]

    def overwrite_defaults(self, render=None, init_pos=None, goal_pos=None, nr_obst=None, bool_energy_regulator=None, positions_obstacles=None, orientation_goal=None, params_name_1st=None):
        if render is not None:
            self.params["render"] = render
        if init_pos is not None:
            self.params["init_pos"] = init_pos
        if goal_pos is not None:
            self.params["goal_pos"] = goal_pos
        if orientation_goal is not None:
            self.params["orientation_goal"] = orientation_goal
        if nr_obst is not None:
            self.params["nr_obst"] = nr_obst
        if bool_energy_regulator is not None:
            self.params["bool_energy_regulator"] = bool_energy_regulator
        if positions_obstacles is not None:
            self.params["positions_obstacles"] = positions_obstacles
        if params_name_1st is not None:
            self.params["params_name_1st"] = params_name_1st

    def check_goal_reached(self, x_ee, x_goal):
        dist = np.linalg.norm(x_ee - x_goal)
        if dist<0.02:
            self.GOAL_REACHED = True
            return True
        else:
            return False

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_kuka(params=self.params)

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )

    def set_planner(self, goal: GoalComposition):
        """
        Initializes the fabric planner for the panda robot.
        """
        if goal is not None:
            goal  = self.goal
        self.construct_fk()
        planner = ParameterizedFabricPlannerExtended(
            self.params["dof"],
            self.forward_kinematics,
            time_step=self.params["dt"],
            collision_geometry = self.params["collision_geometry"],
            collision_finsler = self.params["collision_finsler"],
        )
        planner.set_components(
            collision_links=self.params["collision_links"],
            goal=goal,
            number_obstacles=self.params["nr_obst"],
            number_plane_constraints=0,
            limits=self.params["iiwa_limits"],
        )
        planner.concretize_extensive(mode=self.params["mode"], time_step=self.params["dt"], extensive_concretize=self.params["bool_extensive_concretize"], bool_speed_control=self.params["bool_speed_control"])
        return planner, self.forward_kinematics

    def compute_action_fabrics(self, q, ob_robot):
        nr_obst = self.params["nr_obst"]
        arguments_dict = dict(
            q=q,
            qdot=ob_robot["joint_state"]["velocity"],
            x_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size'],
            radius_body_links=self.params["collision_radii"],
            constraint_0=np.array([0, 0, 1, 0.0]))

        M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = self.planner_avoidance.compute_M_f_action_avoidance(
            **arguments_dict)
        qddot_speed = np.zeros((self.params["dof"],))  # todo: think about what to do with speed regulation term!!
        return action_avoidance, M_avoidance, f_avoidance, qddot_speed

    def combine_action(self, M_avoidance, M_attractor, f_avoidance, f_attractor, qddot_speed, qdot = []):
        xddot_combined = -np.dot(self.planner_avoidance.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + qddot_speed
        #xddot_combined = -np.dot(self.planner_avoidance.Minv(M_avoidance).full(), f_avoidance + f_attractor) + qddot_speed
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
                                                            vel_quaternion=xdot_pos_quat[3:7]) / self.params["dt"]  # action_quat_vel
        return xdot_pos_quat, vel_rpy

    def construct_example(self):
        self.initialize_environment()

        #construct classes:
        self.planner_avoidance, _ = self.set_planner(goal=None)
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics, collision_links=self.params["collision_links"], collision_radii=self.params["collision_radii"])
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0], robot_name=self.params["robot_name"])
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])

        # Parameters
        if self.params["mode_NN"] == "1st":
            self.params_name = self.params["params_name_1st"]
        else:
            self.params_name = self.params["params_name_2nd"]

        # Load parameters
        results_base_directory = './'
        Params = getattr(importlib.import_module('params.' + self.params_name), 'Params')
        self.params_NN = Params(results_base_directory)
        self.params_NN.results_path += self.params_NN.selected_primitives_ids + '/'
        self.params_NN.load_model = True

        # Initialize framework
        self.learner, _, data = initialize_framework(self.params_NN, self.params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Normalization class
        self.normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"], learner=self.learner)

        # Translation of goal:
        self.translation_gpu, self.translation_cpu = self.normalizations.translation_goal(state_goal = np.append(self.params["goal_pos"], self.params["orientation_goal"]), goal_NN=goal_NN)


    def run_kuka_example(self):
        # --- parameters --- #
        n_steps = self.params["n_steps"]
        offset_orientation = np.array(self.params["orientation_goal"])
        goal_pos = self.params["goal_pos"]
        dof = self.params["dof"]

        action = np.zeros(dof)
        ob, *_ = self.env.step(action)

        # initial state:
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        x_t_init = self.kuka_kinematics.get_initial_state_task(q_init=q_init, qdot_init=np.zeros((dof, 1)), offset_orientation=offset_orientation, mode_NN=self.params["mode_NN"])
        x_init_gpu = self.normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=self.translation_cpu, offset_orientation=offset_orientation)
        dynamical_system = self.learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

        #create function dxdq:
        energy_regulation_class = energy_regulation(dim_task=7, mode_NN=self.params["mode_NN"], dof=dof, dynamical_system=dynamical_system)
        energy_regulation_class.relationship_dq_dx(offset_orientation, self.translation_cpu, self.kuka_kinematics, self.normalizations, self.forward_kinematics)

        # Initialize lists
        xee_list = []
        vee_list = []
        quat_prev = copy.deepcopy(x_t_init[3:7])

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            if self.params["nr_obst"]>0:
                self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                self.obstacles = []


            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev, mode_NN=self.params["mode_NN"], qdot=qdot)
            quat_prev = copy.deepcopy(xee_orientation)
            vel_ee, Jac_current = self.kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
            x_t_gpu = self.normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=self.translation_cpu, offset_orientation=offset_orientation)

            # --- action by NN --- #
            time0 = time.perf_counter()
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)

            # # -- transform to configuration space --#
            # --- rescale velocities and pose (correct offset and normalization) ---#
            xdot_pos_quat, euler_vel = self.vel_NN_rescale(transition_info, offset_orientation, xee_orientation, self.normalizations, self.kuka_kinematics)
            x_t_action = self.normalizations.reverse_transformation_pos_quat(state_gpu=transition_info["desired state"], offset_orientation=offset_orientation)

            # ---- velocity action_stableMP: option 1 ---- #
            action_stableMP_pulled = self.kuka_kinematics.inverse_diff_kinematics_quat(xdot=xdot_pos_quat,
                                                                                    angle_quaternion=xee_orientation).numpy()[0]
            # ---- velocity action_stableMP: option 2 ---- #
            # action_stableMP_pulled = cartesian_controller.control_law(position_d=x_t_action[:3], #state_goal,
            #                                                         orientation_d=x_t_action[3:], #orientation_goal,
            #                                                         ee_pose=x_t[0],
            #                                                         ee_velocity=vel_ee,
            #                                                         ee_velocity_d=np.append(xdot_pos_quat[:3], euler_vel),
            #                                                         J=Jac_current)
            # action_stableMP_pulled = np.clip(action_stableMP_pulled, -vel_limits, vel_limits)

            # ---- get a decent acceleration based on the velocity command ---#
            qddot_stableMP = self.pdcontroller.control(desired_velocity=action_stableMP_pulled, current_velocity=qdot)

            if self.params["bool_combined"] == True:
                # ----- Fabrics action ----#
                action_avoidance, M_avoidance, f_avoidance, qddot_speed = self.compute_action_fabrics(q=q, ob_robot=ob_robot)

                if self.params["bool_energy_regulator"] == True:
                    # ---- get action by NN via theorem III.5 in https://arxiv.org/pdf/2309.07368.pdf ---#
                    action_combined = energy_regulation_class.compute_action_theorem_III5(q=q, qdot=qdot,
                                                                                          qddot_attractor = qddot_stableMP,
                                                                                          action_avoidance=action_avoidance,
                                                                                          M_avoidance=M_avoidance,
                                                                                          transition_info=transition_info)
                else:
                    # --- get action by a simpler combination, sum of dissipative systems ---#
                    weight_stableMP = 1.
                    M_stableMP = np.identity(dof, )
                    action_combined = self.combine_action(M_avoidance=M_avoidance,
                                                                M_attractor=M_stableMP,
                                                                f_avoidance=f_avoidance,
                                                                f_attractor=-weight_stableMP*qddot_stableMP[0:dof],
                                                                qddot_speed=qddot_speed,
                                                                qdot=ob_robot["joint_state"]["velocity"][0:dof])
            else: #otherwise only apply action by stable MP
                action_combined = action_stableMP_pulled

            if self.params["mode_env"] is not None:
                if self.params["mode_env"] == "vel" and self.params["bool_combined"] == True:# todo: fix nicely or mode == "acc"): #mode_NN=="2nd":
                    action = self.integrate_to_vel(qdot=qdot, action_acc=action_combined, dt=self.params["dt"])
                else:
                    action = action_combined
            else:
                action = action_combined
            self.solver_times.append(time.perf_counter() - time0)

            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
            vee_list.append(vel_ee)
            self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=self.obstacles)
            self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, quat_prev, x_goal=goal_pos)

            if self.GOAL_REACHED:
                self.time_to_goal = w*self.params["dt"]
                break

            if self.IN_COLLISION:
                self.time_to_goal = float("nan")
                break

        self.env.close()

        results = {
            "min_distance": self.utils_analysis.get_min_dist(),
            "collision": self.IN_COLLISION,
            "goal_reached": self.GOAL_REACHED,
            "time_to_goal": self.time_to_goal,
            "xee_list": xee_list,
            "vee_list": vee_list,
            "solver_times": self.solver_times,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        return results


if __name__ == "__main__":
    example_class = example_kuka_stableMP_fabrics()
    example_class.construct_example()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
