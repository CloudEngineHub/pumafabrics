import os
import time

import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from functions_stableMP_fabrics.filters import PDController
from agent.utils.normalizations_2 import normalization_functions
from functions_stableMP_fabrics.environments import trial_environments
from functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka
from functions_stableMP_fabrics.analysis_utils import UtilsAnalysis
import importlib
from initializer import initialize_framework
import copy
import yaml
import time
import torch
from scipy.spatial.transform import Rotation as R
import pytorch_kinematics as pk


class example_kuka_stableMP_R3S3():
    def __init__(self):
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = -1
        self.solver_times = []
        with open("config/kuka_stableMP.yaml", "r") as setup_stream:
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

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_kuka(params=self.params)

    def check_goal_reached(self, x_ee, x_goal):
        dist = np.linalg.norm(x_ee - x_goal)
        if dist<0.02:
            self.GOAL_REACHED = True
            return True
        else:
            return False

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )

    def construct_example(self):
        self.initialize_environment()

        # Construct classes:
        self.construct_fk()
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics, collision_links=self.params["collision_links"], collision_radii=self.params["collision_radii"])
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0], robot_name=self.params["robot_name"])
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])

        # Parameters
        if self.params["mode_NN"] == "1st":
            self.params_name = '1st_order_R3S3_converge'
        else:
            print("not implemented!!")
            self.params_name = '2nd_order_R3S3_saray'

        # Load parameters
        Params = getattr(importlib.import_module('params.' + self.params_name), 'Params')
        results_base_directory = './'
        self.params_NN = Params(results_base_directory)
        self.params_NN.results_path += self.params_NN.selected_primitives_ids + '/'
        self.params_NN.load_model = True

        # Initialize framework
        self.learner, _, data = initialize_framework(self.params_NN, self.params_name, verbose=False)
        self.goal_NN = data['goals training'][0]

        # Normalization class
        self.normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"])


    def run_kuka_example(self):
        # --- parameters --- #
        offset_orientation = np.array(self.params["orientation_goal"])
        goal_pos = self.params["goal_pos"]

        # Translation of goal:
        translation_gpu, translation_cpu = self.normalizations.translation_goal(state_goal = np.append(goal_pos, offset_orientation), goal_NN=self.goal_NN)

        # initial state:
        ob, *_ = self.env.step(np.zeros(self.params["dof"]))
        q_init = ob['robot_0']["joint_state"]["position"][0:self.params["dof"]]
        x_t_init = self.kuka_kinematics.get_initial_state_task(q_init=q_init, offset_orientation=offset_orientation)
        x_init_gpu = self.normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=translation_cpu, offset_orientation=offset_orientation)
        dynamical_system = self.learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

        # Initialize lists
        xee_list = []
        quat_prev = copy.deepcopy(x_t_init[3:])

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:self.params["dof"]]
            qdot = ob_robot["joint_state"]["velocity"][0:self.params["dof"]]
            if self.params["nr_obst"]>0:
                self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                self.obstacles = []

            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev)
            quat_prev = copy.deepcopy(xee_orientation)
            vel_ee, Jac_current = self.kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
            x_t_gpu = self.normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=translation_cpu, offset_orientation=offset_orientation)

            # --- action by NN --- #
            time0 = time.perf_counter()
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]
            if self.params["mode_NN"] == "2nd":
                print("not implemented!!")
            else:
                action_t_gpu = transition_info["desired velocity"]
                action_cpu = action_t_gpu.T.cpu().detach().numpy()
            x_t_action = self.normalizations.reverse_transformation_pos_quat(state_gpu=x_t_NN, offset_orientation=offset_orientation)
            action_safeMP = self.normalizations.reverse_transformation(action_gpu=action_t_gpu)

            # # -- transform to configuration space --#
            if self.params["mode_NN"] == "1st":
                # ---- option 1 -----#
                action_quat_vel = action_safeMP[3:]
                action_quat_vel_sys = self.kuka_kinematics.quat_vel_with_offset(quat_vel_NN=action_quat_vel,
                                                                           quat_offset=offset_orientation)
                action_safeMP_pulled = self.kuka_kinematics.inverse_diff_kinematics_quat(xdot=np.append(action_safeMP[:3], action_quat_vel_sys), angle_quaternion=xee_orientation).numpy()[0]

                # ---- option 2: with PD controller ------ #
                """
                euler_vel = kuka_kinematics.quat_vel_to_angular_vel(angle_quaternion=xee_orientation,
                                                                          vel_quaternion=action_quat_vel_sys)
                ee_velocity_NN = np.append(action_safeMP[0:3], euler_vel)
                ee_vel_quat_d = kuka_kinematics.angular_vel_to_quat_vel(angle_quaternion=xee_orientation, vel_angular=vel_ee[3:])
                action_safeMP_pulled, action_quat_vel, euler_vel = cartesian_controller.control_law(position_d=x_t_action[:3], #state_goal,
                                                                                                    orientation_d=x_t_action[3:], #orientation_goal,
                                                                                                    ee_pose=x_t[0],
                                                                                                    ee_velocity=vel_ee,
                                                                                                    ee_velocity_d=ee_velocity_NN,
                                                                                                    ee_vel_quat = action_quat_vel_sys,
                                                                                                    ee_vel_quat_d = ee_vel_quat_d,
                                                                                                    J=Jac_current,
                                                                                                    dt=dt)
                """
            else:
                print("not implemented!!")

            if self.params["mode"] == "acc" and self.params["mode_NN"] == "1st":
                # ---- get a decent acceleration based on the velocity command ---#
                action = self.pdcontroller.control(desired_velocity=action_safeMP_pulled, current_velocity=qdot)
            else:
                action = action_safeMP_pulled
            self.solver_times.append(time.perf_counter() - time0)
            self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=goal_pos)

            ob, *_ = self.env.step(action)

            print("desired orientation: ", offset_orientation)
            print("current orientation:", xee_orientation)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
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
            "solver_times": self.solver_times,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        return results


if __name__ == "__main__":
    example_class = example_kuka_stableMP_R3S3()

    # #for PLACE:
    # params_name_1st = "1st_order_R3S3_place_16may"
    # goal_pos = [0.6395448586963173, -0.0510945657935033, 0.27628701041039444]
    # goal_orientation = [2.9054564615095213, 0.33140746775418256, 2.5805536640100573]
    # r = R.from_euler("xyz", goal_orientation)
    # goal_matrix = r.as_matrix()
    # goal_quat = pk.matrix_to_quaternion(torch.FloatTensor(goal_matrix).cuda()).cpu().detach()

    #for sweep:
    # params_name_1st = "1st_order_R3S3_sweep_16may"
    # goal_pos = [0.6531716257929936, -0.4296983978204229, 0.6031781912582894]
    # goal_orientation = [1.299895934107249, 1.1436703298037756, 0.1834614362871042]
    # r = R.from_euler("xyz", goal_orientation)
    # goal_matrix = r.as_matrix()
    # goal_quat = pk.matrix_to_quaternion(torch.FloatTensor(goal_matrix).cuda()).cpu().detach()

    # example_class.overwrite_defaults(goal_pos=goal_pos, orientation_goal=goal_quat, params_name_1st=params_name_1st)
    example_class.construct_example()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
