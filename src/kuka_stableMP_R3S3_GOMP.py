import os
import numpy as np
from functions_stableMP_fabrics.filters import ema_filter_deriv, PDController
from agent.utils.normalizations_2 import normalization_functions
from functions_stableMP_fabrics.environments import trial_environments
from functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from functions_stableMP_fabrics.analysis_utils import UtilsAnalysis
import importlib
from initializer import initialize_framework
import copy
import yaml
from functions_stableMP_fabrics.GOMP_ik import IKGomp
import time

class example_kuka_stableMP_R3S3():
    def __init__(self):
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = -1
        self.solver_times = []
        with open("config/kuka_GOMP.yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.robot_name = self.params["robot_name"]

    def overwrite_defaults(self, render=None, init_pos=None, goal_pos=None, nr_obst=None, positions_obstacles=None):
        if render is not None:
            self.params["render"] = render

        if init_pos is not None:
            self.params["init_pos"] = init_pos

        if goal_pos is not None:
            self.params["goal_pos"] = goal_pos

        if nr_obst is not None:
            self.params["nr_obst"] = nr_obst

        if positions_obstacles is not None:
            self.params["positions_obstacles"] = positions_obstacles

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
        results_base_directory = './'
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0])
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])
        self.construct_fk()
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"])

        # Parameters
        if self.params["mode_NN"] == "1st":
            self.params_name = '1st_order_R3S3_converge'
        else:
            print("not implemented!!")
            self.params_name = '2nd_order_R3S3_saray'

        # Load parameters
        Params = getattr(importlib.import_module('params.' + self.params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        self.learner, _, data = initialize_framework(params, self.params_name, verbose=False)
        self.goal_NN = data['goals training'][0]

        # Initialize GOMP
        self.gomp_class  = IKGomp(q_home=self.params["init_pos"]) #q_home=q_init)
        self.gomp_class.construct_ik(radii_obsts=[0.1])

        # Normalization class
        self.normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"])


    def run_kuka_example(self):
        dof = self.params["dof"]
        dt = self.params["dt"]
        offset_orientation = np.array(self.params["orientation_goal"])

        action = np.zeros(dof)
        ob, *_ = self.env.step(action)

        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        if self.params["nr_obst"] > 0:
            self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
        else:
            self.obstacles = []

        # Translation of goal:
        translation_gpu, translation_cpu = self.normalizations.translation_goal(state_goal = np.append(self.params["goal_pos"], self.params["orientation_goal"]), goal_NN=self.goal_NN)

        # initial state:
        x_t_init = self.gomp_class.get_initial_pose(q_init=q_init, offset_orientation=offset_orientation)
        x_init_gpu = self.normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=translation_cpu, offset_orientation=offset_orientation)
        dynamical_system = self.learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

        # Initialize lists
        xee_list = []
        quat_prev = copy.deepcopy(x_t_init[3:])

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            positions_obstacles = [ob_robot["FullSensor"]["obstacles"][2]["position"]]

            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev)
            x_t, xee_orientation = self.gomp_class.get_current_pose(q, quat_prev=quat_prev)

            quat_prev = copy.deepcopy(xee_orientation)
            vel_ee, Jac_current = self.kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
            x_t_gpu = self.normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=translation_cpu, offset_orientation=offset_orientation)

            # --- action by NN --- #
            time0 = time.perf_counter()
            obstacle_struct = {"centers": [[0.5, -0.25, 0.5, 0., 0., 0., 0.]],
                               "axes": [[0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01]], "safety_margins": [[1.0]]}
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu, obstacles=obstacle_struct)
            x_t_NN = transition_info["desired state"]
            if self.params["mode_NN"] == "2nd":
                print("not implemented!!")
            else:
                action_t_gpu = transition_info["desired velocity"]
                action_cpu = action_t_gpu.T.cpu().detach().numpy()
            x_t_action = self.normalizations.reverse_transformation_pos_quat(state_gpu=x_t_NN, offset_orientation=offset_orientation)
            action_safeMP = self.normalizations.reverse_transformation(action_gpu=action_t_gpu)
            q_d, solver_flag = self.gomp_class.call_ik(x_t_action[0:3], x_t_action[3:7],
                                                  positions_obsts=positions_obstacles,
                                                  q_init_guess=q,
                                                  q_home=q)
            if solver_flag == False:
                q_d = q
            xee_IK, _ = self.gomp_class.get_current_pose(q=q_d, quat_prev=quat_prev)
            print("solver_flag:", solver_flag)
            action = 0.3*self.pdcontroller.control(desired_velocity=q_d, current_velocity=q)
            self.solver_times.append(time.perf_counter() - time0)
            action = np.clip(action, -1*np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))

            self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=self.params["goal_pos"])
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
            self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=self.obstacles)
            self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, quat_prev, x_goal=self.params["goal_pos"])

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
            "solver_times": self.solver_times,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        return results


if __name__ == "__main__":
    example_class = example_kuka_stableMP_R3S3()
    example_class.construct_example()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])