import os
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from functions_stableMP_fabrics.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from functions_stableMP_fabrics.environments import trial_environments
from functions_stableMP_fabrics.analysis_utils import UtilsAnalysis
from functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka
import yaml
import time
from scipy.spatial.transform import Rotation as R
import pytorch_kinematics as pk
import torch

class example_kuka_fabrics():
    def __init__(self):
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = -1
        self.obstacles = []
        self.solver_times = []
        with open("config/kuka_fabrics.yaml", "r") as setup_stream:
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
    def set_planner(self):
        """
        Initializes the fabric planner for the panda robot.
        """
        self.construct_fk()
        planner = ParameterizedFabricPlannerExtended(
            self.params["dof"],
            self.forward_kinematics,
            time_step=self.params["dt"],
        )
        planner.set_components(
            collision_links=self.params["collision_links"],
            goal=self.goal,
            number_obstacles=self.params["nr_obst"],
            number_plane_constraints=1,
            limits=self.params["iiwa_limits"],
        )
        planner.concretize_extensive(mode=self.params["mode"], time_step=self.params["dt"], extensive_concretize=False, bool_speed_control=self.params["bool_speed_control"])
        return planner, self.forward_kinematics

    def compute_action_fabrics(self, q, ob_robot, obstacles: list, nr_obst=0):
        time0 = time.perf_counter()
        x_goal_1_x = ob_robot['FullSensor']['goals'][nr_obst+3]['position']
        x_goal_2_z = ob_robot['FullSensor']['goals'][nr_obst+4]['position']
        p_orient_rot_x = self.rot_matrix @ x_goal_1_x
        p_orient_rot_z = self.rot_matrix @ x_goal_2_z
        arguments_dict = dict(
            q=q,
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0 = ob_robot['FullSensor']['goals'][2+nr_obst]['position'],
            weight_goal_0 = ob_robot['FullSensor']['goals'][2+nr_obst]['weight'],
            x_goal_1 = p_orient_rot_x,
            weight_goal_1 = ob_robot['FullSensor']['goals'][3+nr_obst]['weight'],
            x_goal_2=p_orient_rot_z,
            weight_goal_2=ob_robot['FullSensor']['goals'][4 + nr_obst]['weight'],
            x_obsts=[obstacles[i]["position"] for i in range(len(obstacles))],
            radius_obsts=[obstacles[i]["size"] for i in range(len(obstacles))],
            radius_body_links=self.params["collision_radii"],
            constraint_0=[0, 0, 1, 0],
        )

        action = self.planner.compute_action(
            **arguments_dict)
        self.solver_times.append(time.perf_counter() - time0)
        return action, [], [], []

    def construct_example(self):
        # --- parameters --- #
        self.offset_orientation = np.array(self.params["orientation_goal"])
        self.goal_pos = self.params["goal_pos"]

        self.initialize_environment()
        self.planner, fk = self.set_planner()
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics, collision_links=self.params["collision_links"], collision_radii=self.params["collision_radii"])
        self.kuka_kinematics = KinematicsKuka()

        # rotation matrix for the goal orientation:
        self.rot_matrix = pk.quaternion_to_matrix(torch.FloatTensor(self.params["orientation_goal"]).cuda()).cpu().detach().numpy()

    def run_kuka_example(self):
        ob, *_ = self.env.step(np.zeros(self.dof))
        x_t_init = self.kuka_kinematics.get_initial_state_task(q_init=ob["robot_0"]["joint_state"]["position"][0:self.dof],
                                                          offset_orientation=self.offset_orientation,
                                                          mode_NN="1st")
        quat_prev = x_t_init[3:7]
        xee_list = []
        vee_list = []

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:self.dof]
            qdot = ob_robot["joint_state"]["velocity"][0:self.dof]

            if self.params["nr_obst"]>0:
                self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                self.obstacles = []

            # ----- Fabrics action ----#
            action, _, _, _ = self.compute_action_fabrics(q=q, ob_robot=ob_robot, nr_obst=self.params["nr_obst"], obstacles=self.obstacles)
            action = np.clip(action, -1 * np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            vel_ee,_ = self.kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
            xee_list.append(x_ee[0])
            vee_list.append(vel_ee)
            self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=self.obstacles)
            self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, quat_prev, x_goal=self.goal_pos)

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
    example_class = example_kuka_fabrics()
    q_init_list = [
        np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
        np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
        np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)),
        np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
        np.array((0.07, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
        np.array((-0.50, 0.6, -0.02, -1.5, 0.01, -0.5, -0.010)),
        np.array((0.51, 0.67, -0.17, -1.73, 0.25, -0.86, -0.11)),
        np.array((0.91, 0.79, -0.22, -1.33, 1.20, -1.76, -1.06)),
        np.array((0.83, 0.53, -0.11, -0.95, 1.05, -1.24, -1.45)),
        np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
    ]
    positions_obstacles_list = [
        [[0.5, 0., 0.55], [0.5, 0., 10.1]],
        [[0.5, 0.15, 0.05], [0.5, 0.15, 0.2]],
        [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],
        [[0.5, 0.02, 0.1], [0.24, 0.45, 10.2]],
        [[0.5, -0.0, 0.5], [0.3, -0.1, 10.5]],
        [[0.5, -0.05, 0.3], [0.5, 0.2, 10.25]],
        [[0.5, -0.0, 0.2], [0.5, 0.2, 10.4]],
        [[0.5, -0.0, 0.28], [0.5, 0.2, 10.4]],
        [[0.5, 0.25, 0.55], [0.5, 0.2, 10.4]],
        [[0.5, 0.1, 0.45], [0.5, 0.2, 10.4]],
    ]
    example_class.overwrite_defaults(init_pos=q_init_list[1], positions_obstacles=positions_obstacles_list[1])
    example_class.construct_example()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
