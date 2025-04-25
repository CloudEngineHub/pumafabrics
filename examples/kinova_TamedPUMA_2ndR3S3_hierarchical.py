import os
import numpy as np
import pybullet
import warnings
from pumafabrics.tamed_puma.utils.filters import PDController
from pumafabrics.tamed_puma.utils.normalizations_2 import normalization_functions
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.tamed_puma.tamedpuma.fabrics_controller import FabricsController
from pumafabrics.tamed_puma.tamedpuma.puma_controller import PUMAControl
import importlib
from pumafabrics.puma_extension.initializer import initialize_framework
import copy
import yaml
import time
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric
"""
Example of Kinova gen3-lite running ModulationIK.
"""
class example_kinova_tamedpuma_2ndR3S3_hierarchical(ExampleGeneric):
    def __init__(self, file_name="kinova_hierarchical_tomato", path_config="../pumafabrics/tamed_puma/config/"):
        super(ExampleGeneric, self).__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.solver_times = []
        with open(path_config + file_name + ".yaml", "r") as setup_stream:
             self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.robot_name = self.params["robot_name"]
        warnings.filterwarnings("ignore")

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_kinova(params=self.params)

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/../pumafabrics/tamed_puma/config/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )

    def construct_example(self, with_environment=True, results_base_directory='../pumafabrics/puma_extension/'):
        if with_environment:
            self.initialize_environment()

        # Construct classes:
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"],
                                              robot_name=self.params["robot_name"],
                                              root_link_name=self.params["root_link"],
                                              end_link_name=self.params["end_links"][0])
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])
        self.construct_fk()
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"],
                                            kinematics=self.kuka_kinematics)
        self.fabrics_controller = FabricsController(self.params)
        self.planner, fk = self.fabrics_controller.set_full_planner(goal=self.goal)

        # Parameters
        if self.params["mode_NN"] == "1st":
            self.params_name = self.params["params_name_1st"]
        else:
            self.params_name = self.params["params_name_2nd"]

        # Load parameters
        Params = getattr(importlib.import_module('pumafabrics.puma_extension.params.' + self.params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        self.learner, _, data = initialize_framework(params, self.params_name, verbose=False)
        self.goal_NN = data['goals training'][0]

        # Normalization class
        # self.normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"])

        self.puma_controller = PUMAControl(params=self.params, kinematics=self.kuka_kinematics, NULLSPACE=False)

    def run_kuka_example(self):
        dof = self.params["dof"]
        dt = self.params["dt"]
        offset_orientation = np.array(self.params["orientation_goal"])
        orientation_goal = np.array(self.params["orientation_goal"])

        action = np.zeros(dof)
        ob, *_ = self.env.step(action)

        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        if self.params["nr_obst"] > 0:
            self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
        else:
            self.obstacles = []

        # # initial state:
        goal_pos = self.params["goal_pos"]
        x_t_init, x_init_gpu, translation_cpu, goal_NN = self.puma_controller.initialize_PUMA(q_init=q_init, goal_pos=goal_pos, offset_orientation=offset_orientation)
        dynamical_system, normalizations = self.puma_controller.return_classes()

        # Initialize lists
        xee_list = []
        qdot_diff_list = []
        quat_prev = copy.deepcopy(x_t_init[3:7])

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            if self.params["nr_obst"]>0:
                self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                self.obstacles = []

            # recompute translation to goal pose:
            goal_pos = [goal_pos[i] + self.params["goal_vel"][i]*self.params["dt"] for i in range(len(goal_pos))]
            translation_gpu, translation_cpu = normalizations.translation_goal(state_goal=np.append(goal_pos, orientation_goal), goal_NN=self.goal_NN)
            pybullet.addUserDebugPoints([goal_pos], [[1, 0, 0]], 5, 0.1)

            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev, mode_NN=self.params["mode_NN"], qdot=qdot)
            quat_prev = copy.deepcopy(xee_orientation)

            # --- action by NN --- #
            time0 = time.perf_counter()
            x_t_action, transition_info = self.puma_controller.request_PUMA(q=q,
                                                                            qdot=qdot,
                                                                            x_t=x_t,
                                                                            xee_orientation=xee_orientation,
                                                                            offset_orientation=offset_orientation,
                                                                            translation_cpu=translation_cpu,
                                                                            POS_OUTPUT= True
                                                                            )
            # ----- Fabrics action ----#
            action, _, _, _ = self.fabrics_controller.compute_action_full(q=q, ob_robot=ob_robot,
                                                                          nr_obst=self.params["nr_obst"],
                                                                          obstacles=self.obstacles,
                                                                          goal_pos=x_t_action,
                                                                          weight_goal_0=10)

            pybullet.addUserDebugPoints(list([x_t_action[0:3]]), [[1, 0, 0]], 5, 0.1)
            self.solver_times.append(time.perf_counter() - time0)
            action = np.clip(action, -1*np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))

            self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=goal_pos)
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
            qdot_diff_list.append(1)
            self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=self.obstacles, parent_link=self.params["root_link"])
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
            "qdot_diff_list": qdot_diff_list,
            "solver_times": np.array(self.solver_times)*1000,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        return results

def main(render=True, n_steps=None):
    q_init_list = [
        np.array((-0.13467712, -0.26750494,  0.04957501,  1.53952123,  1.4392644,  -1.57109087))
    ]
    positions_obstacles_list = [
        [[0.0, 0., 1.55], [0.5, 0.2, 0.6]],
    ]
    goal_pos_list = [
        [0.53858072, -0.04530622,  0.4580668]
    ]
    example_class = example_kinova_tamedpuma_2ndR3S3_hierarchical()
    example_class.overwrite_defaults(params=example_class.params, init_pos=q_init_list[0],
                                     goal_pos=goal_pos_list[0],
                                     positions_obstacles=positions_obstacles_list[0],
                                     render=render, n_steps=n_steps)
    example_class.construct_example()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
    return {}

if __name__ == "__main__":
    main()