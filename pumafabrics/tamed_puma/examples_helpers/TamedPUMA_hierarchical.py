import os
import numpy as np
import warnings
from pumafabrics.tamed_puma.utils.filters import PDController
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.tamed_puma.tamedpuma.fabrics_controller import FabricsController
from pumafabrics.tamed_puma.tamedpuma.puma_controller import PUMAControl
import importlib
from pumafabrics.puma_extension.initializer import initialize_framework
from pumafabrics.tamed_puma.create_environment.goal_defaults import goal_default
import copy
import yaml
import time, os
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric

class TamedPUMAhierarchical(ExampleGeneric):
    def __init__(self, file_name="kinova_hierarchical_tomato", path_config="../pumafabrics/tamed_puma/config/", mode_NN="1st"):
        super(ExampleGeneric, self).__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.solver_times = []
        name_base = os.path.dirname(os.path.abspath(__file__))
        full_file_name = name_base+path_config + file_name + ".yaml"
        with open(full_file_name, "r") as setup_stream:
             self.params = yaml.safe_load(setup_stream)
        self.params["mode_NN"] = mode_NN
        self.dof = self.params["dof"]
        self.robot_name = self.params["robot_name"]
        warnings.filterwarnings("ignore")

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        print("self.robot_name: ", self.robot_name)
        print("self.params['root_link']: ", self.params["root_link"])
        print("self.params['end_links']: ", self.params["end_links"])
        with open(absolute_path + "/../config/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )

    def construct_example(self, with_environment=True, results_base_directory='../pumafabrics/puma_extension/'):

        # Construct classes:
        self.results_base_directory=results_base_directory
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
        self.goal = goal_default(robot_name=self.params["robot_name"], end_effector_link=self.params["end_links"][0],
                                 goal_pos=self.params["goal_pos"])
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
        self.puma_controller = PUMAControl(params=self.params, kinematics=self.kuka_kinematics, NULLSPACE=False)

    def initialize_example(self, q_init):
        self.offset_orientation = np.array(self.params["orientation_goal"])

        # # initial state:
        goal_pos = self.params["goal_pos"]
        x_t_init, x_init_gpu, translation_cpu, goal_NN = self.puma_controller.initialize_PUMA(q_init=q_init, goal_pos=goal_pos, offset_orientation=self.offset_orientation, results_base_directory=self.results_base_directory)
        self.dynamical_system, self.normalizations = self.puma_controller.return_classes()

        self.quat_prev = copy.deepcopy(x_t_init[3:7])

    def run(self, runtime_arguments: dict):
        q = runtime_arguments["q"]
        qdot = runtime_arguments["qdot"]
        goal_pos = runtime_arguments["goal_pos"]
        obstacles = runtime_arguments["obstacles"]

        if self.params["dim_task"] == 7:
            state_goal = np.append(goal_pos, self.offset_orientation)
        elif self.params["dim_task"] == 3:
            state_goal = goal_pos

        translation_gpu, translation_cpu = self.normalizations.translation_goal(state_goal=state_goal, goal_NN=self.goal_NN)

        # --- end-effector states and normalized states --- #
        x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, self.quat_prev, mode_NN=self.params["mode_NN"], qdot=qdot)
        self.quat_prev = copy.deepcopy(xee_orientation)
        # print("x_t: ", x_t)
        # print("q: ", q)

        # --- action by NN --- #
        time0 = time.perf_counter()

        # get one action further away to avoid small drag
        x_t_propagate = copy.deepcopy(x_t)
        for z in range(50):
            x_t_action, transition_info = self.puma_controller.request_PUMA(q=q,
                                                                            qdot=qdot,
                                                                            x_t=x_t_propagate,
                                                                            xee_orientation=xee_orientation,
                                                                            offset_orientation=self.offset_orientation,
                                                                            translation_cpu=translation_cpu,
                                                                            POS_OUTPUT=True
                                                                            )
            if len(x_t_action) == 1:
                x_t_action = x_t_action[0]
            x_t_propagate[0][:self.params["dim_task"]] = x_t_action

        # ----- Fabrics action ----#
        action, _, _, _ = self.fabrics_controller.compute_action_full(q=q, qdot=qdot,
                                                                      obstacles=obstacles,
                                                                      goal_pos=x_t_action[0:3],
                                                                      weight_goal_0=5,
                                                                      goal_orient=xee_orientation,
                                                                      weight_goal_1=1,
                                                                      weight_goal_2=1)

        self.solver_times.append(time.perf_counter() - time0)
        action = np.clip(action, -1 * np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))
        self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=goal_pos)
        self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=obstacles, parent_link=self.params["root_link"])
        self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, self.quat_prev, x_goal=goal_pos)
        # print("x_t_action: ", x_t_action)
        return action, self.GOAL_REACHED, error, self.IN_COLLISION, x_t_action