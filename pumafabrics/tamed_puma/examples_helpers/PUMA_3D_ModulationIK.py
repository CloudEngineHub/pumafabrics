import os
import numpy as np
import warnings
import importlib
import copy
import yaml
import time
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from pumafabrics.tamed_puma.utils.filters import PDController
from pumafabrics.tamed_puma.utils.normalizations_2 import normalization_functions
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.puma_extension.initializer import initialize_framework
from pumafabrics.tamed_puma.modulation_ik.Modulation_ik import IKGomp
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric
import os

class PUMA_modulationIK(ExampleGeneric):
    def __init__(self, file_name="kinova_ModulationIK_tomato", path_config="../pumafabrics/tamed_puma/config/"):
        super().__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        name_base = os.path.dirname(os.path.abspath(__file__))
        full_file_name = name_base + path_config + file_name + ".yaml"
        with open(full_file_name, "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.robot_name = self.params["robot_name"]
        self.solver_times = []
        warnings.filterwarnings("ignore")

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/../config/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )

    def construct_example(self, results_base_directory='../pumafabrics/puma_extension/'):

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

        # Initialize GOMP
        self.gomp_class  = IKGomp(  q_home=self.params["init_pos"],
                                    end_link_name = self.params["end_links"][0],
                                    robot_name = self.params["robot_name"],
                                    root_link_name=self.params["root_link"])
        self.gomp_class.construct_ik(nr_obst=self.params["nr_obst"], collision_links=self.params["collision_links"])

        # Normalization class
        self.normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"])

    def initialize_example(self, q_init):
        self.offset_orientation = np.array(self.params["orientation_goal"])

        # Translation of goal:
        goal_pos = self.params["goal_pos"]
        translation_gpu, translation_cpu = self.normalizations.translation_goal(state_goal = np.array(goal_pos), goal_NN=self.goal_NN)

        # initial state:
        x_t_init = self.gomp_class.get_initial_pose(q_init=q_init, offset_orientation=self.offset_orientation)
        x_init_gpu = self.normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=translation_cpu, offset_orientation=self.offset_orientation)
        self.dynamical_system = self.learner.init_dynamical_system(initial_states=x_init_gpu[:, :3].clone(), delta_t=1)

        # Initialize lists
        self.quat_prev = copy.deepcopy(x_t_init[3:])

    def run(self, runtime_arguments:dict):
        q = runtime_arguments["q"]
        qdot = runtime_arguments["qdot"]
        goal_pos = runtime_arguments["goal_pos"]
        positions_obstacles = runtime_arguments["positions_obstacles"]
        obstacles = runtime_arguments["obstacles"]

        # recompute translation to goal pose:
        translation_gpu, translation_cpu = self.normalizations.translation_goal(state_goal=np.array(goal_pos),
                                                                                goal_NN=self.goal_NN)

        # --- end-effector states and normalized states --- #
        x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, self.quat_prev)

        self.quat_prev = copy.deepcopy(xee_orientation)
        vel_ee, Jac_current = self.kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
        x_t_gpu = self.normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=translation_cpu,
                                                            offset_orientation=self.offset_orientation)

        # --- action by NN --- #
        time0 = time.perf_counter()
        centers_obstacles = [[100, 100, 100], [100, 100, 100]]
        for i in range(self.params["nr_obst"]):
            centers_obstacles[i][0:3] = self.params["positions_obstacles"][i]
        obstacles_struct = {"centers": centers_obstacles,
                            "axes": [[0.3, 0.3, 0.3]], "safety_margins": [[1., 1., 1.]]}
        transition_info = self.dynamical_system.transition(space='task', x_t=x_t_gpu[:, :3].clone(),
                                                           obstacles=obstacles_struct)
        x_t_NN = transition_info["desired state"]
        x_t_action = self.normalizations.reverse_transformation_position(position_gpu=x_t_NN)
        q_d, solver_flag = self.gomp_class.call_ik(x_t_action[0][0:3], self.params["orientation_goal"],
                                                   positions_obsts=positions_obstacles,
                                                   q_init_guess=q,
                                                   q_home=q)
        xee_IK, _ = self.gomp_class.get_current_pose(q=q_d, quat_prev=self.quat_prev)
        action = self.pdcontroller.control(desired_velocity=q_d, current_velocity=q)
        self.solver_times.append(time.perf_counter() - time0)
        action = np.clip(action, -1 * np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))

        self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=goal_pos)
        self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=obstacles,
                                                                         parent_link=self.params["root_link"])
        dist_to_goal = self.return_distance_goal_reached()
        self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, self.quat_prev, x_goal=goal_pos)
        return action, self.GOAL_REACHED, dist_to_goal, self.IN_COLLISION, x_t_action