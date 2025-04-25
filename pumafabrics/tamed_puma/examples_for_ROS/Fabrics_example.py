import numpy as np
from pumafabrics.tamed_puma.tamedpuma.fabrics_controller import FabricsController
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric
from pumafabrics.tamed_puma.create_environment.goal_defaults import goal_default
import yaml

class FabricsExample(ExampleGeneric):
    def __init__(self, file_name):
        super().__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.obstacles = []
        with open("../pumafabrics/tamed_puma/config/" + file_name + ".yaml", "r") as setup_stream:
             self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.params["bool_extensive_concretize"] = False
        self.robot_name = self.params["robot_name"]

    def construct_example(self):
        # --- parameters --- #
        self.offset_orientation = np.array(self.params["orientation_goal"])
        self.fabrics_controller = FabricsController(params=self.params)
        self.goal = goal_default(robot_name=self.params["robot_name"], end_effector_link=self.params["end_links"][0], goal_pos=self.params["goal_pos"])
        self.planner, fk = self.fabrics_controller.set_full_planner(goal=self.goal)
        self.kinova_kinematics = KinematicsKuka(end_link_name=self.params["end_links"][0], robot_name=self.robot_name, root_link_name=self.params["root_link"])
        self.utils_analysis = UtilsAnalysis(forward_kinematics=fk,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"],
                                            kinematics=self.kinova_kinematics)

    def initialize_example(self, q_init):
        self.offset_orientation = np.array(self.params["orientation_goal"])
        x_t_init = self.kinova_kinematics.get_initial_state_task(q_init=q_init, qdot_init=np.zeros((self.params["dof"], 1)),
                                                                 offset_orientation=self.offset_orientation,
                                                                 mode_NN=self.params["mode_NN"])
        self.quat_prev = x_t_init[3:7]

    def run(self, runtime_arguments):
        q = runtime_arguments["q"]
        qdot = runtime_arguments["qdot"]
        goal_pos = runtime_arguments["goal_pos"]
        obstacles = runtime_arguments["obstacles"]

        # ----- Fabrics action ----#
        action, _, _, _ = self.fabrics_controller.compute_action_full(q=q, qdot=qdot,
                                                                      obstacles=obstacles,
                                                                      goal_pos=goal_pos)
        if self.params["mode_env"] == "vel" and self.params[
            "mode"] == "acc":  # todo: fix nicely or mode == "acc"): #mode_NN=="2nd":
            action = self.integrate_to_vel(qdot=qdot, action_acc=action, dt=self.params["dt"])
            action = np.clip(action, -1 * np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))
        else:
            action = action

        self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q[0:6], obstacles=obstacles,
                                                                         parent_link=self.params["root_link"])
        self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, self.quat_prev, x_goal=goal_pos)
        return action, self.GOAL_REACHED, error, self.IN_COLLISION, {}
