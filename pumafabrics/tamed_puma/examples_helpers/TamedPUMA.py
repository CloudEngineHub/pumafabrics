import yaml
import numpy as np
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from pumafabrics.tamed_puma.tamedpuma.energy_regulator import energy_regulation
from pumafabrics.tamed_puma.tamedpuma.puma_controller import PUMAControl
from pumafabrics.tamed_puma.tamedpuma.fabrics_controller import FabricsController
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.tamed_puma.utils.filters import PDController
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric
import copy
import time

class TamedPUMAExample(ExampleGeneric):
    def __init__(self, file_name="kuka_TamedPUMA_tomato"):
        super(ExampleGeneric, self).__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.solver_times = []
        with open("../pumafabrics/tamed_puma/config/"+file_name+".yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.network_yaml = file_name
        self.robot_name = self.params["robot_name"]

    def construct_example(self):
        self.fabrics_controller = FabricsController(self.params)
        self.planner_avoidance, self.fk = self.fabrics_controller.set_avoidance_planner(goal=None)
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0], robot_name=self.params["robot_name"])
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.fk,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"],
                                            kinematics=self.kuka_kinematics)
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])
        self.puma_controller = PUMAControl(params=self.params, kinematics=self.kuka_kinematics)

    def initialize_example(self, q_init):
        self.offset_orientation = np.array(self.params["orientation_goal"])
        dof = self.params["dof"]
        goal_pos = self.params["goal_pos"]

        x_t_init, x_init_gpu, translation_cpu, self.goal_NN = self.puma_controller.initialize_PUMA(q_init=q_init,
                                                                                              goal_pos=goal_pos,
                                                                                              offset_orientation=self.offset_orientation)
        self.dynamical_system, self.normalizations = self.puma_controller.return_classes()
        self.quat_prev = copy.deepcopy(x_t_init[3:7])

        # energization:
        self.energy_regulation_class = energy_regulation(dim_task=self.params["dim_task"], mode_NN=self.params["mode_NN"], dof=dof, dynamical_system=self.dynamical_system)
        self.energy_regulation_class.relationship_dq_dx(self.offset_orientation, translation_cpu, self.kuka_kinematics, self.normalizations, self.fk)

    def run(self, runtime_arguments):
        q = runtime_arguments["q"]
        qdot = runtime_arguments["qdot"]
        goal_pos = runtime_arguments["goal_pos"]
        obstacles = runtime_arguments["obstacles"]

        translation_gpu, translation_cpu = self.normalizations.translation_goal(
            state_goal=np.append(goal_pos, self.offset_orientation), goal_NN=self.goal_NN)
        self.energy_regulation_class.relationship_dq_dx(self.offset_orientation, translation_cpu, self.kuka_kinematics,
                                                        self.normalizations, self.fk)

        # --- end-effector states and normalized states --- #
        x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, self.quat_prev, mode_NN=self.params["mode_NN"],
                                                                      qdot=qdot)
        self.quat_prev = copy.deepcopy(xee_orientation)

        # --- action by NN --- #
        time0 = time.perf_counter()
        qddot_PUMA, transition_info = self.puma_controller.request_PUMA(q=q,
                                                                        qdot=qdot,
                                                                        x_t=x_t,
                                                                        xee_orientation=xee_orientation,
                                                                        offset_orientation=self.offset_orientation,
                                                                        translation_cpu=translation_cpu
                                                                        )

        if self.params["bool_combined"] == True:
            # ----- Fabrics action ----#
            action_avoidance, M_avoidance, f_avoidance, qddot_speed = self.fabrics_controller.compute_action_avoidance(
                q=q, qdot=qdot, obstacles=obstacles)

            if self.params["bool_energy_regulator"] == True:
                weight_attractor = 1.
                # ---- get action by CPM via theorem III.5 in https://arxiv.org/pdf/2309.07368.pdf ---#
                action_combined = self.energy_regulation_class.compute_action_theorem_III5(q=q, qdot=qdot,
                                                                                           qddot_attractor=qddot_PUMA,
                                                                                           action_avoidance=action_avoidance,
                                                                                           M_avoidance=M_avoidance,
                                                                                           transition_info=transition_info,
                                                                                           weight_attractor=weight_attractor)
            else:
                # --- get action by FPM, sum of dissipative systems ---#
                action_combined = qddot_PUMA + action_avoidance
        else:  # otherwise only apply action by PUMA
            action_combined = qddot_PUMA

        if self.params["mode_env"] is not None:
            if self.params["mode_env"] == "vel":  # todo: fix nicely or mode == "acc"): #mode_NN=="2nd":
                action = self.integrate_to_vel(qdot=qdot, action_acc=action_combined, dt=self.params["dt"])
                action = np.clip(action, -1 * np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))
            else:
                action = action_combined
        else:
            action = action_combined

        self.solver_times.append(time.perf_counter() - time0)

        self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=obstacles)
        self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, self.quat_prev, x_goal=goal_pos)
        return action, self.GOAL_REACHED, error, self.IN_COLLISION, {}, qddot_PUMA