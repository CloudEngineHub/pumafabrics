import os
import yaml
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.tamedpuma.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from pumafabrics.tamed_puma.tamedpuma.energy_regulator import energy_regulation
from pumafabrics.tamed_puma.tamedpuma.puma_controller import PUMAControl
from pumafabrics.tamed_puma.nullspace_control.nullspace_controller import CartesianImpedanceController
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.tamed_puma.utils.filters import PDController
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric
import copy
import time
"""
Example of KUKA iiwa 14 running TamedPUMA as a controller with lots of obstacles..
"""
class example_kuka_TamedPUMA_1000(ExampleGeneric):
    def __init__(self):
        super(ExampleGeneric, self).__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.solver_times = []
        with open("../pumafabrics/tamed_puma/config/kuka_TamedPUMA_tomato.yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.robot_name = self.params["robot_name"]

    def initialize_environment(self):
        envir_trial = trial_environments()
        self.params["nr_obst"]=2
        (self.env, self.goal) = envir_trial.initialize_environment_kuka(params=self.params)
        self.params["nr_obst"]=100

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/../pumafabrics/tamed_puma/config/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )
    def set_planner(self, goal: GoalComposition): #, degrees_of_freedom: int = 7, mode="acc", dt=0.01, bool_speed_control=True):
        """
        Initializes the fabric planner
        """
        self.construct_fk()
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path +  "/../pumafabrics/tamed_puma/config/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            root_link="iiwa_link_0",
            end_links=["iiwa_link_7"],
        )
        planner = ParameterizedFabricPlannerExtended(
            self.params["dof"],
            self.forward_kinematics,
            time_step=self.params["dt"],
        )
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            collision_links=self.params["collision_links"],
            goal=goal,
            number_obstacles=self.params["nr_obst"],
            number_plane_constraints=0,
            limits=self.params["iiwa_limits"],
        )
        planner.concretize_extensive(mode=self.params["mode"], time_step=self.params["dt"], extensive_concretize=self.params["bool_extensive_concretize"], bool_speed_control=self.params["bool_speed_control"])
        return planner, forward_kinematics

    def compute_action_fabrics(self, q, ob_robot):
        nr_obst = self.params["nr_obst"]
        if nr_obst>19:
            x_obsts = [[10, 10, 10] for i in range(nr_obst)]
            x_obsts[0] = list(ob_robot['FullSensor']['obstacles'][2]['position'])
            x_obsts[1] = list(ob_robot['FullSensor']['obstacles'][3]['position'])
            radius_obsts = [[0.5] for i in range(nr_obst)]
            radius_obsts[0] = list(ob_robot['FullSensor']['obstacles'][2]['size'])
            radius_obsts[1] = list(ob_robot['FullSensor']['obstacles'][3]['size'])
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                x_obsts=x_obsts,
                radius_obsts=radius_obsts,
                radius_body_links=self.params["collision_radii"],
                constraint_0=np.array([0, 0, 1, 0.0]))
        elif nr_obst>0:
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                x_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
                radius_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['size'],
                x_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position'],
                radius_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size'],
                radius_body_links=self.params["collision_radii"],
                constraint_0=np.array([0, 0, 1, 0.0]))
        else:
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                radius_body_links=self.params["collision_radii"],
                constraint_0=np.array([0, 0, 1, 0.0]))
        M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = self.planner_avoidance.compute_M_f_action_avoidance(
            **arguments_dict)
        qddot_speed = np.zeros((self.params["dof"],))  # todo: think about what to do with speed regulation term!!
        return action_avoidance, M_avoidance, f_avoidance, qddot_speed

    def construct_example(self):
        self.initialize_environment()
        self.planner_avoidance, self.fk = self.set_planner(goal=None)
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0], robot_name=self.params["robot_name"])
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"],
                                            kinematics=self.kuka_kinematics)
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])
        self.puma_controller = PUMAControl(params=self.params, kinematics=self.kuka_kinematics)
        self.controller_nullspace = CartesianImpedanceController(robot_name=self.params["robot_name"])

    def run_kuka_example(self):
        # --- parameters --- #
        n_steps = self.params["n_steps"]
        orientation_goal = np.array(self.params["orientation_goal"])
        offset_orientation = np.array(self.params["orientation_goal"])
        goal_pos = self.params["goal_pos"]
        dof = self.params["dof"]
        action = np.zeros(dof)
        ob, *_ = self.env.step(action)

        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        x_t_init, x_init_gpu, translation_cpu, goal_NN = self.puma_controller.initialize_PUMA(q_init=q_init, goal_pos=goal_pos, offset_orientation=offset_orientation)
        dynamical_system, normalizations = self.puma_controller.return_classes()

        # energization:
        energy_regulation_class = energy_regulation(dim_task=7, mode_NN=self.params["mode_NN"], dof=dof, dynamical_system=dynamical_system)
        energy_regulation_class.relationship_dq_dx(offset_orientation, translation_cpu, self.kuka_kinematics, normalizations, self.fk)

        # Initialize lists
        xee_list = []
        qdot_diff_list = []
        quat_prev = copy.deepcopy(x_t_init[3:7])

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            if self.params["nr_obst"]>0 and self.params["nr_obst"]<20:
                self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                self.obstacles = []

            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev, mode_NN=self.params["mode_NN"], qdot=qdot)
            quat_prev = copy.deepcopy(xee_orientation)

            # --- action by NN --- #
            time0 = time.perf_counter()
            qddot_PUMA, transition_info = self.puma_controller.request_PUMA(q=q,
                                                                                qdot=qdot,
                                                                                x_t=x_t,
                                                                                xee_orientation=xee_orientation,
                                                                                offset_orientation=offset_orientation,
                                                                                translation_cpu=translation_cpu
                                                                                )
            if self.params["bool_combined"] == True:
                # ----- Fabrics action ----#
                action_avoidance, M_avoidance, f_avoidance, qddot_speed = self.compute_action_fabrics(q=q, ob_robot=ob_robot)

                if self.params["bool_energy_regulator"] == True:
                    # ---- get action by CPM via theorem III.5 in https://arxiv.org/pdf/2309.07368.pdf ---#
                    action_combined = energy_regulation_class.compute_action_theorem_III5(q=q, qdot=qdot,
                                                                                          qddot_attractor = qddot_PUMA,
                                                                                          action_avoidance=action_avoidance,
                                                                                          M_avoidance=M_avoidance,
                                                                                          transition_info=transition_info)
                else:
                    # --- get action by FPM, sum of dissipative systems ---#
                    action_combined = qddot_PUMA + action_avoidance
            else: #otherwise only apply action by PUMA
                action_combined = qddot_PUMA

            if self.params["mode_env"] is not None:
                if self.params["mode_env"] == "vel": # todo: fix nicely or mode == "acc"): #mode_NN=="2nd":
                    action = self.integrate_to_vel(qdot=qdot, action_acc=action_combined, dt=self.params["dt"])
                    action = np.clip(action, -1*np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))
                else:
                    action = action_combined
            else:
                action = action_combined
            self.solver_times.append(time.perf_counter() - time0)
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
            qdot_diff_list.append(np.mean(np.absolute(qddot_PUMA   - action_combined)))
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
            "qdot_diff_list": qdot_diff_list,
            "solver_times": self.solver_times,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        return results

def main(render=True, n_steps=None):
    q_init_list = [
        np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
    ]
    positions_obstacles_list = [
        [[0.45, 0.02, 0.2], [0.6, 0.02, 0.2]],
    ]
    example_class = example_kuka_TamedPUMA_1000()
    example_class.overwrite_defaults(params=example_class.params, init_pos=q_init_list[0], positions_obstacles=positions_obstacles_list[0],
                                     render=render, nr_obst=100, n_steps=n_steps)
    # Note: the number of obstacles is set to a 100 here and overwritten to 1000 later, to avoid generating 1000 obstacles in Pybullet, but still considering 1000 obstacles in the controller.
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