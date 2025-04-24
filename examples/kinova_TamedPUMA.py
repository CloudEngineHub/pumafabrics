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
import pybullet
"""
Example of the Kinova gen3-lite running TamedPUMA, this example is IN PROGRESS.
"""
class example_kinova_TamedPUMA(ExampleGeneric):
    def __init__(self, file_name="kinova_TamedPUMA_tomato"):
        super(ExampleGeneric, self).__init__()
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.solver_times = []
        with open("../pumafabrics/tamed_puma/config/"+file_name+".yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.network_yaml = file_name
        self.robot_name = self.params["robot_name"]

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_kinova(params=self.params)

    def construct_example(self):
        self.initialize_environment()
        self.fabrics_controller = FabricsController(self.params)
        self.planner_avoidance, self.fk = self.fabrics_controller.set_avoidance_planner(goal=None)
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"],
                                              end_link_name=self.params["end_links"][0],
                                              robot_name=self.params["robot_name"],
                                              root_link_name=self.params["root_link"],)
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.fk,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"],
                                            kinematics=self.kuka_kinematics)
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])
        self.puma_controller = PUMAControl(params=self.params, kinematics=self.kuka_kinematics, NULLSPACE=False)

    def run_kinova_example(self):
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

        # _, Jac_function = self.kuka_kinematics.forward_kinematics_symbolic(fk=self.fk)

        # Initialize lists
        xee_list = []
        qdot_diff_list = []
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

            # recompute translation to goal pose:
            goal_pos = [goal_pos[i] + self.params["goal_vel"][i]*self.params["dt"] for i in range(len(goal_pos))]
            translation_gpu, translation_cpu = normalizations.translation_goal(state_goal=np.append(goal_pos, orientation_goal), goal_NN=goal_NN)
            energy_regulation_class.relationship_dq_dx(offset_orientation, translation_cpu, self.kuka_kinematics,
                                                       normalizations, self.fk)
            pybullet.addUserDebugPoints([goal_pos], [[1, 0, 0]], 5, 0.1)

            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev, mode_NN=self.params["mode_NN"], qdot=qdot)
            quat_prev = copy.deepcopy(xee_orientation)

            # jacobian_casadi = Jac_function(q)
            # print("casadi jacobian:")
            # self.kuka_kinematics.check_instability_jacobian(np.linalg.inv(jacobian_casadi + 1e-3*np.eye(6)))
            # print("end check instability jacobian casadi")
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
                action_avoidance, M_avoidance, f_avoidance, qddot_speed = self.fabrics_controller.compute_action_avoidance(q=q, ob_robot=ob_robot)

                if self.params["bool_energy_regulator"] == True:
                    weight_attractor = 1.
                    # ---- get action by CPM via theorem III.5 in https://arxiv.org/pdf/2309.07368.pdf ---#
                    action_combined = energy_regulation_class.compute_action_theorem_III5(q=q, qdot=qdot,
                                                                                          qddot_attractor = qddot_PUMA,
                                                                                          action_avoidance=action_avoidance,
                                                                                          M_avoidance=M_avoidance,
                                                                                          transition_info=transition_info,
                                                                                          weight_attractor=weight_attractor)
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

def main(render=True):
    q_init_list = [
        np.array((9.480853480682088e-05, 6.418218227090965e-05, 0.0005355616952149348, 1.6995958518588348, 1.5725444257344245, -1.570403244218001))

    ]
    positions_obstacles_list = [
        [[10.0, 0., 0.55], [0.5, 0., 10.1]],
    ]
    speed_obstacles_list = [
        [[0., 0., 0.], [0., 0., 0.]],
    ]
    goal_pos_list = [
        [0.53858072, -0.04530622,  0.4580668]
    ]
    goal_vel_list = [
        [0., 0., 0.] for _ in range(len(q_init_list))
    ]
    id_nr = 0
    network_yaml = "kinova_TamedPUMA_tomato"
    example_class = example_kinova_TamedPUMA(file_name=network_yaml)
    example_class.overwrite_defaults(params=example_class.params, init_pos=q_init_list[id_nr], positions_obstacles=positions_obstacles_list[id_nr], render=render, speed_obstacles=speed_obstacles_list[id_nr], goal_pos=goal_pos_list[id_nr], goal_vel=goal_vel_list[id_nr])
    example_class.construct_example()
    res = example_class.run_kinova_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
    return {}

if __name__ == "__main__":
    main()