"""
This file is to generate a comparison plot between fabrics, PUMA and TamedPUMA (FPM) and TamedPUMA (CPM) and ModulationIK.
"""
import os
import numpy as np
# from examples.kuka_Fabrics import example_kuka_fabrics
# from examples.kuka_TamedPUMA import example_kuka_TamedPUMA
# from examples.kuka_PUMA_3D_ModulationIK import example_kuka_PUMA_modulationIK
import pickle
from evaluations.kuka_comparison_2nd_tomato import comparison_kuka_class
# from evaluations.comparison_general import ComparisonGeneral
# from pumafabrics.tamed_puma.utils.record_data import RecordData, EvaluationDataStructure
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from pumafabrics.tamed_puma.modulation_ik.GOMP_ik import IKGomp
import yaml

class ReadDemonstrations():
    def __init__(self, file_name = "kuka_stableMP_fabrics_2nd_pouring"):
        with open("config/"+file_name+".yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0],
                                          robot_name=self.params["robot_name"])
        self.create_ik_controller()

    def create_ik_controller(self):
        self.modulation_class  = IKGomp(q_home=self.params["init_pos"],
                                        end_link_name=self.params["end_links"][0])
        self.modulation_class.construct_ik(nr_obst=0)

    def call_ik_controller(self, x_t_position, x_t_orientation, q_guess):
        q_d, solver_flag = self.modulation_class.call_ik(x_t_position, x_t_orientation,
                                                         positions_obsts=[],
                                                         q_init_guess=q_guess,
                                                         q_home=q_guess)
        return q_d, solver_flag

    def read_demonstrations(self):
        goal_pos_list = []
        q_init_list = []
        current_directory = os.getcwd()
        demonstrations_ee = {f'ee_state_{i}': [] for i in range(12)}
        demonstrations_joints = {f"joints_state_{i}": {"q": [], "delta_t": []} for i in range(12)}
        for i in range(10):
            q_guess = np.array([1.28421, -1.08275, 0.709752, 1.22488, 2.78079, -0.549531, -0.868621])
            file_demo_ee = open(
                current_directory + f"/datasets/kuka/pouring/ee_state_{i}.pk", 'rb')
            demonstration_ee = pickle.load(file_demo_ee)
            demonstrations_ee[f"ee_state_{i}"] = demonstration_ee
            if i==0:
                print("demonstration end-effector position 0: ", demonstrations_ee[f"ee_state_{i}"]['x_pos'][0])
            goal_pos_list.append([float(demonstration_ee['x_pos'][-1][i]) for i in range(3)])

            for i_demo in range(len(demonstration_ee["x_pos"])):
                q_d, solver_flag  = self.call_ik_controller(demonstration_ee["x_pos"][i], demonstration_ee["x_rot"][0], q_guess=q_guess)
                q_guess = q_d
                demonstrations_joints[f"joints_state_{i}"]["q"].append(q_d)
            demonstrations_joints[f"joints_state_{i}"]["delta_t"] = demonstration_ee["delta_t"]
            q_init_list.append(np.array(demonstrations_joints[f"joints_state_{i}"]["q"][0]))
        return demonstrations_ee, goal_pos_list, q_init_list

    def get_list_ee_via_fk(self, q_list):
        x_ee_demonstrations = [self.kuka_kinematics.forward_kinematics(list(q), end_link_name=self.params["end_links"][0])[0:3] for q in q_list]
        return x_ee_demonstrations

    def timesteps_from_dt(self, dt_list: list):
        time_steps = [0.]
        for dt in dt_list:
            time_steps.append(time_steps[-1] + dt)
        time_steps.pop(0)
        return time_steps


class comparisonDemonstrations():
    def __init__(self):
        self.cases = ["PUMA$_free$", "PUMA$_obst$",  "Occlusion-IK", "GF", "GM", "CM"]
        self.results_tot = {"PUMA$_free$": {"solver_times":[]}, "PUMA$_obst$": {"solver_times":[]},  "Occlusion-IK": {"solver_times":[]}, "GF": {"solver_times":[]}, "GM": {"solver_times":[]}, "CM": {"solver_times":[]}}
        self.results = {"min_distance": [], "collision": [], "goal_reached": [], "time_to_goal": [], "xee_list": [], "qdot_diff_list": [],
           "dist_to_NN": [],  "vel_to_NN": [], "solver_times": [], "solver_time": [], "solver_time_std": []}
        self.n_runs_total = 20

    def produce_results_pouring(self, q_init_list, goal_pos_list, nr_obst=0):
        nr_obst = nr_obst
        positions_obstacles_list = [
            # changing
            [[0., -0.6, 0.2], [0.5, 0., 10.1]],  # 0
            [[0.0, -0.6, 0.2], [0.5, 0.2, 10.4]],  # 1
            [[-0.20, -0.72, 0.22], [0.24, 0.45, 10.2]],  # 2
            [[-0.1, -0.65, 0.4], [0.6, 0.02, 10.2]],  # 3
            [[-0.1, -0.5, 0.22], [0.3, -0.1, 10.5]],  # 4
            # # #others:
            [[0.1, -0.56, 0.3], [0.5, 0., 10.1]],  # 0
            [[0.15, -0.6, 0.3], [0.5, 0.15, 0.2]],  # 1
            [[-0.1, -0.72, 0.22], [0.6, 0.02, 10.2]],  # 2
            [[-0.1, -0.72, 0.22], [0.3, -0.1, 10.5]],  # 3
            [[-0.1, -0.72, 0.22], [0.5, 0.2, 10.25]],  # 4
            [[0.03, -0.6, 0.15], [0.5, 0.2, 10.4]],  # 5
            [[0.0, -0.6, 0.10], [0.5, 0.2, 10.4]],  # 6
            [[0., -0.72, 0.1], [0.5, 0.2, 10.4]],  # 7
            [[0.0, -0.72, 0.10], [0.5, 0.2, 10.4]],  # 8
            [[0.0, -0.6, 0.10], [0.5, 0.2, 10.4]],  # 9
        ]
        speed_obstacles_list = [
            # # changing goal pose:
            [[0.03, 0., 0.], [0., 0., 0.]],  # 0
            [[0.03, 0., 0.], [0., 0., 0.]],  # 1
            [[0.02, 0., 0.], [0., 0., 0.]],  # 2
            [[0.03, 0., 0.], [0., 0., 0.]],  # 3
            [[0.05, 0., 0.], [0., 0., 0.]],  # 4
            # # others:
            [[0., 0., 0.], [0., 0., 0.]],  # 0
            [[0., 0., 0.], [0., 0., 0.]],  # 1
            [[0.05, 0., 0.], [0., 0., 0.]],  # 2
            [[0.04, 0., 0.], [0., 0., 0.]],  # 3
            [[0.05, 0., 0.], [0., 0., 0.]],  # 4
            [[0., 0., 0.], [0., 0., 0.]],  # 5
            [[0., 0., 0.], [0., 0., 0.]],  # 6
            [[0.01, 0., 0.], [0., 0., 0.]],  # 7
            [[0., 0., 0.], [0., 0., 0.]],  # 8
            [[0., 0., 0.], [0., 0., 0.]],  # 9
        ]
        goal_vel_list = [
            [0., 0., 0.] for _ in range(len(q_init_list))
        ]
        n_runs = len(q_init_list)
        network_yaml = "kuka_stableMP_fabrics_2nd_pouring"
        network_yaml_GOMP = "kuka_GOMP_pouring"

        self.kuka_class_pouring = comparison_kuka_class(results=self.results, n_runs=n_runs)
        results_pouring = self.kuka_class_pouring.KukaComparison(q_init_list, positions_obstacles_list,
                                                                   speed_obstacles_list, network_yaml, network_yaml_GOMP,
                                                                   nr_obst, goal_pos_list, goal_vel_list)

        # plot the results:
        self.kuka_class_pouring.table_results(results_pouring)
        return results_pouring

def main_pouring(render=False):
    LOAD_RESULTS = False
    nr_obst = 0
    network_yaml = "kuka_stableMP_fabrics_2nd_pouring"
    demonstrations_class = ReadDemonstrations(file_name=network_yaml)
    demonstrations_ee, goal_pos_list, q_init_list = demonstrations_class.read_demonstrations()
    q_init_list = q_init_list
    n_runs = len(q_init_list)

    kuka_class = comparisonDemonstrations()
    results = kuka_class.produce_results_pouring(q_init_list=q_init_list, goal_pos_list=goal_pos_list, nr_obst=nr_obst)
    return results

if __name__ == "__main__":
    main_pouring()