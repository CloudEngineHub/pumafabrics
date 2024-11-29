"""
This file is to generate a comparison plot between fabrics, PUMA and TamedPUMA (FPM) and TamedPUMA (CPM) and ModulationIK.
"""
import os
import numpy as np
# from examples.kuka_Fabrics import example_kuka_fabrics
# from examples.kuka_TamedPUMA import example_kuka_TamedPUMA
# from examples.kuka_PUMA_3D_ModulationIK import example_kuka_PUMA_modulationIK
import pickle
from kuka_comparison_2nd import comparison_kuka_class
# from evaluations.comparison_general import ComparisonGeneral
# from pumafabrics.tamed_puma.utils.record_data import RecordData, EvaluationDataStructure
from src.functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka
import yaml

class ReadDemonstrations():
    def __init__(self, file_name = "kuka_stableMP_fabrics_2nd"):
        with open("config/"+file_name+".yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0],
                                          robot_name=self.params["robot_name"])

    def read_demonstrations(self):
        goal_pos_list = []
        q_init_list = []
        current_directory = os.getcwd()
        demonstrations_joints = {f"joints_state_{i}": [] for i in range(10)}
        for i in range(10):
            file_demo_joints = open(
                current_directory + f"/datasets/kuka/pick_tomato_31may_joints/joints_state_{i}.pk",
                'rb')
            demonstration_joints = pickle.load(file_demo_joints)
            demonstrations_joints[f"joints_state_{i}"] = demonstration_joints
            q_init_list.append(np.array(demonstration_joints["q"][0]))
        print("q_init_list: ", q_init_list)

        demonstrations_ee = {f'ee_state_{i}': {"x_pos": [], "delta_t":[] }for i in range(10)}
        for i in range(10):
            demonstration_ee = self.get_list_ee_via_fk(q_list=demonstrations_joints[f"joints_state_{i}"]["q"])
            demonstrations_ee[f"ee_state_{i}"]["x_pos"] = demonstration_ee
            demonstrations_ee[f"ee_state_{i}"]["delta_t"] = self.timesteps_from_dt(demonstrations_joints[f"joints_state_{i}"]["delta_t"])
            if i == 0:
                print("start ee position: ", demonstration_ee[0])
            goal_pos_list.append([float(demonstration_ee[-1][i]) for i in range(3)])
            if i == 0:
                print("goal position: ", demonstration_ee[-1])
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

    def produce_results_tomato(self, q_init_list, goal_pos_list, nr_obst=0, xee_demonstrations=None):
        LOAD_RESULTS = False
        nr_obst = nr_obst
        positions_obstacles_list = [
            # with goal changing:
            [[0.5, 0., 0.55], [0.5, 0., 10.1]],
            [[0.5, 0.15, 0.05], [0.5, 0.15, 0.2]],
            [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],
            [[0.45, 0.02, 0.2], [0.6, 0.02, 0.2]],
            [[0.5, -0.0, 0.5], [0.3, -0.1, 10.5]],
            # others:
            [[0.5, 0., 0.55], [0.5, 0., 10.1]],
            [[0.5, 0.15, 0.05], [0.5, 0.15, 0.2]],
            [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],
            [[0.45, 0.02, 0.2], [0.6, 0.02, 0.2]],
            [[0.5, -0.0, 0.5], [0.3, -0.1, 10.5]],
            [[0.5, -0.05, 0.3], [0.5, 0.2, 10.25]],
            [[0.5, -0.0, 0.2], [0.5, 0.2, 10.4]],
            [[0.5, -0.0, 0.28], [0.5, 0.2, 10.4]],
            [[0.5, 0.25, 0.55], [0.5, 0.2, 10.4]],
            [[0.5, 0.1, 0.45], [0.5, 0.2, 10.4]],
        ]
        speed_obstacles_list = [
            # with goal changing:
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            # others:
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
        ]
        goal_vel_list = [
            [0., 0., 0.] for _ in range(len(q_init_list))
        ]
        n_runs = len(q_init_list)
        network_yaml = "kuka_stableMP_fabrics_2nd"
        network_yaml_GOMP = "kuka_GOMP"

        self.kuka_class_tomato = comparison_kuka_class(results=self.results, n_runs=n_runs)
        results_tomato = self.kuka_class_tomato.KukaComparison(q_init_list, positions_obstacles_list,
                                                                   speed_obstacles_list, network_yaml, network_yaml_GOMP,
                                                                   nr_obst, goal_pos_list, goal_vel_list, xee_demonstrations=xee_demonstrations)

        # plot the results:
        self.kuka_class_tomato.table_results(results_tomato)
        # self.results = results_tomato
        return results_tomato

def main_tomato(render=False):
    LOAD_RESULTS = False
    nr_obst = 0
    network_yaml = "kuka_stableMP_fabrics_2nd"
    demonstrations_class = ReadDemonstrations(file_name=network_yaml)
    demonstrations_ee, goal_pos_list, q_init_list = demonstrations_class.read_demonstrations()
    q_init_list = q_init_list
    n_runs = len(q_init_list)

    kuka_class = comparisonDemonstrations()
    results = kuka_class.produce_results_tomato(q_init_list=q_init_list, goal_pos_list=goal_pos_list, nr_obst=nr_obst,
                                                xee_demonstrations=demonstrations_ee)
    return results

if __name__ == "__main__":
    main_tomato()