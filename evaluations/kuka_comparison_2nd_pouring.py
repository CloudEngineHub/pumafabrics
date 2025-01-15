"""
This file is to generate a comparison plot between safeMP and safeMP+fabrics and theorem III.5.
"""

import numpy as np
from examples.kuka_Fabrics import example_kuka_fabrics
# from kuka_stableMP_R3S3 import example_kuka_stableMP_R3S3
# from kuka_stableMP_fabrics_theoremIII_5_2nd import example_kuka_stableMP_fabrics
from examples.kuka_TamedPUMA import example_kuka_TamedPUMA
from examples.kuka_PUMA_3D_ModulationIK import example_kuka_PUMA_modulationIK
from texttable import Texttable
import latextable
import copy
import pickle
from scipy import interpolate


class comparison_kuka_class():
    def __init__(self, results=None, n_runs=10):
        self.cases = ["PUMA$_free$", "PUMA$_obst$",  "Occlusion-IK", "GF", "GM", "CM"]
        if results is None:
            self.results = {"min_distance": [], "collision": [], "goal_reached": [], "time_to_goal": [], "xee_list": [],
                   "qdot_diff_list": [],
                   "dist_to_NN": [], "vel_to_NN": [], "solver_times": [], "solver_time": [], "solver_time_std": []}
        else:
            self.results = results
        self.n_runs = n_runs

    def resample_path(self, path, num_points):
        # Convert the path to a numpy array for easier manipulation
        path = np.array(path)

        # Create an array of the cumulative distance along the path
        distance = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0)

        # Create an interpolating function for the path
        alpha = np.linspace(distance.min(), distance.max(), num_points)
        interpolator = interpolate.interp1d(distance, path, axis=0)

        # Use the interpolator to generate the resampled path
        resampled_path = interpolator(alpha)

        return resampled_path.tolist()

    def distance_to_NN(self, xee_list_SMP, xee_list):
        """
        Note: currently distance position AND orientation!!
        """
        len_SMP = len(xee_list_SMP)
        if len_SMP != len(xee_list):
            xee_list_i = self.resample_path(xee_list, len_SMP)
        else:
            xee_list_i = xee_list
        distances = np.absolute(np.array(xee_list_i) - np.array(xee_list_SMP))
        distance = np.mean(distances)
        distance_std = np.std(distances)
        return distances, distance, distance_std

    def run_i(self, example_class, case:str, q_init_list:list, results_stableMP=None, positions_obstacles_list=[], speed_obstacles_list=[], goal_pos_list=None, goal_vel_list=None):
        results_tot = copy.deepcopy(self.results)
        example_class.overwrite_defaults(render=False)
        for i_run in range(self.n_runs):
            if goal_pos_list is not None:
                example_class.overwrite_defaults(goal_pos=goal_pos_list[i_run])
            if goal_vel_list is not None:
                example_class.overwrite_defaults(goal_vel=goal_vel_list[i_run])
            example_class.overwrite_defaults(init_pos=q_init_list[i_run], positions_obstacles=positions_obstacles_list[i_run], speed_obstacles=speed_obstacles_list[i_run])
            example_class.construct_example()
            results_i = example_class.run_kuka_example()
            if case == "GF" and results_i["goal_reached"] == False:
                print("i_run:", i_run)
            if results_stableMP is None:
                distances, _, _ = self.distance_to_NN(results_i["xee_list"], results_i["xee_list"])
            else:
                distances, _, _ = self.distance_to_NN(results_stableMP["xee_list"][i_run], results_i["xee_list"])
            results_tot["dist_to_NN"].append(distances)
            for key in results_i:
                results_tot[key].append(results_i[key])
        return results_tot

    def KukaComparison(self, q_init_list:list, positions_obstacles_list:list, speed_obstacles_list:list, network_yaml: str, network_yaml_GOMP:str, nr_obst:int, goal_pos_list=None, goal_vel_list=None):

        # --- run safe MP (only) example ---#
        class_SMP = example_kuka_TamedPUMA(file_name=network_yaml)
        class_SMP.overwrite_defaults(bool_combined=False, nr_obst=0)
        results_stableMP = self.run_i(class_SMP, case=self.cases[0], q_init_list=q_init_list, results_stableMP=None,
                                      positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                      goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list)

        # --- run safe MP (only) example ---#
        class_SMP_obst = example_kuka_TamedPUMA(file_name=network_yaml)
        class_SMP_obst.overwrite_defaults(bool_combined=False, nr_obst=nr_obst)
        results_stableMP_obst = self.run_i(class_SMP_obst, case=self.cases[1], q_init_list=q_init_list, results_stableMP=None,
                                           positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                           goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list)

        # run the occlusion-based IK baseline ---#
        class_IK = example_kuka_PUMA_modulationIK(file_name=network_yaml_GOMP)
        class_IK.overwrite_defaults(bool_energy_regulator=True, bool_combined=True, render=False, nr_obst=nr_obst)
        results_IK = self.run_i(class_IK, case=self.cases[2], q_init_list=q_init_list, results_stableMP=results_stableMP,
                                positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list)

        # --- run fabrics (only) example ---#
        class_fabrics = example_kuka_fabrics(file_name=network_yaml)
        class_fabrics.overwrite_defaults(nr_obst=nr_obst)
        results_fabrics = self.run_i(class_fabrics, case=self.cases[3], q_init_list=q_init_list, results_stableMP=results_stableMP,
                                     positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                     goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list)

        # run safe MP + fabrics example ---#
        class_GM = example_kuka_TamedPUMA(file_name=network_yaml)
        class_GM.overwrite_defaults(bool_energy_regulator=False, bool_combined=True, nr_obst=nr_obst)
        results_stableMP_fabrics = self.run_i(class_GM, case=self.cases[4], q_init_list=q_init_list, results_stableMP=results_stableMP,
                                              positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                              goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list)

        # run theorem III.5 ---#
        class_CM = example_kuka_TamedPUMA(file_name=network_yaml)
        class_CM.overwrite_defaults(bool_energy_regulator=True, bool_combined=True, nr_obst=nr_obst)
        results_CM = self.run_i(class_CM, case=self.cases[5], q_init_list=q_init_list, results_stableMP=results_stableMP,
                                positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list)

        self.results = {self.cases[0]: results_stableMP, self.cases[1]: results_stableMP_obst, self.cases[2]: results_IK, self.cases[3]: results_fabrics, self.cases[4]: results_stableMP_fabrics, self.cases[5]: results_CM}
        with open("results/data_files/simulation_kuka_2nd.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        return self.results

    def KukaComparisonLoad(self):
        results = {}
        file_i = open(f"results/data_files/simulation_kuka_2nd9.pkl", 'rb')
        results= pickle.load(file_i)
        return results

    def table_results(self, results):
        # --- create and plot table --- #
        rows = []
        title_row = [' ','Success-Rate', 'Time-to-Success [s]', "Min Clearance [m]", "Computation time [ms]", 'Path difference to PUMA', "Input difference to PUMA"]
        nr_column = len(title_row)
        n_runs = len(results[self.cases[0]]["goal_reached"])
        rows.append(title_row)
        for case in self.cases:
            if case == "SMP":
                collision_episodes_rate_str = "-"
            else:
                collision_episodes_rate_str = np.round(np.sum(results[case]["collision"]) / len(results[case]["collision"]), decimals=8)
            if results[case]["min_distance"] == [1000]*len(results[case]["min_distance"]):
                min_clearance_str = "-"
            else:
                min_clearance_str = str(np.round(np.nanmean(results[case]["min_distance"]), decimals=2)) + " $\pm$ " + str(
                    np.round(np.nanstd(results[case]["min_distance"]), decimals=2))

            rows.append([case,
                         str(np.round(np.sum(results[case]["goal_reached"]) / n_runs, decimals=1)), #+ "+-" + str(np.round(np.nanstd(results[case]["goal_reached"]), decimals=4)),
                         str(np.round(np.nanmean(results[case]["time_to_goal"]), decimals=4)) + " $\pm$ " + str(np.round(np.nanstd(results[case]["time_to_goal"]), decimals=4)),
                         # collision_episodes_rate_str,
                         min_clearance_str,
                         str(np.round(np.nanmean(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)),
                         str(np.round(np.nanmean(np.concatenate(results[case]["dist_to_NN"], axis=0)), decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["dist_to_NN"], axis=0)), decimals=2)),
                         str(np.round(np.nanmean(np.concatenate(results[case]["qdot_diff_list"], axis=0)), decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["qdot_diff_list"], axis=0)), decimals=2)),
                         ])
        table = Texttable()
        table.set_cols_align(["c"] * nr_column)
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(rows)
        print('\nTexttable Latex:')
        print(latextable.draw_latex(table)) #, caption="\small{Statistics for 50 simulated scenarios of our proposed methods \ac{gm} and \ac{cm} compared to 50 scenarios of \ac{gf} and \ac{smp}}"))
        print("results[case][goal_reached]:", results["GF"]["goal_reached"])

if __name__ == "__main__":
    LOAD_RESULTS = False
    nr_obst = 2
    q_init_list = [
        # changing goal pos
        np.array((1.81414, -1.77245, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 0
        np.array((1.81414, -1.77245, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 1
        np.array((-0.06968, -2.0944, 1.25021, 1.91157, -0.902882, 0.387756, 1.26118)),  # 2
        np.array((0.487286, -2.0944, 1.46101, 1.53229, -0.980283, 0.194411, 1.53735)),  # 3
        np.array((0.674393, -1.78043, 1.75829, 1.0226, 0.356607, -0.0418928, 0.283865)),  # 4
        # # others:
        np.array((1.71414, -1.61, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 0, 5
        np.array((1.7, -1.5, 1.5, 1.47711, 2.75051, -1.18862, -1.2)),  # 1, 6
        np.array((-0.06968, -2.0944, 1.25021, 1.91157, -0.902882, 0.387756, 1.26118)),  # 2, 7
        np.array((0.487286, -2.0944, 1.46101, 1.53229, -0.980283, 0.194411, 1.53735)),  # 3, 8
        np.array((0.674393, -1.78043, 1.75829, 1.0226, 0.356607, -0.0418928, 0.283865)),  # 4, 9
        np.array((1.28421, -1.08275, 0.709752, 1.22488, 2.78079, -0.549531, -0.868621)),  # 5, 10
        np.array((0.164684, -1.8114, 1.2818, 2.05525, 0.378834, -0.0280146, 0.340511)),  # 6, 11
        np.array((1.08108, -1.51439, 0.755646, 1.52847, -1.54951, 0.874368, 2.71138)),  # 7, 12
        np.array((-1.41497, 1.23653, 2.93949, 1.60902, 2.35079, -1.53339, -0.231835)),  # 8, 13
        np.array((1.31414, -1.77245, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 9, 14
    ]
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
    goal_pos_list = [
        # # #changing goal pose:
        [-0.20, -0.72, 0.15],
        [-0.20, -0.72, 0.15],
        [-0.20, -0.72, 0.22],
        [-0.20, -0.62, 0.6],
        [-0.20, -0.62, 0.22],
        # # others:
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
        [-0.09486833, -0.72446137, 0.22143809],
    ]
    goal_vel_list = [
        [0., 0., 0.] for _ in range(len(q_init_list))
    ]
    goal_vel_list[0] = [0., 0., 0.]
    if len(q_init_list) > 1:
        goal_vel_list[1] = [-0.001, 0., 0.]
    if len(q_init_list) > 2:
        goal_vel_list[2] = [-0.01, 0., 0.0]
    n_runs = len(q_init_list)
    network_yaml = "kuka_stableMP_fabrics_2nd_pouring"
    network_yaml_GOMP = "kuka_GOMP_pouring"
    kuka_class = comparison_kuka_class(n_runs=n_runs)
    if LOAD_RESULTS == False:
        # Regenerate the results:
        results = kuka_class.KukaComparison(q_init_list, positions_obstacles_list, speed_obstacles_list, network_yaml, network_yaml_GOMP, nr_obst, goal_pos_list, goal_vel_list)
    else:
        results = kuka_class.KukaComparisonLoad()

    #plot the results:
    kuka_class.table_results(results)

    #plot results in figure:
    # plots_class = plotting_functions()
    # plots_class.comparison_plot_iiwa(results=results, cases=cases)

