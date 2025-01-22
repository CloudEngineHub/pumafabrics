"""
This file is to generate a comparison plot between safeMP and safeMP+fabrics and theorem III.5.
"""

import numpy as np
from examples.kuka_Fabrics import example_kuka_fabrics
from examples.kuka_TamedPUMA import example_kuka_TamedPUMA
from examples.kuka_PUMA_3D_ModulationIK import example_kuka_PUMA_modulationIK
from texttable import Texttable
import latextable
import copy
import pickle
from scipy.interpolate import interp1d
from scipy import interpolate
import matplotlib.pyplot as plt

class comparison_kuka_class():
    def __init__(self, results=None, n_runs=10):
        self.cases = ["PUMA$_free$", "PUMA$_obst$",  "Occlusion-IK", "Fabrics", "FPM", "CPM"]
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

    def interpolate_positions(self, positions, dt_varying, dt_fixed, length):
        timesteps_fixed = np.linspace(0.0, (length-1)*dt_fixed, length)
        position_set = np.zeros((length, 3))
        for i in range(3):
            positions_i = np.array(positions)[:, i]
            interp_function = interp1d(dt_varying, list(positions_i), kind='linear', axis=0, fill_value="extrapolate")
            position_set[:, i] = interp_function(timesteps_fixed)
        return position_set

    def distance_to_NN(self, xee_list_PUMA, xee_list):
        """
        Note: currently distance position AND orientation!!
        """
        len_PUMA = len(xee_list_PUMA)
        if len_PUMA != len(xee_list):
            xee_list_i = self.resample_path(xee_list, len_PUMA)
        else:
            xee_list_i = xee_list
        distances = np.absolute(np.array(xee_list_i) - np.array(xee_list_PUMA))
        distance = np.mean(distances)
        distance_std = np.std(distances)
        return distances, distance, distance_std

    def resample_path_new(self, original_list, target_length):
        resampled_list = np.zeros((target_length, 3))
        original_indices = np.linspace(0, len(original_list) - 1, num=len(original_list))
        target_indices = np.linspace(0, len(original_list) - 1, num=target_length)
        for i in range(3):
            resampled_list[:, i] = np.interp(target_indices, original_indices, original_list[:, i])
        return resampled_list

    def plot_trajectories_2D(self, xee_list, x_demonstrations, xee_list_original, x_demonstrations_original, case):
        plt.figure(figsize=(6, 4))
        plt.plot(xee_list_original[:, 0], xee_list_original[:, 1], label='x_ee', color='purple')
        plt.plot(x_demonstrations_original[:, 0], x_demonstrations_original[:, 1], label='demonstrations', color='blue', linewidth=3)
        plt.plot(xee_list[:, 0], xee_list[:, 1], label='x_ee', color='purple', marker='o', linewidth=3)
        plt.plot(x_demonstrations[:, 0], x_demonstrations[:, 1], label='demonstrations', color='blue', marker='o')
        plt.plot(x_demonstrations[-1, 0], x_demonstrations[-1, 1], label="goal", color='green', marker='x', linewidth=5)
        plt.legend()
        plt.title("trajectories for" + case)
        plt.show()

    def distance_to_demonstration(self, xee_list, xee_demonstrations, xee_list_dt = 0.02, case="some_case"):
        length = 100
        x_pos_demonstrations = xee_demonstrations["x_pos"]
        len_demonstrations = len(x_pos_demonstrations)
        len_xee = len(xee_list)
        dt_demonstrations = xee_demonstrations["delta_t"]
        if len_demonstrations<len_xee:
            len_interpolate = len_demonstrations
        else:
            len_interpolate = len_xee
        demonstrations_fixed_dt =  self.interpolate_positions(x_pos_demonstrations, dt_demonstrations, xee_list_dt, len_interpolate)
        interpolated_demonstrations = self.resample_path_new(np.array(x_pos_demonstrations), len_interpolate)
        interpolated_xee_list = self.resample_path_new(np.array(xee_list), len_interpolate)
        self.plot_trajectories_2D(interpolated_xee_list, interpolated_demonstrations, np.array(xee_list), np.array(xee_demonstrations['x_pos']), case)
        distances = np.linalg.norm(np.array(xee_list[0:len_interpolate]) - np.array(interpolated_demonstrations[0:len_interpolate]), axis=1)
        distance = np.mean(distances)
        distance_std = np.std(distances)
        return distances, distance, distance_std

    def run_i(self, example_class, case:str, q_init_list:list, results_PUMA=None, positions_obstacles_list=[],
              speed_obstacles_list=[], goal_pos_list=None, goal_vel_list=None, xee_demonstrations=None):
        results_tot = copy.deepcopy(self.results)
        example_class.overwrite_defaults(params=example_class.params, render=False)
        for i_run in range(self.n_runs):
            print("case: ", case, ", run_id: ", i_run)
            if goal_pos_list is not None:
                example_class.overwrite_defaults(params=example_class.params, goal_pos=goal_pos_list[i_run])
            if goal_vel_list is not None:
                example_class.overwrite_defaults(params=example_class.params, goal_vel=goal_vel_list[i_run])
            example_class.overwrite_defaults(params=example_class.params, init_pos=q_init_list[i_run], positions_obstacles=positions_obstacles_list[i_run], speed_obstacles=speed_obstacles_list[i_run])
            example_class.construct_example()
            results_i = example_class.run_kuka_example()
            if xee_demonstrations == None:
                print("wrong distance checker activitated!!!!!!")
                if case == "Fabrics" and results_i["goal_reached"] == False:
                    print("i_run:", i_run)
                if results_PUMA is None:
                    distances, _, _ = self.distance_to_NN(results_i["xee_list"], results_i["xee_list"])
                else:
                    distances, _, _ = self.distance_to_NN(results_PUMA["xee_list"][i_run], results_i["xee_list"])
            else:
                xee_positions_list = [results_i["xee_list"][z][0:3] for z in range(len(results_i["xee_list"]))]
                distances, _, _ = self.distance_to_demonstration(xee_positions_list,
                                                                 xee_demonstrations[f"ee_state_{i_run}"],
                                                                 xee_list_dt=example_class.params["dt"],
                                                                 case=case)
            results_tot["dist_to_NN"].append(distances)
            for key in results_i:
                results_tot[key].append(results_i[key])
        return results_tot

    def KukaComparison(self, q_init_list:list, positions_obstacles_list:list, speed_obstacles_list:list,
                       network_yaml: str, network_yaml_GOMP:str, nr_obst:int, goal_pos_list=None,
                       goal_vel_list=None, xee_demonstrations=None):

        # --- run safe MP (only) example ---#
        class_PUMA = example_kuka_TamedPUMA(file_name=network_yaml)
        class_PUMA.overwrite_defaults(params=class_PUMA.params, bool_combined=False, nr_obst=0)
        results_PUMA = self.run_i(class_PUMA, case=self.cases[0], q_init_list=q_init_list, results_PUMA=None,
                                      positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                      goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list, xee_demonstrations=xee_demonstrations)

        # --- run safe MP (only) example ---#
        class_PUMA_obst = example_kuka_TamedPUMA(file_name=network_yaml)
        class_PUMA_obst.overwrite_defaults(params=class_PUMA_obst.params, bool_combined=False, nr_obst=nr_obst)
        results_PUMA_obst = self.run_i(class_PUMA_obst, case=self.cases[1], q_init_list=q_init_list, results_PUMA=None,
                                           positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                           goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list, xee_demonstrations=xee_demonstrations)

        # run the occlusion-based IK baseline ---#
        class_IK = example_kuka_PUMA_modulationIK(file_name=network_yaml_GOMP)
        class_IK.overwrite_defaults(params=class_IK.params, bool_energy_regulator=True, bool_combined=True, render=True, nr_obst=nr_obst)
        results_IK = self.run_i(class_IK, case=self.cases[2], q_init_list=q_init_list, results_PUMA=results_PUMA,
                                positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list, xee_demonstrations=xee_demonstrations)

        # --- run fabrics (only) example ---#
        class_fabrics = example_kuka_fabrics(file_name=network_yaml)
        class_fabrics.overwrite_defaults(params=class_fabrics.params, nr_obst=nr_obst)
        results_fabrics = self.run_i(class_fabrics, case=self.cases[3], q_init_list=q_init_list, results_PUMA=results_PUMA,
                                     positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                     goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list, xee_demonstrations=xee_demonstrations)

        # run safe MP + fabrics example ---#
        class_FPM = example_kuka_TamedPUMA(file_name=network_yaml)
        class_FPM.overwrite_defaults(params=class_FPM.params, bool_energy_regulator=False, bool_combined=True, nr_obst=nr_obst)
        results_FPM = self.run_i(class_FPM, case=self.cases[4], q_init_list=q_init_list, results_PUMA=results_PUMA,
                                              positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                              goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list, xee_demonstrations=xee_demonstrations)

        # run theorem III.5 ---#
        class_CPM = example_kuka_TamedPUMA(file_name=network_yaml)
        class_CPM.overwrite_defaults(params=class_CPM.params, bool_energy_regulator=True, bool_combined=True, nr_obst=nr_obst)
        results_CPM = self.run_i(class_CPM, case=self.cases[5], q_init_list=q_init_list, results_PUMA=results_PUMA,
                                positions_obstacles_list=positions_obstacles_list, speed_obstacles_list=speed_obstacles_list,
                                goal_pos_list=goal_pos_list, goal_vel_list=goal_vel_list, xee_demonstrations=xee_demonstrations)

        self.results = {self.cases[0]: results_PUMA, self.cases[1]: results_PUMA_obst, self.cases[2]: results_IK, self.cases[3]: results_fabrics, self.cases[4]: results_FPM, self.cases[5]: results_CPM}
        with open("../pumafabrics/puma_adapted/results/data_files/simulation_kuka_2nd"+network_yaml+".pkl", 'wb') as f:
            pickle.dump(self.results, f)
        return self.results

    def KukaComparisonLoad(self):
        file_i = open(f"../pumafabrics/puma_adapted/results/data_files/simulation_kuka_2nd"+network_yaml+".pkl", 'rb')
        results= pickle.load(file_i)
        return results

    def table_results(self, results):
        # --- create and plot table --- #
        rows = []
        title_row = [' ','Success-Rate', 'Time-to-Success [s]', "Min Clearance [m]", "Computation time [ms]", 'Path difference to PUMA'] #, "Input difference to PUMA"]
        nr_column = len(title_row)
        n_runs = len(results[self.cases[0]]["goal_reached"])
        rows.append(title_row)
        for case in self.cases:
            if results[case]["min_distance"] == [1000]*len(results[case]["min_distance"]):
                min_clearance_str = "-"
            else:
                min_clearance_str = str(np.round(np.nanmean(results[case]["min_distance"]), decimals=2)) + " $\pm$ " + str(
                    np.round(np.nanstd(results[case]["min_distance"]), decimals=2))

            rows.append([case,
                         str(np.round(np.sum(results[case]["goal_reached"]) / n_runs, decimals=1)),
                         str(np.round(np.nanmean(results[case]["time_to_goal"]), decimals=4)) + " $\pm$ " + str(np.round(np.nanstd(results[case]["time_to_goal"]), decimals=4)),
                         min_clearance_str,
                         str(np.round(np.nanmean(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)),
                         str(np.round(np.nanmedian(np.concatenate(results[case]["dist_to_NN"], axis=0)), decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["dist_to_NN"], axis=0)), decimals=2)),
                         ])
        table = Texttable()
        table.set_cols_align(["c"] * nr_column)
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(rows)
        print('\nTexttable Latex:')
        print(latextable.draw_latex(table))
        print("results[case][goal_reached]:", results["GF"]["goal_reached"])

if __name__ == "__main__":
    LOAD_RESULTS = False
    nr_obst = 2
    q_init_list = [
        # with goal changing:
        np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
        np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
        np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)),
        np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
        np.array((0.07, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
        # others:
        np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
        np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
        np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)),
        np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
        np.array((0.07, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
        np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
        np.array((0.51, 0.67, -0.17, -1.73, 0.25, -0.86, -0.11)),
        np.array((0.91, 0.79, -0.22, -1.33, 1.20, -1.76, -1.06)),
        np.array((0.83, 0.53, -0.11, -0.95, 1.05, -1.24, -1.45)),
        np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
    ]
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
    goal_pos_list = [
        # #changing goal pose:
        [0.58, -0.014, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.7, -0.214, 0.315],
        [0.7, -0.214, 0.115],
        # others:
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
        [0.58, -0.214, 0.115],
    ]
    goal_vel_list = [
        [0., 0., 0.] for _ in range(len(q_init_list))
    ]
    goal_vel_list[0] = [0., 0., 0.]
    if len(q_init_list) > 1:
        goal_vel_list[1] = [-0.01, 0., 0.]
    if len(q_init_list) > 2:
        goal_vel_list[2] = [-0.01, 0., 0.0]
    n_runs = len(q_init_list)
    network_yaml = "kuka_TamedPUMA_tomato"
    network_yaml_GOMP = "kuka_ModulationIK_tomato"
    kuka_class = comparison_kuka_class(n_runs=n_runs)
    if LOAD_RESULTS == False:
        # Regenerate the results:
        results = kuka_class.KukaComparison(q_init_list, positions_obstacles_list, speed_obstacles_list, network_yaml, network_yaml_GOMP, nr_obst, goal_pos_list, goal_vel_list)
    else:
        results = kuka_class.KukaComparisonLoad()

    #plot the results:
    kuka_class.table_results(results)

