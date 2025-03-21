import numpy as np
from texttable import Texttable
import latextable
from evaluations.kuka_comparison_2nd_tomato import comparison_kuka_class
import copy
import pickle

class obtain_total_results():
    def __init__(self):
        self.cases = ["PUMA$_free$", "PUMA$_obst$",  "Occlusion-IK", "Fabrics", "FPM", "CPM"]
        self.results_tot = {"PUMA$_free$": {"solver_times":[]}, "PUMA$_obst$": {"solver_times":[]},  "Occlusion-IK": {"solver_times":[]}, "Fabrics": {"solver_times":[]}, "FPM": {"solver_times":[]}, "CPM": {"solver_times":[]}}
        self.results = {"min_distance": [], "collision": [], "goal_reached": [], "time_to_goal": [], "xee_list": [], "qdot_diff_list": [],
           "dist_to_NN": [],  "vel_to_NN": [], "solver_times": [], "solver_time": [], "solver_time_std": []}
        self.kuka_class_tomato = comparison_kuka_class()
        self.kuka_class_pouring= comparison_kuka_class()

    def append_results(self, results_0, results_1, results_0_no_obst, results_1_no_obst):
        results_tot = copy.deepcopy(self.results_tot)
        for case in self.cases:
            for key in results_0[case].keys():
                print("key:", key)
                if key == "dist_to_NN" or key == "vel_to_NN":
                    results_tot[case][key] = results_0_no_obst[case][key] + results_1_no_obst[case][key]
                elif key == "solver_times":
                    results_tot[case][key].extend(results_0[case][key])
                    results_tot[case][key].extend(results_1[case][key])
                else:
                    results_tot[case][key] = results_0[case][key] + results_1[case][key]
        return results_tot

    def table_results_individual(self, results_tomato, results_pouring, results_tomato_no_obst, results_pouring_no_obst):
        print("%%%%%%%%%%%%%%%%%% results tomato with 2 obstacles: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        self.kuka_class_tomato.table_results(results_tomato)
        print("%%%%%%%%%%%%%%%%%% results pouring with 2 obstacles: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        self.kuka_class_pouring.table_results(results_pouring)
        print("%%%%%%%%%%%%%%%%%% results tomato with 0 obstacles: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        self.kuka_class_tomato.table_results(results_tomato_no_obst)
        print("%%%%%%%%%%%%%%%%%% results pouring with 0 obstacles: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        self.kuka_class_pouring.table_results(results_pouring_no_obst)

    def table_results_total(self, results):

        with open("results_evaluations/simulation_kuka_results_ALL.pkl", 'wb') as f:
            pickle.dump(results, f)

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
                         str(np.round(np.sum(results[case]["goal_reached"]) / n_runs, decimals=2)),
                         str(np.round(np.nanmean(results[case]["time_to_goal"]), decimals=4)) + " $\pm$ " + str(np.round(np.nanstd(results[case]["time_to_goal"]), decimals=4)),
                         min_clearance_str,
                         str(np.round(np.nanmean(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["solver_times"], axis=0)), decimals=6)),
                         str(np.round(np.nanmean(np.concatenate(results[case]["dist_to_NN"], axis=0)), decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["dist_to_NN"], axis=0)), decimals=2)),
                         ])
        table = Texttable()
        table.set_cols_align(["c"] * nr_column)
        table.set_deco(Texttable.HEADER | Texttable.VLINES)
        table.add_rows(rows)
        print('\nTexttable Latex:')
        print(latextable.draw_latex(table))
        print("results[case][goal_reached]:", results["Fabrics"]["goal_reached"])

    def produce_results_tomato(self, results=None, nr_obst=2):
        LOAD_RESULTS = False
        nr_obst = nr_obst
        q_init_list = [
            np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
            np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
            np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)),
            np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
            np.array((0.07, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
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
            [[0.5, 0., 0.55], [0.5, 0., 10.1]],
            [[0.55, 0.15, 0.05], [0.55, 0.15, 0.23]],
            [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],
            [[0.45, 0.02, 0.28], [0.7, 0.02, 0.28]],
            [[0.5, -0.0, 0.5], [0.3, -0.1, 10.5]],
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
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., 0., 0.]],
        ]
        goal_pos_list = [
            [0.58, -0.014, 0.115],
            [0.58, -0.214, 0.115],
            [0.58, -0.214, 0.115],
            [0.7, -0.214, 0.315],
            [0.7, -0.214, 0.115],
            [0.58, -0.214, 0.115],
            [0.58, -0.214, 0.115],
            [0.58, -0.214, 0.115],
            [0.58, -0.214, 0.115],
            [0.58, -0.214, 0.115],
            [0.58, -0.25,  0.115],
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

        self.kuka_class_tomato = comparison_kuka_class(results=self.results, n_runs=n_runs)
        if LOAD_RESULTS == False:
            # Regenerate the results:
            results_tomato = self.kuka_class_tomato.KukaComparison(q_init_list, positions_obstacles_list, speed_obstacles_list, network_yaml, network_yaml_GOMP, nr_obst, goal_pos_list, goal_vel_list)
        else:
            results_tomato = self.kuka_class_tomato.KukaComparisonLoad()

        #plot the results:
        self.kuka_class_tomato.table_results(results_tomato)
        # self.results = results_tomato
        return results_tomato


    def produce_results_pouring(self, results=None, nr_obst=2):
        LOAD_RESULTS = False
        nr_obst = nr_obst
        q_init_list = [
            np.array((1.81414, -1.77245, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 0
            np.array((1.81414, -1.77245, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 1
            np.array((-0.06968, -2.0944, 1.25021, 1.91157, -0.902882, 0.387756, 1.26118)),  # 2
            np.array((0.487286, -2.0944, 1.46101, 1.53229, -0.980283, 0.194411, 1.53735)),  # 3
            np.array((0.674393, -1.78043, 1.75829, 1.0226, 0.356607, -0.0418928, 0.283865)),  # 4
            np.array((1.71414, -1.61, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 5
            np.array((1.7, -1.5, 1.5, 1.47711, 2.75051, -1.18862, -1.2)),  # 6
            np.array((-0.06968, -2.0944, 1.25021, 1.91157, -0.902882, 0.387756, 1.26118)),  # 7
            np.array((0.487286, -2.0944, 1.46101, 1.53229, -0.980283, 0.194411, 1.53735)),  # 8
            np.array((0.674393, -1.78043, 1.75829, 1.0226, 0.356607, -0.0418928, 0.283865)),  # 9
            np.array((1.28421, -1.08275, 0.709752, 1.22488, 2.78079, -0.549531, -0.868621)),  # 10
            np.array((0.164684, -1.8114, 1.2818, 2.05525, 0.378834, -0.0280146, 0.340511)),  # 11
            np.array((1.08108, -1.51439, 0.755646, 1.52847, -1.54951, 0.874368, 2.71138)),  # 12
            np.array((-1.41497, 1.23653, 2.93949, 1.60902, 2.35079, -1.53339, -0.231835)),  # 13
            np.array((1.31414, -1.77245, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 14
        ]
        positions_obstacles_list = [
            [[0., -0.6, 0.2], [0.5, 0., 10.1]],  # 0
            [[0.0, -0.6, 0.2], [0.5, 0.2, 10.4]],  # 1
            [[-0.20, -0.72, 0.22], [0.24, 0.45, 10.2]],  # 2
            [[-0.1, -0.65, 0.4], [0.6, 0.02, 10.2]],  # 3
            [[-0.1, -0.5, 0.22], [0.3, -0.1, 10.5]],  # 4
            [[0.1, -0.56, 0.3], [0.5, 0., 10.1]],  # 5
            [[0.15, -0.6, 0.3], [0.5, 0.15, 0.2]],  # 6
            [[-0.1, -0.72, 0.22], [0.6, 0.02, 10.2]],  # 7
            [[-0.1, -0.72, 0.22], [0.3, -0.1, 10.5]],  # 8
            [[-0.1, -0.72, 0.22], [0.5, 0.2, 10.25]],  # 9
            [[0.03, -0.6, 0.15], [0.5, 0.2, 10.4]],  # 10
            [[0.0, -0.6, 0.10], [0.5, 0.2, 10.4]],  # 11
            [[0., -0.72, 0.1], [0.5, 0.2, 10.4]],  # 12
            [[0.0, -0.72, 0.10], [0.5, 0.2, 10.4]],  # 13
            [[0.0, -0.6, 0.10], [0.5, 0.2, 10.4]],  # 14
        ]
        speed_obstacles_list = [
            [[0.03, 0., 0.], [0., 0., 0.]],  # 0
            [[0.03, 0., 0.], [0., 0., 0.]],  # 1
            [[0.02, 0., 0.], [0., 0., 0.]],  # 2
            [[0.03, 0., 0.], [0., 0., 0.]],  # 3
            [[0.05, 0., 0.], [0., 0., 0.]],  # 4
            [[0., 0., 0.], [0., 0., 0.]],  # 5
            [[0., 0., 0.], [0., 0., 0.]],  # 6
            [[0.05, 0., 0.], [0., 0., 0.]],  # 7
            [[0.04, 0., 0.], [0., 0., 0.]],  # 8
            [[0.05, 0., 0.], [0., 0., 0.]],  # 9
            [[0., 0., 0.], [0., 0., 0.]],  # 10
            [[0., 0., 0.], [0., 0., 0.]],  # 11
            [[0.01, 0., 0.], [0., 0., 0.]],  # 12
            [[0., 0., 0.], [0., 0., 0.]],  # 13
            [[0., 0., 0.], [0., 0., 0.]],  # 14
        ]
        goal_pos_list = [
            [-0.20, -0.72, 0.15], #0
            [-0.20, -0.72, 0.15], #1
            [-0.20, -0.72, 0.22], #2
            [-0.20, -0.62, 0.6], #3
            [-0.20, -0.62, 0.22], #4
            [-0.09486833, -0.72446137, 0.22143809], #5
            [-0.09486833, -0.72446137, 0.22143809], #6
            [-0.09486833, -0.72446137, 0.22143809], #7
            [-0.09486833, -0.72446137, 0.22143809], #8
            [-0.09486833, -0.65446137, 0.22143809], #9
            [-0.09486833, -0.72446137, 0.22143809], #10
            [-0.09486833, -0.72446137, 0.22143809], #11
            [-0.09486833, -0.72446137, 0.22143809], #12
            [-0.09486833, -0.72446137, 0.22143809], #13
            [-0.09486833, -0.72446137, 0.22143809], #14
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
        network_yaml = "kuka_TamedPUMA_pouring"
        network_yaml_GOMP = "kuka_ModulationIK_pouring"

        self.kuka_class_pouring = comparison_kuka_class(results=self.results, n_runs=n_runs)
        if LOAD_RESULTS == False:
            # Regenerate the results:
            results_pouring = self.kuka_class_pouring.KukaComparison(q_init_list, positions_obstacles_list, speed_obstacles_list, network_yaml, network_yaml_GOMP,
                                                                     nr_obst, goal_pos_list, goal_vel_list)
        else:
            results_pouring = self.kuka_class_pouring.KukaComparisonLoad()

        #plot the results:
        self.kuka_class_pouring.table_results(results_pouring)
        return results_pouring

if __name__ == "__main__":
    LOAD_RESULTS=True
    total_results = obtain_total_results()
    if LOAD_RESULTS == False:
        results_tomato = total_results.produce_results_tomato()
        results_tomato_no_obst = total_results.produce_results_tomato(nr_obst=0)
        results_pouring = total_results.produce_results_pouring()
        results_pouring_no_obst = total_results.produce_results_pouring(nr_obst=0)
    else:
        file_i = open('results_evaluations/simulation_kuka_results_ALL_tomato.pkl', 'rb')
        results_tomato = pickle.load(file_i)
        file_i = open('results_evaluations/simulation_kuka_results_ALL_tomato_no_obst.pkl', 'rb')
        results_tomato_no_obst = pickle.load(file_i)
        file_i = open('results_evaluations/simulation_kuka_results_ALL_pouring.pkl', 'rb')
        results_pouring = pickle.load(file_i)
        file_i = open('results_evaluations/simulation_kuka_results_ALL_pouring_no_obst.pkl', 'rb')
        results_pouring_no_obst = pickle.load(file_i)

    #save latest results
    with open("results_evaluations/simulation_kuka_results_ALL_tomato.pkl", 'wb') as f:
        pickle.dump(results_tomato, f)
    with open("results_evaluations/simulation_kuka_results_ALL_tomato_no_obst.pkl", 'wb') as f:
        pickle.dump(results_tomato_no_obst, f)
    with open("results_evaluations/simulation_kuka_results_ALL_pouring.pkl", 'wb') as f:
        pickle.dump(results_pouring, f)
    with open("results_evaluations/simulation_kuka_results_ALL_pouring_no_obst.pkl", 'wb') as f:
        pickle.dump(results_pouring_no_obst, f)
    results_tot = total_results.append_results(results_tomato, results_pouring, results_tomato_no_obst, results_pouring_no_obst)

    # # plot table of individual and collective results:
    total_results.table_results_individual(results_tomato, results_pouring, results_tomato_no_obst, results_pouring_no_obst)
    total_results.table_results_total(results_tot)





