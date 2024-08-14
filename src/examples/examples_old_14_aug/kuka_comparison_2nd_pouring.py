"""
This file is to generate a comparison plot between safeMP and safeMP+fabrics and theorem III.5.
"""

import numpy as np
from functions_stableMP_fabrics.environments import trial_environments
from kuka_fabrics_comparison import example_kuka_fabrics
# from kuka_stableMP_R3S3 import example_kuka_stableMP_R3S3
# from kuka_stableMP_fabrics_theoremIII_5_2nd import example_kuka_stableMP_fabrics
from kuka_stableMP_fabrics_2nd import example_kuka_stableMP_fabrics
from texttable import Texttable
import latextable
import copy
import pickle
from scipy import interpolate
from functions_stableMP_fabrics.plotting_functions import plotting_functions

LOAD_RESULTS = False
cases = ["PUMA$_free$", "PUMA$_obst$", "GF", "GM", "CM"]
results = {"min_distance": [], "collision": [], "goal_reached": [], "time_to_goal": [], "xee_list": [], "qdot_diff_list": [],
           "dist_to_NN": [],  "vel_to_NN": [], "solver_times": [], "solver_time": [], "solver_time_std": []}
q_init_list = [
    np.array((-1.25068, 2.0944, 1.61353, 1.55983, 0.561357, 1.32142, -2.17296)),  # 0
    np.array((-0.0299795, -2.0944, 1.20398, 1.93522, -0.956052, 0.702318, 1.38504)),  # 1
    np.array((-0.06968, -2.0944, 1.25021, 1.91157, -0.902882, 0.387756, 1.26118)),  # 2
    np.array((0.487286, -2.0944, 1.46101, 1.53229, -0.980283, 0.194411, 1.53735)),  # 3
    np.array((0.674393, -1.78043, 1.75829, 1.0226, 0.356607, -0.0418928, 0.283865)),  # 4
    np.array((1.28421, -1.08275, 0.709752, 1.22488, 2.78079, -0.549531, -0.868621)),  # 5
    np.array((0.164684, -1.8114, 1.2818, 2.05525, 0.378834, -0.0280146, 0.340511)),  # 6
    np.array((1.08108, -1.51439, 0.755646, 1.52847, -1.54951, 0.874368, 2.71138)),  # 7
    # np.array((0.0670587, -1.17035, 1.30989, 1.64705, 1.74562, -0.350649, -0.368835)), #8
    np.array((-1.41497, 1.23653, 2.93949, 1.60902, 2.35079, -1.53339, -0.231835)),  # 9
    np.array((1.31414, -1.77245, 1.18276, 1.47711, 2.75051, -1.18862, -1.57065)),  # 10
    # np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)), #11
    # np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
    # np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
    # np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)),
    # np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
    # np.array((0.07, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
    # np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
    # np.array((0.51, 0.67, -0.17, -1.73, 0.25, -0.86, -0.11)),
    # np.array((0.91, 0.79, -0.22, -1.33, 1.20, -1.76, -1.06)),
    # np.array((0.83, 0.53, -0.11, -0.95, 1.05, -1.24, -1.45)),
    # np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
]
# positions_obstacles_list = [
#     [[0.0, -0.72, 0.10], [0.5, 0.2, 10.4]],  # 7 :done
#     # [[0.1, -0.7, 0.3], [0.5, 0.2, 10.4]],  # 8: done, dynamic
#     # [[0., -0.72, 0.1], [0.5, 0.2, 10.4]],  # 9: done
#     [[0.0, -0.72, 0.10], [0.5, 0.2, 10.4]],  # 10: done
#     [[0.5, 0.0, 0.40], [0.5, 0.2, 10.4]],  # 11: done
#     # [[0.5, 0., 0.55], [0.5, 0., 10.1]],
#     # [[0.5, 0.15, 0.05], [0.5, 0.15, 0.2]],
#     # [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],
#     # [[0.45, 0.02, 0.2], [0.6, 0.02, 0.2]],
#     # [[0.5, -0.0, 0.5], [0.3, -0.1, 10.5]],
#     # [[0.5, -0.05, 0.3], [0.5, 0.2, 10.25]],
#     # [[0.5, -0.0, 0.2], [0.5, 0.2, 10.4]],
#     # [[0.5, -0.0, 0.28], [0.5, 0.2, 10.4]],
#     # [[0.5, 0.25, 0.55], [0.5, 0.2, 10.4]],
#     # [[0.5, 0.1, 0.45], [0.5, 0.2, 10.4]],
# ]
positions_obstacles_list = [
    [[0.2, -0.5, 0.15], [0.5, 0., 10.1]],  # 0
    [[0.0, -0.5, 0.15], [0.5, 0.15, 0.2]],  # 1
    [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],  # 2
    [[-0.1, -0.72, 0.22], [0.6, 0.02, 10.2]],  # 3
    [[-0.1, -0.72, 0.22], [0.3, -0.1, 10.5]],  # 4
    [[-0.1, -0.72, 0.22], [0.5, 0.2, 10.25]],  # 5
    [[0.03, -0.6, 0.15], [0.5, 0.2, 10.4]],  # 6
    [[0.0, -0.72, 0.10], [0.5, 0.2, 10.4]],  # 7 :done
    # [[-0.1, -0.7,  0.3], [0.5, 0.2, 10.4]], #8: done, dynamic
    [[0., -0.72, 0.1], [0.5, 0.2, 10.4]],  # 9: done
    [[0.0, -0.72, 0.10], [0.5, 0.2, 10.4]],  # 10: done
    [[0.5, 0.0, 0.40], [0.5, 0.2, 10.4]],  # 11: done
]
speed_obstacles_list = [
    [[0., 0., 0.], [0., 0., 0.]],  # 0
    [[0., 0., 0.], [0., 0., 0.]],  # 1
    [[0., 0., 0.], [0., 0., 0.]],  # 2
    [[-0.05, 0., 0.], [0., 0., 0.]],  # 3
    [[0.04, 0., 0.], [0., 0., 0.]],  # 4
    [[-0.05, 0., 0.], [0., 0., 0.]],  # 5
    [[0., 0., 0.], [0., 0., 0.]],  # 6
    [[0., 0., 0.], [0., 0., 0.]],  # 7 :done
    # [[0.05, 0., 0.], [0., 0., 0.]], #8
    [[0.01, 0., 0.], [0., 0., 0.]],  # 9
    [[0., 0., 0.], [0., 0., 0.]],  # 10
    [[0., 0., 0.], [0., 0., 0.]],  # 11: done
]
n_runs = len(q_init_list)
network_yaml = "kuka_stableMP_fabrics_2nd_pouring"
#goal_pose = np.array([0.60829608, 0.04368581, 0.352421, 0.61566569, -0.37995015, 0.67837375, -0.12807299])

def resample_path(path, num_points):
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

def distance_to_NN(xee_list_SMP, xee_list):
    """
    Note: currently distance position AND orientation!!
    """
    len_SMP = len(xee_list_SMP)
    if len_SMP != len(xee_list):
        xee_list_i = resample_path(xee_list, len_SMP)
    else:
        xee_list_i = xee_list
    distances = np.absolute(np.array(xee_list_i) - np.array(xee_list_SMP))
    distance = np.mean(distances)
    distance_std = np.std(distances)
    return distances, distance, distance_std

def run_i(example_class, case:str, q_init_list:list, results_stableMP=None):
    results_tot = copy.deepcopy(results)
    example_class.overwrite_defaults(render=False)
    for i_run in range(n_runs):
        example_class.overwrite_defaults(init_pos=q_init_list[i_run], positions_obstacles=positions_obstacles_list[i_run], speed_obstacles=speed_obstacles_list[i_run])
        example_class.construct_example()
        results_i = example_class.run_kuka_example()
        if case == "GF" and results_i["goal_reached"] == False:
            print("i_run:", i_run)
        if results_stableMP is None:
            distances, _, _ = distance_to_NN(results_i["xee_list"], results_i["xee_list"])
        else:
            distances, _, _ = distance_to_NN(results_stableMP["xee_list"][i_run], results_i["xee_list"])
        results_tot["dist_to_NN"].append(distances)
        for key in results_i:
            results_tot[key].append(results_i[key])
    with open("simulation_kuka_"+case+".pkl", 'wb') as f:
        pickle.dump(results_tot, f)
    return results_tot

def KukaComparison():

    # --- run safe MP (only) example ---#
    class_SMP = example_kuka_stableMP_fabrics()
    class_SMP.overwrite_defaults(bool_combined=False, nr_obst=0)
    results_stableMP = run_i(class_SMP, case=cases[0], q_init_list=q_init_list, results_stableMP=None)

    # --- run safe MP (only) example ---#
    class_SMP_obst = example_kuka_stableMP_fabrics()
    class_SMP_obst.overwrite_defaults(bool_combined=False, nr_obst=2)
    results_stableMP_obst = run_i(class_SMP_obst, case=cases[1], q_init_list=q_init_list, results_stableMP=None)

    # --- run fabrics (only) example ---#
    results_fabrics = run_i(example_kuka_fabrics(), case=cases[2], q_init_list=q_init_list, results_stableMP=results_stableMP)

    # # --- hierarchical method ---#
    # # (env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
    # # example_hierachical = example_point_robot_hierarchical()
    # # q_list_hierarchical = example_hierachical.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos, dt=dt, mode=mode, mode_NN=mode_NN)

    # run safe MP + fabrics example ---#
    class_GM = example_kuka_stableMP_fabrics()
    class_GM.overwrite_defaults(bool_energy_regulator=False, bool_combined=True)
    results_stableMP_fabrics = run_i(class_GM, case=cases[3], q_init_list=q_init_list, results_stableMP=results_stableMP)

    # run theorem III.5 ---#
    class_CM = example_kuka_stableMP_fabrics()
    class_CM.overwrite_defaults(bool_energy_regulator=True, bool_combined=True)
    results_CM = run_i(class_CM, case=cases[4], q_init_list=q_init_list, results_stableMP=results_stableMP)

    results = {cases[0]: results_stableMP, cases[1]: results_stableMP_obst, cases[2]: results_fabrics, cases[3]: results_stableMP_fabrics, cases[4]: results_CM}
    return results

def KukaComparisonLoad():
    results = {}
    for case in cases:
        file_i = open(f'simulation_kuka_{case}.pkl', 'rb')
        results[case] = pickle.load(file_i)
    return results

def table_results(results):
    # --- create and plot table --- #
    rows = []
    title_row = [' ','Success-Rate', 'Time-to-Success [s]', "Min Clearance [m]", "Computation time [ms]", 'Path difference to PUMA'] #, "Input difference to PUMA"]
    nr_column = len(title_row)
    rows.append(title_row)
    for case in cases:
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
                     str(np.round(np.nanmean(results[case]["solver_times"])*1000, decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(results[case]["solver_times"])*1000, decimals=2)),
                     str(np.round(np.nanmean(np.concatenate(results[case]["dist_to_NN"], axis=0)), decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["dist_to_NN"], axis=0)), decimals=2)),
                     #str(np.round(np.nanmean(np.concatenate(results[case]["qdot_diff_list"], axis=0)), decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results[case]["qdot_diff_list"], axis=0)), decimals=2)),
                     ])
    table = Texttable()
    table.set_cols_align(["c"] * nr_column)
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(rows)
    print('\nTexttable Latex:')
    print(latextable.draw_latex(table)) #, caption="\small{Statistics for 50 simulated scenarios of our proposed methods \ac{gm} and \ac{cm} compared to 50 scenarios of \ac{gf} and \ac{smp}}"))
    print("results[case][goal_reached]:", results["GF"]["goal_reached"])

if __name__ == "__main__":
    if LOAD_RESULTS == False:
        # Regenerate the results:
        results = KukaComparison()
    else:
        results = KukaComparisonLoad()

    #plot the results:
    table_results(results)

    #plot results in figure:
    # plots_class = plotting_functions()
    # plots_class.comparison_plot_iiwa(results=results, cases=cases)

