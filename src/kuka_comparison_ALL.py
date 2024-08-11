import numpy as np
from texttable import Texttable
import latextable
from kuka_comparison_2nd import comparison_kuka_class

exec(open('kuka_comparison_2nd_no_obst.py').read())
exec(open('kuka_comparison_2nd_no_obst_pouring.py').read())

# cases = ["PUMA", "GF", "GM", "CM"]
# results = {"min_distance": [], "collision": [], "goal_reached": [], "time_to_goal": [], "xee_list": [], "qdot_diff_list": [],
#            "dist_to_NN": [],  "vel_to_NN": [], "solver_times": [], "solver_time": [], "solver_time_std": []}
# n_runs = 20




def append_results(results_list):
    h=1
def table_results(results):
    # --- create and plot table --- #
    rows = []
    title_row = [' ','Success-Rate',  'Path difference to PUMA', "Input difference to PUMA"]
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
                     #str(np.round(np.nanmean(results[case]["time_to_goal"]), decimals=4)) + " $\pm$ " + str(np.round(np.nanstd(results[case]["time_to_goal"]), decimals=4)),
                     # collision_episodes_rate_str,
                     #min_clearance_str,
                     #str(np.round(np.nanmean(results[case]["solver_times"])*1000, decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(results[case]["solver_times"])*1000, decimals=2)),
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

results_list = [results, results_pouring]





