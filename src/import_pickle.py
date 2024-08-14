import pickle
import numpy as np

file_all = open(f'simulation_kuka_results_ALL.pkl', 'rb')
results_all = pickle.load(file_all)

file_i = open(f'results/data_files/simulation_kuka_2nd7.pkl', 'rb')
results = pickle.load(file_i)
kkk=1

file_2 = open(f'results/data_files/simulation_kuka_2nd9.pkl', 'rb')
results_5 = pickle.load(file_2)
kkk=1

for case in results.keys():
    for key in results[case].keys():
        if key != "vel_to_NN":
            print("key:", key)
            results[case][key][5] = results_5[case][key][0]

kkk = 1

with open("simulation_kuka_results_pouring_complete.pkl", 'wb') as f:
    pickle.dump(results, f)

file_3 = open(f'simulation_kuka_results_pouring_complete.pkl', 'rb')
results_total = pickle.load(file_3)







# for case in results_no_obst.keys():
#     for i in range(len(results_no_obst[case]["dist_to_NN"])):
#         str_dist_to_NN = str(np.round(np.nanmean(np.concatenate(results_no_obst[case]["dist_to_NN"][i], axis=0)), decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results_no_obst[case]["dist_to_NN"][i], axis=0)), decimals=2))
#         print(i, " with ", str_dist_to_NN)


# file_i = open(f'ee_state_0.pk', 'rb')
# results_no_obst = pickle.load(file_i)
# kk=1