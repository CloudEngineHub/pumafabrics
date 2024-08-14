import pickle
import numpy as np

file_i = open(f'results/data_files/simulation_kuka_results_ALL.pkl', 'rb')
results_no_obst = pickle.load(file_i)
kkk=1



# for case in results_no_obst.keys():
#     for i in range(len(results_no_obst[case]["dist_to_NN"])):
#         str_dist_to_NN = str(np.round(np.nanmean(np.concatenate(results_no_obst[case]["dist_to_NN"][i], axis=0)), decimals=2)) + " $\pm$ " + str(np.round(np.nanstd(np.concatenate(results_no_obst[case]["dist_to_NN"][i], axis=0)), decimals=2))
#         print(i, " with ", str_dist_to_NN)


# file_i = open(f'ee_state_0.pk', 'rb')
# results_no_obst = pickle.load(file_i)
# kk=1