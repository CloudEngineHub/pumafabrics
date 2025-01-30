import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from pumafabrics.tamed_puma.kinematics.kinematics_basics import KinematicsBasics

# --- for dinova ---#
indexes = [60]
folder_name = "dinova_30jan_pick"
robot_name = "dinova"
root_link_name = "base_link"
end_link_name = "arm_tool_frame"

# # ---- for kinova ---#
# indexes = [50+i for i in range(9)]
# folder_name = "kinova_30jan_pick"
# robot_name = "gen3lite"
# root_link_name = "kinova_base_link"
# end_link_name = "tool_frame"

kinematics_basics = KinematicsBasics(robot_name=robot_name,
                                     end_link_name=end_link_name,
                                     root_link_name=root_link_name)
for z, index in enumerate(indexes):
    results_dict = {"x_pos":[], "x_rot":[], "x_dot": [], "delta_t": []}
    results_dict_processed = {"x_pos": [], "x_rot": [], "x_dot": [], "delta_t": []}
    file_i = open(folder_name+"_original"+f'/recording_demonstration_{index}.pk', 'rb')
    results_i = pickle.load(file_i)

    if robot_name == "kinova":
        results_dict["x_pos"] = [np.array(results_i["x_pos"][i]) for i in range(len(results_i["x_pos"]))]
        results_dict["x_rot"] = [R.from_quat(results_i["x_quat"][i]).as_matrix() for i in range(len(results_i["x_quat"]))]
    else:
        results_dict["x_pos"] = [np.array(results_i["pos_fk"][i]) for i in range(len(results_i["x_pos"]))]
        results_dict["x_rot"] = [R.from_quat(results_i["quat_fk"][i]).as_matrix() for i in range(len(results_i["x_quat"]))]
    time_list = [np.array(results_i["time"][i]) for i in range(len(results_i["time"]))]
    results_dict["delta_t"] = np.diff(np.array(time_list))

    for i in range(len(results_i["q"])):
        if robot_name == "dinova":
            q = np.append(np.array(results_i["base_pose"][i]), np.array(results_i["q"][i]))
            q_dot = np.append(np.array(results_i["base_vel"][i]), np.array(results_i["q_dot"][i]))
        else:
            q = np.array(results_i["q"][i])
            q_dot = np.array(results_i["q_dot"][i])
        x_dot, _ = kinematics_basics.get_state_velocity(q, q_dot)
        results_dict["x_dot"].append(x_dot)

    # --- remove when delta_t=0, if the same message was received twice ---#
    for i, delta_t in enumerate(results_dict["delta_t"]):
        if delta_t != 0.:
            results_dict_processed["delta_t"].append(delta_t)
            results_dict_processed["x_pos"].append(results_dict["x_pos"][i])
            results_dict_processed["x_rot"].append(results_dict["x_rot"][i])
            results_dict_processed["x_dot"].append(results_dict["x_dot"][i])

    with open(folder_name + f'/ee_state_{z}.pk', "wb") as f:
        pickle.dump(results_dict_processed, f)
