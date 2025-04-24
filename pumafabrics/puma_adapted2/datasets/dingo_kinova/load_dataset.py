import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
from pumafabrics.tamed_puma.kinematics.kinematics_basics import KinematicsBasics

# --- for dinova ---#
# indexes = [60]
# folder_name = "dinova_30jan_pick"
# robot_name = "dinova"
# root_link_name = "base_link"
# end_link_name = "arm_tool_frame"

# # ---- for kinova ---#
indexes = [0+i for i in range(10)]
folder_name = "kinova_23apr"
robot_name = "gen3lite_1"
root_link_name = "kinova_base_link"
end_link_name = "tool_frame"

kinematics_basics = KinematicsBasics(robot_name=robot_name,
                                     end_link_name=end_link_name,
                                     root_link_name=root_link_name)
for z, index in enumerate(indexes):
    file_i = open(folder_name + "" + f'/ee_state_{z}.pkl', 'rb')
    results_i = pickle.load(file_i)
    kkk=1
