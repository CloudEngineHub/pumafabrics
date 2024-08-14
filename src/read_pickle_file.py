import pickle
from functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka
from scipy.spatial.transform import Rotation as R
import numpy as np
from grasp_planning import IK_OPTIM
import os

kuka_kinematics = KinematicsKuka(dt=0.02, end_link_name="iiwa_link_7", robot_name="iiwa14")

# Create IK solver
q_home = np.array([0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0], dtype=float)
absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/examples/urdfs/iiwa14.urdf"

planner = IK_OPTIM(urdf=URDF_FILE,
                   root_link = 'world',
                   end_link  = 'iiwa_link_ee')
planner.set_init_guess(q_home)
planner.set_boundary_conditions() # joint limits

planner.add_objective_function(name="objective")
planner.add_position_constraint(name="g_position", tolerance=0)
planner.add_orientation_constraint(name="g_rotation", tolerance=0.01)
planner.setup_problem(verbose=False)

# Path to the .pk file
for i in range(11):
    file_path_template = 'datasets/kuka/pouring/ee_state_%s.pk'
    file_path = file_path_template % i

    # Load the content from the .pk file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    orientation_last = data["x_rot"][-1]

    # DO NOT use pk for this transformation!!!
    # orientation_last_quaternions = kuka_kinematics.rot_matrix_to_quat(orientation_last)
    # print("orientation_quaternions: ", orientation_last_quaternions)

    # but use scipy!
    orientation_last_R = R.from_matrix(orientation_last)
    orientation_last_quat_scipy = orientation_last_R.as_quat()
    # print("position_last: ", data["x_pos"][-1])
    # print("orientation_last_quat_scipy:", orientation_last_quat_scipy)

    # position and orientation initial:
    x_init_pos = data["x_pos"][0]
    x_init_rot = data["x_rot"][0]
    # print("position_initial: ", x_init_pos)
    # print("orientation_initial:", x_init_rot)

    # goal_position = np.array([[-0.09486833, -0.72446137, 0.22143809]])
    # goal_orientation = np.array([0.50443695, -0.51479307, 0.68849319, -0.08067585])
    # r = R.from_quat(goal_orientation)
    orientation_matrix = R.from_matrix(x_init_rot)
    T_W_Ref = np.eye(4)
    T_W_Ref[:3, :3] = x_init_rot
    T_W_Ref[:3, 3] = x_init_pos

    planner.set_init_guess(q_home)
    planner.param_ca_dict["objective"]["num_param"] = q_home # setting home configuration
    planner.param_ca_dict["g_position"]["num_param"] = T_W_Ref
    planner.param_ca_dict["g_rotation"]["num_param"] = T_W_Ref

    x, solver_flag = planner.solve()

    print(f"Iteration {i}")
    print(f"Solver status: {solver_flag}" )
    print("x:", x)