from grasp_planning import IK_OPTIM
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R
# from functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka
# from kinematics_kuka import KinematicsKuka
import copy
import spatial_casadi as sc

class IKGomp():
    def __init__(self, q_home=None):
        # Current robot's state
        if q_home is None:
            self.q_home = np.array([0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0], dtype=float)
        else:
            self.q_home = q_home

    def construct_ik(self, urdf_path="/../examples/urdfs/iiwa14.urdf", nr_obst=0):
        # URDF model
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        URDF_FILE = absolute_path + urdf_path #"/examples/urdfs/iiwa14.urdf"

        # Create IK solver
        self.planner = IK_OPTIM(urdf=URDF_FILE,
                           root_link='world',
                           end_link='iiwa_link_ee')
        self.planner.set_init_guess(self.q_home)
        self.planner.set_boundary_conditions()  # joint limits

        self.planner.add_objective_function(name="objective")
        self.planner.add_position_constraint(name="g_position", tolerance=0.0)
        self.planner.add_orientation_constraint(name="g_rotation", tolerance=10.)

        # Define collision constraint for each link
        active_links = [f'iiwa_link_{i}' for i in range(8)]
        active_links.append('iiwa_link_ee')
        for i in range(nr_obst):
            self.planner.add_collision_constraint(name="sphere_col_"+str(i),
                                                  link_names=active_links,
                                                  r_link=0.10,
                                                  r_obst=0.05,
                                                  tolerance=0.01)
        # Formulate problem
        self.planner.setup_problem(verbose=False)
        return

    def call_ik(self, goal_position, goal_orientation, positions_obsts: list, q_init_guess:np.ndarray, q_home:np.ndarray):
        T_W_Ref = self.construct_T_matrix(position=goal_position, orientation=goal_orientation)

        # Obstacle's pose
        if len(positions_obsts)>0:
            for i in range(len(positions_obsts)):
                T_W_Obst = self.construct_T_matrix(position=positions_obsts[i])
                self.planner.param_ca_dict["sphere_col_"+str(i)]["num_param"] = T_W_Obst[:3, 3]

        # Call IK solver
        if q_init_guess is not None:
            self.planner.set_init_guess(q_init_guess)
        self.planner.param_ca_dict["objective"]["num_param"] = q_home  # setting home configuration
        self.planner.param_ca_dict["g_position"]["num_param"] = T_W_Ref
        self.planner.param_ca_dict["g_rotation"]["num_param"] = T_W_Ref
        # if T_W_Obst is not None:
        #     self.planner.param_ca_dict["sphere_col"]["num_param"] = T_W_Obst[:3, 3]

        start = time.time()
        x, solver_flag = self.planner.solve()
        end = time.time()
        # print(f"Computational time: {end - start}")
        # print(f"Solver status: {solver_flag}")
        # print("========")
        # print("Desired T\n", T_W_Ref)
        # print("solution: q_d:", x)
        return x.full().transpose()[0], solver_flag

    def construct_T_matrix(self, position:np.ndarray, orientation=None):
        """
        Transformation matrix from pose (position + quaternion)
        """
        try:
            r = R.from_quat(orientation)
            orientation_matrix = r.as_matrix()
        except:
            orientation_matrix = np.eye(3)
        position_transpose = position[:, np.newaxis]
        T = np.concatenate([orientation_matrix, position_transpose], axis=1)
        T= np.concatenate([T, np.array([[0, 0, 0, 1]])])
        return T

    def construct_T_matrices(self, positions_obsts:list):
        """
        Multiple transformation matrices
        """
        T_list = []
        for positions_obst in positions_obsts:
            T = self.construct_T_matrix(positions_obst)
            T_list.append(T)
        return T_list

    def forward_kinematics(self, q):
        T_W_Grasp = self.planner._robot_model.eval_fk(q)

        position = T_W_Grasp[:3, 3]
        rot_matrix = T_W_Grasp[:3, :3]
        orientation_quat = self.rot_matrix_to_quat(rot_matrix=rot_matrix)
        return np.append(position, orientation_quat)

    def get_current_pose(self, q, quat_prev):
        x_t_pose = self.forward_kinematics(q)
        x_t_pose[3:7] = self.check_quaternion_flipped(x_t_pose[3:7], quat_prev)
        return np.array([x_t_pose]), x_t_pose[3:7]

    def get_initial_pose(self, q_init, offset_orientation):
        # initial state pose:
        x_t_init_pose = self.forward_kinematics(q_init)
        x_t_init_pose[3:7] = self.check_quaternion_initial(x_orientation=x_t_init_pose[3:], quat_offset=offset_orientation)
        return x_t_init_pose

    def rot_matrix_to_quat(self, rot_matrix):
        r = sc.Rotation.from_matrix(rot_matrix)
        quatern = r.as_quat()
        return quatern

    # ---------------------- check flips of quaternions ---------------------------------- #
    def check_quaternion_flipped(self, quat, quat_prev):
        dist_quat = np.linalg.norm(quat - quat_prev)
        if dist_quat > 1.0:
            quat_new = -1*copy.deepcopy(quat)
            "flip quaternion!"
        else:
            quat_new = copy.deepcopy(quat)
        return quat_new

    def check_quaternion_initial(self, x_orientation, quat_offset):
        """Check that we start in hemisphere nearest to goal"""

        #orientations (flipped and unflipped)
        orientation_1 = copy.deepcopy(x_orientation)
        orientation_2 = -copy.deepcopy(orientation_1)

        # Calculate the dot product
        dot_product_1 = np.dot(orientation_1, quat_offset)
        dot_product_2 = np.dot(orientation_2, quat_offset)

        #check angular distance:
        angular_distance_1 = np.arccos(dot_product_1)
        angular_distance_2 = np.arccos(dot_product_2)

        if angular_distance_1 < angular_distance_2:
            quat_new = copy.deepcopy(orientation_1)
            print("Not flipped:  ", angular_distance_1)
        else:
            quat_new = copy.deepcopy(orientation_2)
            print("Flipped: ", angular_distance_2)

        return quat_new

if __name__ == '__main__':
    # for this example: check if q = q_d
    goal_orientation = np.array([-0.18765569, -0.44188011, 0.34243112, -0.80763125])
    q = np.array([0.8999998 ,  0.78999992, -0.21999469, -1.33000009,  1.2, -1.76000009, -1.05999994])

    position_obst1 = np.array([-1., 0.4, 0.15])
    ik_gomp = IKGomp(q_home=q)
    ik_gomp.construct_ik()

    init_pose, _ = ik_gomp.get_current_pose(q, goal_orientation)
    q_d, solver_flag  = ik_gomp.call_ik(init_pose[0][0:3], init_pose[0][3:7], positions_obsts=[position_obst1], q_init_guess=q, q_home=q)
    ik_gomp.get_current_pose(q_d, goal_orientation)

    print("solution computed via IK is not the inverted forward kinematics:")
    print(f"Current configuration: {q}")
    print(f"Solution: {q_d}" )
    print(f"Solver status: {solver_flag}" )