from grasp_planning import IK_OPTIM
import numpy as np
import time
import os
from scipy.spatial.transform import Rotation as R
from functions_stableMP_fabrics.kinematics_kuka import KinematicsKuka

class IKGomp():
    def __init__(self, q_home=None):
        # Current robot's state
        if q_home is None:
            self.q_home = np.array([0.0, 0.0, 1.5, 1.5, 0.0, 0.0, 0.0], dtype=float)
        else:
            self.q_home = q_home

        # Obstacle's pose
        T_W_Obst = np.eye(4)
        T_W_Obst[:3, 3] = np.array([-1., 0.4, 0.15]).T

        self.kinematics_kuka = KinematicsKuka()

    def construct_ik(self, urdf_path="/../examples/urdfs/iiwa14.urdf"):
        # URDF model
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        URDF_FILE = absolute_path + "/examples/urdfs/iiwa14.urdf"

        # Create IK solver
        self.planner = IK_OPTIM(urdf=URDF_FILE,
                           root_link = "world",
                           end_link  = 'iiwa_link_ee')
        self.planner.set_home_config(self.q_home)
        self.planner.set_init_guess(self.q_home)
        self.planner.set_boundary_conditions() # joint limits
        self.planner.add_position_constraint(tolerance=0.0)
        self.planner.add_orientation_constraint(tolerance=0.0)
        # Define collision constraint for each link
        active_links = [f'iiwa_link_{i}' for i in range(8)]
        active_links.append('iiwa_link_ee')
        # for link in active_links:
        #     planner.add_collision_constraint(child_link=link,
        #                                      r_link=0.2,
        #                                      r_obst=0.2,
        #                                      tolerance=0.01)
        # Formulate problem
        self.planner.setup_problem(verbose=False)
        return

    def call_ik(self, goal_position, goal_orientation, positions_obsts: list):
        r = R.from_quat(goal_orientation)
        orientation_matrix = r.as_matrix()
        T_W_Ref = np.eye(4)
        T_W_Ref[:3, :3] = orientation_matrix
        T_W_Ref[:3, 3] = goal_position

        # Obstacle's pose
        T_W_Obst = np.eye(4)
        T_W_Obst[:3, 3] = np.array([-1., 0.4, 0.15]).T

        # Call IK solver
        self.planner.update_constraints_params(T_W_Ref=T_W_Ref)
        start = time.time()
        x, solver_flag = self.planner.solve()
        end = time.time()
        print(f"Computational time: {end - start}")
        print(f"Solver status: {solver_flag}")
        print("========")
        print("Desired T\n", T_W_Ref)

        T_W_Grasp = self.planner._robot_model.eval_fk(x)
        print("IK solver\n", T_W_Grasp)

        x_t, _, _ = self.kinematics_kuka.get_state_task(q=x.full().transpose()[0], quat_prev = np.array([1., 0., 0., 0.]))
        print("solution kinematics kuka:", x_t)

        # T_W_Grasp = planner._robot_model.eval_fk(x)
        # print(T_W_Grasp)
        return x, solver_flag

if __name__ == '__main__':
    goal_position = np.array([[0.52677347, 0.35190244, 0.48685025]])
    goal_orientation = np.array([0.81098249, 0.17874092, 0.44319898, -0.33754081])
    position_obst1 = np.array([-1., 0.4, 0.15])

    ik_gomp = IKGomp()
    ik_gomp.construct_ik()
    x, solver_flag  = ik_gomp.call_ik(goal_position, goal_orientation, positions_obsts=[position_obst1])

    print(f"Solution: {x}" )
    print(f"Solver status: {solver_flag}" )