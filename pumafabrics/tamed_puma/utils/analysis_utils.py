"""
This file includes tools for the final analysis of the planner, such as distance to obstacles and time to reach the goal.
"""
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka

class UtilsAnalysis():
    def __init__(self, forward_kinematics: GenericURDFFk, collision_links:list, collision_radii: dict, kinematics) -> None:
        self.min_dist = 1000
        self.collision_links = collision_links
        self.collision_radii = collision_radii
        self.fk = forward_kinematics
        self.goal_reach_thr = 0.05
        self.kuka_kinematics = kinematics

    def check_distance_collision(self, q: np.ndarray, obstacles: list, margin=0.0, parent_link='iiwa_link_0') -> bool:
        """
        Outputs if there is a collision: True, False
        Keeps track of the minimum distance to an obstacle
        """
        if len(obstacles) == 0:
            return False

        for i_link, collision_link in enumerate(self.collision_links):
            pose_link_i = self.fk.numpy(
                q,
                parent_link=parent_link,
                child_link=collision_link,
            )
            for obstacle in obstacles:
                dist = np.linalg.norm(pose_link_i[:3, 3] - np.array(obstacle["position"])) - obstacle["size"] - \
                       self.collision_radii[int(collision_link[-1])] + margin
                if dist < self.min_dist:
                    self.min_dist = dist

        if self.min_dist < 0:
            print("IN COLLISION, minimal dist during this run: ", self.min_dist)
            return True
        else:
            return False

    def _request_ee_state(self, q, quat_prev):
         # --- end-effector states and normalized states --- #
        x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev, mode_NN="1st")
        return x_t, xee_orientation

    def check_error_pos_ee(self, x_goal, x_ee, x_orientation=None, orientation_goal=None):
        error_position = np.linalg.norm(x_ee - x_goal)
        if x_orientation is not None:
            error_orientation = np.linalg.norm(x_orientation - orientation_goal)
            return error_position, error_orientation
        else:
            return error_position

    def check_goal_reached(self, error:np.ndarray, threshold=None):
        if threshold is None:
            threshold = self.goal_reach_thr

        # Get norm error
        norm_error = np.linalg.norm(error)

        # If close to goal, reached
        if norm_error < threshold:
            return True
        else:
            return False

    def check_goal_reaching(self, q, quat_prev, x_goal):
        x_ee, _ = self._request_ee_state(q, quat_prev)
        error = self.check_error_pos_ee(x_goal, x_ee[0][0:3])
        GOAL_REACHED = self.check_goal_reached(error=error)
        return GOAL_REACHED, error

    def get_min_dist(self):
        return self.min_dist
