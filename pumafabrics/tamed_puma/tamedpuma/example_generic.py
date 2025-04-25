import numpy as np

class ExampleGeneric:
    def __init__(self):
        pass

    def overwrite_defaults(self, params, render=None, init_pos=None, goal_pos=None, nr_obst=None, bool_energy_regulator=None,
                           positions_obstacles=None, orientation_goal=None, params_name_1st=None, speed_obstacles=None, goal_vel=None,
                           bool_combined=None, n_steps=None):
        self.params = params
        if render is not None:
            self.params["render"] = render
        if init_pos is not None:
            self.params["init_pos"] = init_pos
        if goal_pos is not None:
            self.params["goal_pos"] = goal_pos
        if goal_vel is not None:
            self.params["goal_vel"] = goal_vel
        if orientation_goal is not None:
            self.params["orientation_goal"] = orientation_goal
        if nr_obst is not None:
            self.params["nr_obst"] = nr_obst
        if bool_energy_regulator is not None:
            self.params["bool_energy_regulator"] = bool_energy_regulator
        if bool_combined is not None:
            self.params["bool_combined"] = bool_combined
        if positions_obstacles is not None:
            self.params["positions_obstacles"] = positions_obstacles
        if params_name_1st is not None:
            self.params["params_name_1st"] = params_name_1st
        if speed_obstacles is not None:
            self.params["speed_obstacles"] = speed_obstacles
        if goal_vel is not None:
            self.params["goal_vel"] = goal_vel
        if n_steps is not None:
            self.params["n_steps"] = n_steps

    def integrate_to_vel(self, qdot, action_acc, dt):
        qdot_action = action_acc *dt +qdot
        return qdot_action

    def check_goal_reached(self, x_ee, x_goal):
        self.dist = np.linalg.norm(x_ee - x_goal)
        if self.dist<0.02:
            self.GOAL_REACHED = True
            return True
        else:
            return False
        
    def return_distance_goal_reached(self):
        return self.dist