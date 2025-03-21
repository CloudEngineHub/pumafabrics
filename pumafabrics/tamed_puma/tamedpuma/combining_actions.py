import numpy as np
class combine_fabrics_safeMP():
    def __init__(self, v_min=0, v_max=0, acc_min=0, acc_max=0):
        self.v_min = v_min
        self.v_max = v_max
        self.acc_min = acc_min
        self.acc_max = acc_max

    def combine_action(self, M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot = []):
        xddot_combined = -np.dot(planner.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + xddot_speed
        if planner._mode == "vel":
            action_combined = qdot + planner._time_step * xddot_combined
        else:
            action_combined = xddot_combined

        # constrain action
        action = self.get_action_in_limits(action_combined, mode=planner._mode)
        return action

    def get_action_in_limits(self, action_old, mode="acc"):
        if mode == "vel":
            action = np.clip(action_old, self.v_min, self.v_max)
        else:
            action = np.clip(action_old, self.acc_min, self.acc_max)
        return action