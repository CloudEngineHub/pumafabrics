import os
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.tamedpuma.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from pumafabrics.tamed_puma.utils.plot_point_robot import plotting_functions
from pumafabrics.tamed_puma.tamedpuma.combining_actions import combine_fabrics_safeMP
from pumafabrics.tamed_puma.create_environment.environments import trial_environments

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.

class example_point_robot_fabrics():
    def __init__(self):
        pass

    def set_planner(self, goal: GoalComposition, ONLY_GOAL=False, bool_speed_control=True, mode="acc", dt=0.01):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        Params
        ----------
        goal: StaticSubGoal
            The goal to the motion planning problem.
        """
        degrees_of_freedom = 2
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/../pumafabrics/tamed_puma/config/urdfs/point_robot.urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            root_link="world",
            end_links=["base_link_y"],
        )
        collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
        collision_finsler = "2.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
        planner = ParameterizedFabricPlannerExtended(
            degrees_of_freedom,
            forward_kinematics,
            time_step=dt,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
            )
        collision_links = ["base_link_y"]
        # The planner hides all the logic behind the function set_components.
        if ONLY_GOAL == 1:
            planner.set_components(
                goal=goal,
            )
        else:
            planner.set_components(
                collision_links=collision_links,
                goal=goal,
                number_obstacles=2,
            )
        # planner.concretize(extensive_concretize=True, bool_speed_control=bool_speed_control)
        planner.concretize_extensive(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
        return planner

    def run_point_robot_urdf(self, n_steps=2000, env=None, goal=None, init_pos=np.array([-5, 5]), goal_pos=[-2.4355761, -7.5252747], mode="acc", mode_NN="2nd", dt=0.01):
        """
        Set the gym environment, the planner and run point robot example.
        The initial zero action step is needed to initialize the sensor in the
        urdf environment.

        Params
        ----------
        n_steps
            Total number of simulation steps.
        render
            Boolean toggle to set rendering on (True) or off (False).
        """
        # --- parameters --- #
        dof = 2
        if mode == "vel":
            str_mode = "velocity"
        elif mode == "acc":
            str_mode = "acceleration"
        else:
            print("this control mode is not defined")

        # (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
        planner = self.set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)
        planner_goal = self.set_planner(goal=goal, ONLY_GOAL=True, bool_speed_control=True, mode=mode, dt=dt)
        planner_avoidance = self.set_planner(goal=None, bool_speed_control=True, mode=mode, dt=dt)

        # create class for combined functions on fabrics + safeMP combination
        v_min = -50*np.ones((dof,))
        v_max = 50*np.ones((dof,))
        acc_min = -50*np.ones((dof,))
        acc_max = 50*np.ones((dof,))
        combined_geometry = combine_fabrics_safeMP(v_min = v_min, v_max=v_max, acc_min=acc_min, acc_max=acc_max)

        action_safeMP = np.array([0.0, 0.0, 0.0])
        action_fabrics = np.array([0.0, 0.0, 0.0])
        ob, *_ = env.step(action_safeMP)
        q_list = np.zeros((2, n_steps))

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:2]
            qdot = ob_robot["joint_state"]["velocity"][0:2]
            q_list[:, w] = q

            # --- get action by fabrics --- #
            arguments_dict = dict(
                q=ob_robot["joint_state"]["position"][0:dof],
                qdot=ob_robot["joint_state"]["velocity"][0:dof],
                x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:dof],
                weight_goal_0=10, #ob_robot['FullSensor']['goals'][2]['weight'],
                x_obst_0=ob_robot['FullSensor']['obstacles'][3]['position'],
                radius_obst_0=ob_robot['FullSensor']['obstacles'][3]['size'],
                x_obst_1=ob_robot['FullSensor']['obstacles'][4]['position'],
                radius_obst_1=ob_robot['FullSensor']['obstacles'][4]['size'],
                radius_body_base_link_y=np.array([0.2])
            )
            action_fabrics[0:dof] = planner.compute_action(**arguments_dict)

            # --- update environment ---#
            ob, *_, = env.step(np.append(action_fabrics, 0))

        env.close()
        make_plots = plotting_functions(results_path="../examples/images/")
        make_plots.plotting_q_values(q_list, dt=dt, q_start=q_list[:, 0], q_goal=np.array(goal_pos), file_name="point_robot_q_Fabrics")
        return q_list

def main(render=True):
    # --- Initial parameters --- #
    mode = "acc"
    mode_NN = "2nd"
    dt = 0.01
    init_pos = np.array([0.0, 0.0])
    goal_pos = [-2.4355761, -7.5252747]

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos)

    # --- run example --- #
    example_class = example_point_robot_fabrics()
    res = example_class.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN)
    return {}

if __name__ == "__main__":
    main()