import os
import gymnasium as gym
import numpy as np
import copy
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

from functions_stableMP_fabrics.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
from agent.utils.dynamical_system_operations import normalize_state
from agent.utils.normalizations import normalizaton_sim_NN
from functions_stableMP_fabrics.plotting_functions import plotting_functions
from functions_stableMP_fabrics.environments import trial_environments

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.

class example_point_robot_hierarchical():
    def __init__(self):
        dt = 0.01
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
        with open(absolute_path + "/examples/urdfs/point_robot.urdf", "r", encoding="utf-8") as file:
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


    def combine_action(self, M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot = []):
        xddot_combined = -np.dot(planner.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + xddot_speed
        if planner._mode == "vel":
            action_combined = qdot + planner._time_step * xddot_combined
        else:
            action_combined = xddot_combined
        return action_combined

    def simple_plot(self, list_fabrics_goal, list_fabrics_avoidance, list_safeMP, list_diff, dt=0.01):
        time_x = np.arange(0.0, len(list_fabrics_goal)*dt, dt)
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(time_x, list_fabrics_goal)
        ax[0].plot(time_x, list_safeMP, '--')
        ax[1].plot(time_x, list_fabrics_avoidance, '-')
        # ax[1].plot(time_x, list_fabrics_goal, "--")
        ax[2].plot(time_x, list_diff, '-')
        ax[2].plot()
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[0].set(xlabel="time (s)", ylabel="x [m]", title="Actions fabrics, safeMP")
        ax[1].set(xlabel="time (s)", ylabel="x [m]", title="Action fabr avoidance")
        ax[2].set(xlabel="time (s)", ylabel="x [m]", title="Action difference")
        ax[0].legend(["x", "y", "$x_{IL}$", "$y_{IL}$"])
        ax[1].legend(["x", "y"])
        ax[2].legend(["x", "y"])
        plt.savefig("difference_fabrics_safeMP.png")

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
        # mode = "vel"
        # dt = 0.01
        #
        # # replication example panda robot
        # init_pos = np.array([0.6, 0.0, 0.0])
        # goal_pos = [0.1, -0.6]
        scaling_factor = 3
        scaling_room = {"x": [-scaling_factor, scaling_factor], "y": [-scaling_factor, scaling_factor]}

        # pointmass example with obstacle (also uncomment obstacle)
        # init_pos = np.array([-0.9, -0.1, 0.0])
        # goal_pos = [3.5, 0.5]
        # scaling_factor = 10
        # scaling_room = {"x": [-scaling_factor, scaling_factor], "y":[-scaling_factor, scaling_factor]}

        # (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
        planner = self.set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)

        # action_safeMP = np.array([0.0, 0.0, 0.0])
        action_fabrics = np.array([0.0, 0.0, 0.0])
        ob, *_ = env.step(action_fabrics)
        q_list = np.zeros((2, n_steps))

        # Parameters
        params_name = '2nd_order_2D'
        x_t_init = np.array([np.append(ob['robot_0']["joint_state"]["position"][0:2], ob['robot_0']["joint_state"]["velocity"][0:2])]) # initial states
        # simulation_length = 2000
        results_base_directory = './'

        # Load parameters
        Params = getattr(importlib.import_module('params.' + params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Translation of goal:
        normalizations = normalizaton_sim_NN(scaling_room=scaling_room)
        state_goal = np.array((goal._sub_goals[0]._config["desired_position"]))
        goal_normalized = normalizations.call_normalize_state(state=state_goal)
        translation = normalizations.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN)
        translation_gpu = torch.FloatTensor(translation).cuda()

        # Initialize dynamical system
        min_vel = learner.min_vel
        max_vel = learner.max_vel
        x_init_gpu, x_init_cpu = normalizations.transformation_to_NN(x_t=x_t_init, translation_gpu=translation_gpu,
                                                      dt=dt, min_vel=min_vel, max_vel=max_vel)
        dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)
        # dynamical_system.saturate

        # Initialize trajectory plotter
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        trajectory_plotter = TrajectoryPlotter(fig, x0=x_init_cpu, pause_time=1e-5, goal=data['goals training'][0])
        # x_t_NN = torch.FloatTensor(x_t_init_scaled).cuda()

        # Initialize lists

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:2]
            qdot = ob_robot["joint_state"]["velocity"][0:2]
            q_list[:, w] = q
            x_t = np.array([np.append(q, qdot)])

            # --- translate to axis system of NN ---#
            x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                           dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)

            # --- get action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]

            # denormalized position of NN, used as goal for fabrics
            pos_safeMP = normalizations.reverse_transformation_pos(x_t_NN[0][0:2])

            # --- get action by fabrics --- #
            arguments_dict = dict(
                q=ob_robot["joint_state"]["position"][0:dof],
                qdot=ob_robot["joint_state"]["velocity"][0:dof],
                x_goal_0=pos_safeMP,
                weight_goal_0=30, #ob_robot['FullSensor']['goals'][2]['weight'],
                x_obst_0=ob_robot['FullSensor']['obstacles'][3]['position'],
                radius_obst_0=ob_robot['FullSensor']['obstacles'][3]['size'],
                x_obst_1=ob_robot['FullSensor']['obstacles'][4]['position'],
                radius_obst_1=ob_robot['FullSensor']['obstacles'][4]['size'],
                # x_obst_2=ob_robot['FullSensor']['obstacles'][5]['position'],
                # radius_obst_2=ob_robot['FullSensor']['obstacles'][5]['size'],
                radius_body_base_link_y=np.array([0.2])
            )
            action_fabrics[0:dof] = planner.compute_action(**arguments_dict)

            # --- update environment ---#
            ob, *_, = env.step(np.append(action_fabrics, 0))

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
        plt.savefig(params.results_path+"images/point_robot_hierarchical")
        env.close()
        # simple_plot(list_fabrics_goal=list_fabr_goal, list_fabrics_avoidance=list_fabr_avoidance,
        #             list_safeMP=list_safeMP, list_diff = list_diff, dt=dt)
        make_plots = plotting_functions(results_path=params.results_path)
        make_plots.plotting_q_values(q_list, dt=dt, q_start=q_list[:, 0], q_goal=np.array(goal_pos))
        return q_list

if __name__ == "__main__":
    # --- Initial parameters --- #
    mode = "acc"
    mode_NN = "2nd"
    dt = 0.01
    init_pos = np.array([0., 0.])
    goal_pos = [-2.4355761, -7.5252747]
    render = True

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos)

    # --- run example --- #
    example_class = example_point_robot_hierarchical()
    res = example_class.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN)