import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from src.functions_stableMP_fabrics.geometry_IL import construct_IL_geometry
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from agent.utils.normalizations import normalizaton_sim_NN
from functions_stableMP_fabrics.plotting_functions import plotting_functions
from functions_stableMP_fabrics.environments import trial_environments
from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework

class example_panda_stableMP_fabrics_2D_2nd_hierarchical():
    def __init__(self):
        dt = 0.01
    def set_planner(self, goal: GoalComposition, degrees_of_freedom: int = 7, mode="acc", dt=0.01, bool_speed_control=True):
        """
        Initializes the fabric planner for the panda robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        Params
        ----------
        goal: StaticSubGoal
            The goal to the motion planning problem.
        degrees_of_freedom: int
            Degrees of freedom of the robot (default = 7)
        """

        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/panda_for_fk.urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            rootLink="panda_link0",
            end_link="panda_link9",
        )
        collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
        collision_finsler = "2.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
        planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            forward_kinematics,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
        )
        collision_links = ['panda_link9', 'panda_link7', 'panda_link3', 'panda_link4']
        panda_limits = [
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8973, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973]
            ]
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            collision_links=collision_links,
            goal=goal,
            number_obstacles=0,
            number_plane_constraints=1,
            limits=panda_limits,
        )
        planner.concretize(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
        return planner

    def combine_action(self, M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot = []):
        xddot_combined = -np.dot(planner.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + xddot_speed
        if planner._mode == "vel":
            action_combined = qdot + planner._time_step * xddot_combined
        else:
            action_combined = xddot_combined
        return action_combined

    def run_panda_example(self, n_steps=2000, env=None, goal=None, init_pos=np.array([0.0, 0.0, 0.]), goal_pos=[0.1, -0.6, 0.4], mode="acc", dt=0.01):
        # --- parameters --- #
        dof = 7
        nr_obst = 0
        dim_task = 2
        # mode = "vel"
        # dt = 0.01
        # init_pos = np.zeros((dof,))
        # goal_pos = [0.1, -0.6, 0.4]
        scaling_room = {"x": [-3, 3], "y":[-3, 3]}

        planner = self.set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)
        # planner.export_as_c("planner.c")
        action = np.zeros(dof)
        ob, *_ = env.step(action)
        env.add_collision_link(0, 3, shape_type='sphere', size=[0.10])
        env.add_collision_link(0, 4, shape_type='sphere', size=[0.10])
        env.add_collision_link(0, 7, shape_type='sphere', size=[0.10])

        # construct symbolic pull-back of Imitation learning geometry:
        # forward_kinematics = set_forward_kinematics()
        fk = planner.get_forward_kinematics("panda_link9")
        geometry_safeMP = construct_IL_geometry(planner=planner, dof=dof, dimension_task=dim_task, forwardkinematics=fk,
                                                variables=planner.variables, first_dim=0)
        # kinematics and IK functions
        x_function, xdot_function = geometry_safeMP.fk_functions()

        # create pulled function in configuration space
        h_function = geometry_safeMP.create_function_h_pulled()
        geometry_safeMP.set_limits(v_min=-50*np.ones((dof,)), v_max=50*np.ones((dof,)), acc_min=-50*np.ones((dof,)), acc_max=50*np.ones((dof,)))

        # Parameters
        params_name = '2nd_order_2D'
        q_init = ob['robot_0']["joint_state"]["position"]
        qdot_init = ob['robot_0']["joint_state"]["velocity"]
        x_t_init = np.array([np.append(x_function(q_init)[0:dim_task], xdot_function(q_init, qdot_init)[0:dim_task])])
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
        goal_normalized = normalizations.call_normalize_state(state=state_goal[0:dim_task])
        translation = normalizations.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN)
        translation_gpu = torch.FloatTensor(translation).cuda()

        # Initialize dynamical system
        min_vel = learner.min_vel
        max_vel = learner.max_vel
        x_init_gpu, x_init_cpu = normalizations.transformation_to_NN(x_t=x_t_init, translation_gpu=translation_gpu,
                                                      dt=dt, min_vel=min_vel, max_vel=max_vel)
        dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

        # Initialize trajectory plotter
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        trajectory_plotter = TrajectoryPlotter(fig, x0=x_init_cpu, pause_time=1e-5, goal=data['goals training'][0])

        # Initialize lists
        q_list = np.zeros((dof, n_steps))
        x_list = np.zeros((dim_task, n_steps))

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"]
            qdot = ob_robot["joint_state"]["velocity"]
            q_list[:, w] = q

            # --- End-effector state ---#
            xyz_ee = x_function(q).full().transpose()[0]
            x_ee = xyz_ee[0:dim_task]
            x_list[:, w] = x_ee
            xdot_ee = xdot_function(q, qdot).full().transpose()[0]
            x_t = np.array([np.append(x_ee, xdot_ee)])

            # --- translate to axis system of NN ---#
            x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                           dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)

            # --- get action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]
            # denormalized position of NN, used as goal for fabrics
            pos_safeMP = normalizations.reverse_transformation_pos(x_t_NN[0][0:2])
            xyz_safeMP = np.array([pos_safeMP[0], pos_safeMP[1], xyz_ee[2]])

            # --- action fabrics --- #
            arguments_dict = dict(
                q=ob_robot["joint_state"]["position"],
                qdot=ob_robot["joint_state"]["velocity"],
                x_goal_0=xyz_safeMP, #ob_robot['FullSensor']['goals'][2+nr_obst]['position'],  #xyz_safeMP
                weight_goal_0=10, #ob_robot['FullSensor']['goals'][2+nr_obst]['weight'],
                x_goal_1=ob_robot['FullSensor']['goals'][3+nr_obst]['position'],
                weight_goal_1=ob_robot['FullSensor']['goals'][3+nr_obst]['weight'],
                # x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
                # radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
                # x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
                # radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
                radius_body_links={3: 0.1, 4: 0.1, 9: 0.1, 7: 0.1},
                constraint_0=np.array([0, 0, 1, 0.0])
            )
            action_fabrics = planner.compute_action(**arguments_dict)

            ob, *_ = env.step(action_fabrics)

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
        plt.savefig(params.results_path+"images/panda_hierarchical_plot_2D")
        make_plots = plotting_functions(results_path=params.results_path)
        make_plots.plotting_x_values(x_list, dt=dt, x_start=x_list[:, 0], x_goal=np.array(goal_pos), scaling_room=scaling_room)
        env.close()
        return {}


if __name__ == "__main__":
    render = True
    dof = 7
    mode = "vel"
    dt = 0.01
    nr_obst = 0
    init_pos = np.zeros((dof,))
    goal_pos = [0.1, -0.6, 0.4]

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initialize_environment_panda(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos, nr_obst = 0)
    example_class = example_panda_stableMP_fabrics_2D_2nd_hierarchical()
    res = example_class.run_panda_example(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode)
