import os
import gymnasium as gym
import numpy as np
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
from agent.utils.normalizations import normalizaton_sim_NN
from src.functions_stableMP_fabrics.geometry_IL import construct_IL_geometry
from functions_stableMP_fabrics.plotting_functions import plotting_functions

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.

def initalize_environment(render, mode="acc", dt=0.01, init_pos = np.array([-0.9, -0.1, 0.0]), goal_pos = [3.5, 0.5]):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.

    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode=mode),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=dt, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = init_pos
    vel0 = np.array([0.0, 0.0, 0.0])
    full_sensor = FullSensor(
        goal_mask=["position", "weight"],
        obstacle_mask=["position", "size"],
        variance=0.0,
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [10.0, 0.0, 0.0], "radius": 1.0},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 0.5,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 'world',
            "child_link": 'base_link_y',
            "desired_position": goal_pos,
            "epsilon": 0.1,
            "type": "staticSubGoal"
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_goal(goal.sub_goals()[0])
    env.add_obstacle(obst1)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, ONLY_GOAL=False, bool_speed_control=True, mode="acc", dt=0.01):
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
            number_obstacles=1,
        )
    # planner.concretize(extensive_concretize=True, bool_speed_control=bool_speed_control)
    planner.concretize_extensive(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
    return planner

def run_point_robot_urdf(n_steps=2000, render=True):
    # --- parameters --- #
    dof = 2
    dim_task = 2
    mode = "acc"
    mode_NN = "2nd"
    dt = 0.01
    init_pos = np.array([-0.1, -0.7, 0.0])
    goal_pos = [1.2, 1.4]
    # scaling_factor = 1
    scaling_room = {"x": [-10, 10], "y":[-10, 10]}
    if mode == "acc":
        str_mode = "acceleration"
    elif mode == "vel":
        str_mode = "velocity"
    else:
        print("this control mode is not defined")

    # initialize environment and planner
    (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
    planner = set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)
    planner.set_base_geometry()
    action = np.zeros(env.n())
    ob, *_ = env.step(action)

    # construct symbolic pull-back of Imitation learning geometry:
    # forward_kinematics = set_forward_kinematics()
    fk = planner.get_forward_kinematics("base_link_y")
    geometry_safeMP = construct_IL_geometry(planner=planner, dof=dof, dimension_task=dim_task, forwardkinematics=fk,
                                            variables=planner.variables, first_dim=0)
    # kinematics and IK functions
    x_function, xdot_function = geometry_safeMP.fk_functions()

    # create pulled function in configuration space
    h_function = geometry_safeMP.create_function_h_pulled()
    # geometry_safeMP.set_limits(v_min=np.array([-10, -10]), v_max=np.array([10, 10]), acc_min=np.array([-10, -10]), acc_max=np.array([10, 10]))

    # Parameters
    params_name = '2nd_order_2D'
    q_init = ob['robot_0']["joint_state"]["position"]
    qdot_init = ob['robot_0']["joint_state"]["velocity"]
    x_t_init = np.array([np.append(x_function(q_init[0:dof])[0:dof], xdot_function(q_init[0:dof], qdot_init[0:dof])[0:dof])])
    # x_t_init = np.array([np.append(ob['robot_0']["joint_state"]["position"][0:2], ob['robot_0']["joint_state"]["velocity"][0:2])]) # initial states
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
    min_vel = torch.FloatTensor([[-0.0137, -0.0231]]).cuda() #todo: must not be hardcoded
    max_vel = torch.FloatTensor([[0.0272, 0.0169]]).cuda()
    x_init_gpu, x_init_cpu = normalizations.transformation_to_NN(x_t=x_t_init, translation_gpu=translation_gpu,
                                                  dt=dt, min_vel=min_vel, max_vel=max_vel)
    dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

    # Initialize trajectory plotter
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    fig.show()
    trajectory_plotter = TrajectoryPlotter(fig, x0=x_init_cpu, pause_time=1e-5, goal=data['goals training'][0])

    # Initialize lists
    list_diff = []
    list_fabr_goal = []
    list_fabr_avoidance = []
    list_safeMP = []
    q_list = np.zeros((dof, n_steps))
    x_list = np.zeros((dim_task, n_steps))

    for w in range(n_steps):
        # --- state from observation --- #
        ob_robot = ob['robot_0']
        q = ob_robot["joint_state"]["position"]
        qdot = ob_robot["joint_state"]["velocity"]
        q_list[:, w] = q[0:dof]

        # --- End-effector state ---#
        x_ee = x_function(q[0:dof]).full().transpose()[0][0:dof]
        x_list[:, w] = x_ee
        xdot_ee = xdot_function(q[0:dof], qdot[0:dof]).full().transpose()[0]
        x_t = np.array([np.append(x_ee, xdot_ee)])

        # --- translate to axis system of NN ---#
        x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                       dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)

        # --- get action by NN --- #
        transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
        x_t_NN = transition_info["desired state"]
        if mode == "acc":
            action_t_gpu = transition_info["phi"]
            xddot_t_NN = transition_info["phi"]
        else:
            action_t_gpu = transition_info["desired " + str_mode]
            xddot_t_NN = transition_info["phi"]
        action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, dt=dt, mode_NN=mode_NN)
        xddot_safeMP = normalizations.reverse_transformation(action_gpu=xddot_t_NN, dt=dt, mode_NN="2nd")

        # -- transform to configuration space --#
        qddot_safeMP = geometry_safeMP.get_numerical_h_pulled(q_num=q[0:dof], qdot_num=qdot[0:dof], h_NN=xddot_safeMP[0:dim_task])
        # qddot_safeMP = -1*qddot_safeMP #todo: check if it should be reversed
        action_safeMP_pulled = geometry_safeMP.get_action_safeMP_pulled(qdot[0:dof], qddot_safeMP[0:dof], mode=mode, dt=dt)

        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"][0:dof],
            qdot=ob_robot["joint_state"]["velocity"][0:dof],
            x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:dof],
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][3]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][3]['size'],
            radius_body_base_link_y=np.array([0.2])
        )
        action = planner.compute_action(**arguments_dict)
        print("action: ", action)
        print("action safeMP task: ", action_safeMP)
        print("action safeMP pulled: ", action_safeMP_pulled)
        ob, *_ = env.step(np.append(action_safeMP, 0.0))

        # --- Update plot ---#
        trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
    plt.savefig(params.results_path+"images/point_robot_pulled")
    make_plots = plotting_functions(results_path=params.results_path)
    make_plots.plotting_x_values(x_list, dt=dt, x_start=x_list[:, 0], x_goal=np.array(goal_pos),
                                 scaling_room=scaling_room)
    env.close()
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=100, render=True)