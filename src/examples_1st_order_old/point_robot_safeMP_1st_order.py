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

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
from agent.utils.normalizations import normalizaton_sim_NN

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
            "child_link": 'base_link',
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
    degrees_of_freedom = 3
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/examples/urdfs/point_robot.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="world",
        end_link="base_link",
    )
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "2.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler
    )
    collision_links = ["base_link"]
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
    planner.concretize(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
    return planner


def combine_action(M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot = []):
    xddot_combined = -np.dot(planner.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + xddot_speed
    if planner._mode == "vel":
        action_combined = qdot + planner._time_step * xddot_combined
    else:
        action_combined = xddot_combined
    return action_combined

def translation_to_NN(xy_sim, translation):
    xy_translated = xy_sim - translation
    return xy_translated

def plotting_q_values(q_list, dt=0.01, q_start=np.array([0, 0]), q_goal=np.array([0, 0]), scaling_room={"x":[-1, 1], "y":[-1, 1]}):
    time_x = np.arange(0.0, len(q_list)*dt, dt)
    fig, ax = plt.subplots(1, 1)
    ax.plot(q_list[0, :], q_list[1, :], '--', color="b")
    ax.plot(q_start[0], q_start[1], "o", color='g')
    ax.plot(q_goal[0], q_goal[1], "x", color='r')
    ax.grid()
    ax.set_xlim(scaling_room["x"][0], scaling_room["x"][1])
    ax.set_ylim(scaling_room["y"][0], scaling_room["y"][1])
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Positions q")
    ax.legend(["q", "start", "end"])
    plt.savefig("positions_simulation_trial.png")


def run_point_robot_urdf(n_steps=2000, render=True):
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
    mode = "vel"
    dim_task = 2
    dt = 0.01
    init_pos = np.array([0.5, 0.5, 0.0])
    goal_pos = [4.5, 0.5]
    scaling_room = {"x": [0, 5], "y":[0, 5]}
    if mode == "vel":
        str_mode = "velocity"
    elif mode == "acc":
        str_mode = "acceleration"
    else:
        print("this control mode is not defined")
    (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
    planner = set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)
    # planner_goal = set_planner(goal=goal, ONLY_GOAL=True, bool_speed_control=True, mode=mode, dt=dt)
    # planner_avoidance = set_planner(goal=None, bool_speed_control=True, mode=mode, dt=dt)

    action_safeMP = np.array([0.0, 0.0, 0.0])
    action_fabrics = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action_safeMP)
    q_list = np.zeros((2, n_steps))
    index_min = 4
    qdot_list = np.zeros((2, n_steps+index_min))

    # Parameters
    params_name = '1st_order_2D'
    if params_name[0] == '2':
        x_t_init = np.array([np.append(ob['robot_0']["joint_state"]["position"][0:dim_task], ob['robot_0']["joint_state"]["velocity"][0:dim_task])]) # initial states
    else:
        x_t_init = np.array(ob['robot_0']["joint_state"]["position"][0:dim_task])
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

    for w in range(n_steps):
        # --- state from observation --- #
        ob_robot = ob['robot_0']
        q = ob_robot["joint_state"]["position"][0:2]
        qdot = ob_robot["joint_state"]["velocity"][0:2]
        q_list[:, w] = q
        qdot_list[:, w+index_min] = qdot
        if params_name[0] == '2':
            x_t = np.array([np.append(q, qdot)])
        else:
            x_t = np.array(q)

        # --- translate to axis system of NN ---#
        x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                       dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)

        # --- get action by NN --- #
        transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
        x_t_NN = transition_info["desired state"]
        if mode == "acc":
            qdot_prev = torch.FloatTensor(qdot_list[:, w]).cuda()
            action_t_gpu = (transition_info["phi"]-qdot_prev)/(dt*index_min)
        else:
            action_t_gpu = transition_info["desired "+str_mode]

        action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, dt=dt, mode=mode)
        ob, *_, = env.step(action_safeMP)

        # --- Update plot ---#
        trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
    plt.savefig("point_robot_safeMP_plot2")
    env.close()
    plotting_q_values(q_list, dt=dt, q_start=q_list[:, 0], q_goal=np.array(goal_pos), scaling_room=scaling_room)
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=1000, render=True)