import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from geometry_IL import construct_IL_geometry
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from agent.utils.normalizations import normalizaton_sim_NN

from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework

def initalize_environment(render=True, mode="acc", dt=0.01, init_pos = np.zeros((7,)), goal_pos = [0.1, -0.6, 0.4]):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf="panda.urdf", mode=mode),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=dt, robots=robots, render=render
    )
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.5, -0.3, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": goal_pos,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 5.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.1, 0.0, 0.0],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = (obst1, obst2)
    env.reset()
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7, mode="acc", dt=0.01, bool_speed_control=True):
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
        number_obstacles=2,
        number_plane_constraints=1,
        limits=panda_limits,
    )
    planner.concretize(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
    return planner

def combine_action(M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot = []):
    xddot_combined = -np.dot(planner.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + xddot_speed
    if planner._mode == "vel":
        action_combined = qdot + planner._time_step * xddot_combined
    else:
        action_combined = xddot_combined
    return action_combined

def plotting_x_values(x_list, dt=0.01, x_start=np.array([0, 0]), x_goal=np.array([0, 0]), scaling_room=dict):
    time_x = np.arange(0.0, len(x_list)*dt, dt)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_list[0, :], x_list[1, :], '--', color="b")
    ax.plot(x_start[0], x_start[1], "o", color='g')
    ax.plot(x_goal[0], x_goal[1], "x", color='r')
    ax.grid()
    ax.set_xlim(scaling_room["x"][0], scaling_room["x"][1])
    ax.set_ylim(scaling_room["y"][0], scaling_room["y"][1])
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Configurations q")
    ax.legend(["x", "start", "end"])
    plt.savefig("planar_states_simulation.png")

def run_panda_example(n_steps=5000, render=True):
    # --- parameters --- #
    dof = 7
    dim_task = 2
    mode = "vel"
    dt = 0.01
    init_pos = np.zeros((dof,))
    goal_pos = [0.1, -0.6, 0.4]
    # scaling_factor = 1
    scaling_room = {"x": [-3, 3], "y":[-3, 3]}
    if mode == "vel":
        str_mode = "velocity"
    elif mode == "acc":
        str_mode = "acceleration"
    else:
        print("this control mode is not defined")

    (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
    planner = set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)
    planner_avoidance = set_planner(goal=None, bool_speed_control=True, mode=mode, dt=dt)
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
    params_name = '1st_order_2D'
    q_init = ob['robot_0']["joint_state"]["position"]
    qdot_init = ob['robot_0']["joint_state"]["velocity"]
    if params_name[0] == '2':
        x_t_init = np.array([np.append(x_function(q_init)[0:dim_task], xdot_function(q_init, qdot_init)[0:dim_task])])
    else:
        x_t_init = np.array(x_function(q_init)[0:dim_task])
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
    list_diff = []
    list_fabr_goal = []
    list_fabr_avoidance = []
    list_safeMP = []
    q_list = np.zeros((dof, n_steps))
    x_list = np.zeros((dim_task, n_steps))
    index_min = 4
    xdot_list = np.zeros((dim_task, n_steps+index_min))

    for w in range(n_steps):
        # --- state from observation --- #
        ob_robot = ob['robot_0']
        q = ob_robot["joint_state"]["position"]
        qdot = ob_robot["joint_state"]["velocity"]
        q_list[:, w] = q

        # --- End-effector state ---#
        x_ee = x_function(q).full().transpose()[0][0:dim_task]
        x_list[:, w] = x_ee
        xdot_ee = xdot_function(q, qdot).full().transpose()[0]
        xdot_list[:, w+index_min] = xdot_ee
        if params_name[0] == '2':
            x_t = np.array([np.append(x_ee, xdot_ee)])
        else:
            x_t = np.array(x_ee)

        # --- translate to axis system of NN ---#
        x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                       dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)

        # --- get action by NN --- #
        transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
        x_t_NN = transition_info["desired state"]
        xdot_prev = torch.FloatTensor(xdot_list[:, w]).cuda()
        if mode == "acc":
            action_t_gpu = (transition_info["phi"]-xdot_prev)/(dt*index_min)
            xddot_t_NN = action_t_gpu
        else:
            action_t_gpu = transition_info["desired "+str_mode]
            xddot_t_NN = (transition_info["phi"]-xdot_prev)/(dt*index_min)
        action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, dt=dt)
        xddot_safeMP = normalizations.reverse_transformation(action_gpu=xddot_t_NN, dt=dt)

        # -- transform to configuration space --#
        qddot_safeMP = geometry_safeMP.get_numerical_h_pulled(q_num=q, qdot_num=qdot, h_NN=xddot_safeMP[0:dim_task])
        action_safeMP_pulled = geometry_safeMP.get_action_safeMP_pulled(qdot, qddot_safeMP, mode=mode, dt=dt)

        # --- action fabrics --- #
        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][4]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][4]['weight'],
            x_goal_1=ob_robot['FullSensor']['goals'][5]['position'],
            weight_goal_1=ob_robot['FullSensor']['goals'][5]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
            radius_body_links={3: 0.1, 4: 0.1, 9: 0.1, 7: 0.1},
            constraint_0=np.array([0, 0, 1, 0.0]))
        action_fabrics = planner.compute_action(**arguments_dict)
        M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = planner_avoidance.compute_M_f_action_avoidance(
            **arguments_dict)
        M_safeMP = np.identity(dof,)
        qddot_speed = np.zeros((dof,)) #todo: think about what to do with speed regulation term!!
        action_fabrics_safeMP = combine_action(M_avoidance, M_safeMP, f_avoidance, -qddot_safeMP[0:dof], qddot_speed, planner,
                                               qdot=ob_robot["joint_state"]["velocity"][0:dof])


        ob, *_ = env.step(action_fabrics_safeMP) #action_safeMP_pulled

        # --- Update plot ---#
        trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
    plt.savefig("planar_robot_safeMP_plot")
    plotting_x_values(x_list, dt=dt, x_start=x_list[:, 0], x_goal=np.array(goal_pos), scaling_room=scaling_room)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_example(n_steps=5000)
