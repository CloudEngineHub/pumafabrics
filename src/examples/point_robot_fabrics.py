import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import matplotlib.pyplot as plt

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.

def initalize_environment(render):
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
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    full_sensor = FullSensor(
        goal_mask=["position", "weight"],
        obstacle_mask=["position", "size"],
        variance=0.0,
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [2.0, 0.0, 0.0], "radius": 1.0},
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
            "desired_position": [3.5, 0.5],
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


def set_planner(goal: GoalComposition, ONLY_GOAL=False, bool_speed_control=True):
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
    with open(absolute_path + "/urdfs/point_robot.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="world",
        end_link="base_link",
    )
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
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
    planner.concretize(mode='acc', time_step=0.01, extensive_concretize=True, bool_speed_control=bool_speed_control)
    return planner


def combine_action(M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot = []):
    xddot_combined = -np.dot(planner.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + xddot_speed
    if planner._mode == "vel":
        action_combined = qdot + planner._time_step * xddot_combined
    else:
        action_combined = xddot_combined
    return action_combined

def simple_plot(list_fabrics_goal, list_fabrics_avoidance, list_goal_Mf, list_avoidance_Mf, dt=0.01):
    time_x = np.arange(0.0, len(list_fabrics_goal)*dt, dt)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(time_x, list_fabrics_goal)
    ax[0].plot(time_x, list_goal_Mf, '--')
    ax[1].plot(time_x, list_fabrics_avoidance, '-')
    ax[1].plot(time_x, list_avoidance_Mf, "--")
    # ax[2].plot(time_x, list_diff, '-')
    # ax[2].plot()
    ax[0].grid()
    ax[1].grid()
    # ax[2].grid()
    ax[0].set(xlabel="time (s)", ylabel="x [m]", title="Goal actions")
    ax[1].set(xlabel="time (s)", ylabel="x [m]", title="Avoidance actions")
    # ax[2].set(xlabel="time (s)", ylabel="x [m]", title="Action difference")
    ax[0].legend(["x", "y", "z", "$x_{IL}$", "$y_{IL}$", "$z_{IL}$"])
    ax[1].legend(["x", "y", "z"])
    # ax[2].legend(["x", "y", "z"])
    plt.savefig("difference_fabrics_safeMP.png")

def run_point_robot_urdf(n_steps=10000, render=True):
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
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal, bool_speed_control=True)
    planner_goal = set_planner(goal=goal, ONLY_GOAL=True, bool_speed_control=True)
    planner_avoidance = set_planner(goal=None, bool_speed_control=True)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)

    list_fabrics_goal = []
    list_fabrics_avoidance =[]
    list_fabrics_goal_Mf = []
    list_fabrics_avoidance_Mf = []
    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][3]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][3]['size'],
            radius_body_base_link=np.array([0.2])
        )
        action = planner.compute_action(**arguments_dict)
        M_num, f_num, action_num, xddot_speed = planner.compute_M_f_action(**arguments_dict)
        # 2 separate planners (avoidance + goal reaching):
        M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = planner_avoidance.compute_M_f_action_avoidance(**arguments_dict)
        M_attractor, f_attractor, action_attractor, xddot_speed_attractor = planner_goal.compute_M_f_action_attractor(**arguments_dict)
        action_combined = combine_action(M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot=ob_robot["joint_state"]["velocity"])
        print("Difference all in 1 planner or two separate planners (avoidance + attractor):  ", action - action_combined)
        print("Difference with and without speed control term:  ",
              action_avoidance - -np.dot(planner.Minv(M_avoidance), f_avoidance))
        ob, *_, = env.step(action)

        # --- save in lists --- #
        list_fabrics_goal.append(action_attractor)
        list_fabrics_avoidance.append(action_avoidance)
        list_fabrics_goal_Mf.append(-np.dot(planner.Minv(M_attractor), f_attractor)) # + xddot_speed
        list_fabrics_avoidance_Mf.append(-np.dot(planner.Minv(M_avoidance), f_avoidance))
    env.close()
    simple_plot(list_fabrics_goal, list_fabrics_avoidance, list_fabrics_goal_Mf, list_fabrics_avoidance_Mf)
    return {}


if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=3000, render=False)