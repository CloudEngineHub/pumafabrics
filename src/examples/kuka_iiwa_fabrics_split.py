import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

"""
Experiments with Kuka iiwa
If 
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.3, -0.3, 0.5], "radius": 0.1},
    }
for the first obstacle, fabrics ends up in a local minima and is unable to recover from it. 
This would be a good case to improve with imitation learning.
"""

def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf="urdfs/iiwa7.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.3, -0.3, 0.3], "radius": 0.1}, #todo: IMPORTANT when z=0.5: fabrics becomes unstable/local minima
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
            "parent_link": "iiwa_link_0",
            "child_link": "iiwa_link_ee",
            "desired_position": [0.1, -0.6, 0.4],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        # "subgoal1": {
        #     "weight": 5.0,
        #     "is_primary_goal": False,
        #     "indices": [0, 1, 2],
        #     "parent_link": "iiwa7",
        #     "child_link": "panda_hand",
        #     "desired_position": [0.1, 0.0, 0.0],
        #     "epsilon": 0.05,
        #     "type": "staticSubGoal",
        # }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = (obst1, obst2)
    pos0 = np.array([0.0, 0.8, -1.5, 2.0, 0.0, 0.0, 0.0])
    env.reset(pos=pos0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7, ONLY_GOAL=False, bool_speed_control=True):
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

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "20 * ca.norm_2(x)**4"
    # damper = {
    #     "alpha_b": 0.5,
    #     "alpha_eta": 0.5,
    #     "alpha_shift": 0.5,
    #     "beta_distant": 0.01,
    #     "beta_close": 6.5,
    #     "radius_shift": 0.1,
    # }
    # planner = ParameterizedFabricPlanner(
    #     degrees_of_freedom,
    #     forward_kinematics,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/urdfs/iiwa7.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="iiwa_link_0",
        end_link="iiwa_link_ee",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    collision_links = ["iiwa_link_3", "iiwa_link_4", "iiwa_link_5", "iiwa_link_6", "iiwa_link_7"]
    iiwa_limits = [
        [-2.96705973, 2.96705973],
        [-2.0943951, 2.0943951],
        [-2.96705973, 2.96705973],
        [-2.0943951, 2.0943951],
        [-2.96705973, 2.96705973],
        [-2.0943951, 2.0943951],
        [-3.05432619, 3.05432619],
    ]
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
            number_plane_constraints=1,
            limits=iiwa_limits,
        )
    planner.concretize(extensive_concretize=True, bool_speed_control=bool_speed_control)
    return planner


def run_panda_example(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal=goal, bool_speed_control=True)
    planner_goal = set_planner(goal=goal, ONLY_GOAL=True, bool_speed_control=True)
    planner_avoidance = set_planner(goal=None, bool_speed_control=True)
    # planner.export_as_c("planner.c")
    action = np.zeros(7)
    ob, *_ = env.step(action)
    collision_radii = {3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}
    # collision_links_nrs = [3, 4, 5, 6, 7]
    for collision_link_nr in collision_radii.keys():
        env.add_collision_link(0, collision_link_nr, shape_type='sphere', size=[0.10])


    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][4]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][4]['weight'],
            # x_goal_1=ob_robot['FullSensor']['goals'][5]['position'],
            # weight_goal_1=ob_robot['FullSensor']['goals'][5]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
            radius_body_links=collision_radii,
            constraint_0=np.array([0, 0, 1, 0.0]))

        # M_num, f_num, action = planner.compute_M_f_action(**arguments_dict)
        # M_av, f_av, action_av = planner.compute_M_f_action_avoidance(**arguments_dict)
        # M_att, f_att, action_att = planner.compute_M_f_action_attractor(**arguments_dict)

        # 1 planner including all components:
        action = planner.compute_action(**arguments_dict)
        M_num, f_num, action_num, xddot_speed = planner.compute_M_f_action(**arguments_dict)
        action_speed = action_num + xddot_speed
        print("Difference action normal (speed controlled) and where speed control is added later on: ", action - action_speed)

        # 2 separate planners (avoidance + goal reaching):
        M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = planner_avoidance.compute_M_f_action_avoidance(**arguments_dict)
        M_attractor, f_attractor, action_attractor, xddot_speed_attractor = planner_goal.compute_M_f_action_attractor(**arguments_dict)
        action_combined = -np.dot(planner.Minv(M_avoidance+M_attractor), f_avoidance+f_attractor) + xddot_speed
        print("Difference speed control term in 1 planner or split planner: ", xddot_speed - (xddot_speed_avoidance + xddot_speed_attractor))
        # print("Difference avoidance separate or avoidance 1 planner: ", action_av - action_avoidance)
        # print("Difference attractor separate or attractor 1 planner: ", action_att - action_attractor)
        # print("Difference all in 1 planner or two separate planners (avoidance + attractor):  ", action - action_combined)
        ob, *_ = env.step(action_combined)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_example(n_steps=5000)
