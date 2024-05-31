import numpy as np
from copy import deepcopy
import gymnasium as gym
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.goals.goal_composition import GoalComposition

class PybulletEnvironments():
    def __init__(self):
        dt = 0.01
    def initialize_environment_kuka(self, render=True, mode="acc",
                                    dt=0.01, init_pos=np.zeros((7,)),
                                    goal_pos=[0.1, -0.6, 0.4],
                                    nr_obst=0,
                                    obst0_pos = [0.5, -0.25, 0.5],
                                    obst1_pos = [0.24355761, 0.45252747, 0.2],
                                    end_effector_link="iiwa_link_ee", robot_name="iiwa7"):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        steps the simulation once.
        """
        robots = [
            GenericUrdfReacher(urdf="urdfs/" + robot_name + ".urdf", mode=mode),
        ]
        link_id = robots[0]._urdf_joints
        env: UrdfEnv = gym.make(
            "urdf-env-v0",
            dt=dt, robots=robots, render=render
        )
        full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
        )
        # Definition of the obstacle.
        # static_obst_dict = {
        #     "type": "sphere",
        #     "geometry": {"position": obst0_pos, "radius": 0.1},
        #     "rgba": [1, 0, 0, 1]
        #     # todo: IMPORTANT when z=0.5: fabrics becomes unstable/local minima
        # }
        static_obst_dict = {
            "type": "box",
            "geometry": {"position": obst0_pos,
                         "length":0.1,
                         "height":1.0,
                         "width":0.1},
            "rgba": [0.5, 0.5, 0.9, 1.0]
            # todo: IMPORTANT when z=0.5: fabrics becomes unstable/local minima
        }
        obst1 = BoxObstacle(name="staticObst", content_dict=static_obst_dict)
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": obst1_pos, "radius": 0.1},
            # [-0.24355761, -0.75252747, 0.5], #[0.04355761, -0.75252747, 0.5]
            "rgba": [1, 0, 0, 1]
        }
        obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "iiwa_link_0",
                "child_link": end_effector_link,
                "desired_position": goal_pos,  # [0.1, -0.6, 0.4],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 10.,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "iiwa_link_7",
                "child_link": "iiwa_link_ee_x",
                "desired_position": [0.045, 0., 0.],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal2": {
                "weight": 10.,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "iiwa_link_7",
                "child_link": end_effector_link,
                "desired_position": [0.0, 0.0, 0.045],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
        }
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        if nr_obst == 1:
            obstacles = [obst1]
        elif nr_obst == 2:
            obstacles = [obst1, obst2]
        else:
            obstacles = []
        pos0 = init_pos  # np.array(np.array([0.002, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010]) )
        env.reset(pos=pos0)  # , vel=np.array(np.array([1., 1., 1., 1., 1., 1., 0.5])))
        env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            env.add_obstacle(obst)
        for sub_goal in goal.sub_goals():
            env.add_goal(sub_goal)
        env.set_spaces()

        # --- Camera angle ---- #
        env.reconfigure_camera(1.5, 70., -20., (0., 0., 0.5))
        # env.reconfigure_camera(2.5, -5., -42., (0.3, 1., -0.5))
        return (env, goal)