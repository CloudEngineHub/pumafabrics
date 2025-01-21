import numpy as np
from copy import deepcopy
import gymnasium as gym
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

class trial_environments():
    def __init__(self):
        dt = 0.01

    def initalize_environment_pointmass(self, render, mode="acc", dt=0.01, init_pos=np.array([-0.9, -0.1, 0.0]), goal_pos=[3.5, 0.5]):
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
            "geometry": {"position": [0.0, -6.6, 0.0], "radius": 0.7},
            "rgba": [1.0, 0.0, 0.0, 1.0],
        }
        obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [-2.0, -4.0, 0.0], "radius": 1.0}, #-2.0, 2.0 [-2.0, -5.0
            "rgba": [0.0, 0.0, 1.0, 1.0],
        }
        obst2 = SphereObstacle(name="staticObst2", content_dict=static_obst_dict)
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [5.0, -3.0, 0.0], "radius": 1.0},
            "rgba": [0.0, 0.0, 1.0, 1.0],
        }
        obst3 = SphereObstacle(name="staticObst3", content_dict=static_obst_dict)
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
        obstacles = (obst1, obst2) #, obst3)
        for obst in obstacles:
            env.add_obstacle(obst)
        env.set_spaces()
        return (env, goal)

    def initialize_environment_planar(self, render=True, mode="acc", dt=0.01, init_pos=np.array([0.1, 0.1]), goal_pos=[1.2, 1.4], nr_obst=0):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        steps the simulation once.
        """
        robots = [
            GenericUrdfReacher(urdf="examples/urdfs/planar_urdf_2_joints.urdf", mode=mode),
        ]
        env: UrdfEnv = gym.make(
            "urdf-env-v0",
            dt=dt, robots=robots, render=render
        )
        full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
        )
        #Definition of the obstacle.
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [0.0, -0.9, 0.3], "radius": 0.1},
        }
        obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [-0.0, 1.2, 1.4], "radius": 0.1},
        }
        obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_link4",
                "desired_position": goal_pos,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
        }
        visualize_goal_dict = deepcopy(goal_dict)
        visualize_goal_dict['subgoal0']['indices'] = [0] + goal_dict['subgoal0']['indices']
        visualize_goal_dict['subgoal0']['desired_position'] = [0.0] + goal_dict['subgoal0']['desired_position']
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        vis_goal = GoalComposition(name="goal", content_dict=visualize_goal_dict)
        if nr_obst == 1:
            obstacles = [obst1]
        elif nr_obst == 2:
            obstacles = [obst1, obst2]
        else:
            obstacles = []
        pos0 = init_pos
        env.reset(pos=pos0)
        env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            env.add_obstacle(obst)
        for sub_goal in vis_goal.sub_goals():
            env.add_goal(sub_goal)
        env.set_spaces()
        return (env, goal)

    def initialize_environment_panda(self, render=True, mode="acc", dt=0.01, init_pos=np.zeros((7,)), goal_pos=[0.1, -0.6, 0.4], nr_obst=0):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        steps the simulation once.
        """
        robots = [
            GenericUrdfReacher(urdf="panda.urdf", mode=mode),
        ]
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
        if nr_obst == 1:
            obstacles = [obst1]
        elif nr_obst == 2:
            obstacles = [obst1, obst2]
        else:
            obstacles = []
        env.reset()
        env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            env.add_obstacle(obst)
        for sub_goal in goal.sub_goals():
            env.add_goal(sub_goal)
        env.set_spaces()
        return (env, goal)


    def initialize_environment_kuka(self, params):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        steps the simulation once.
        """
        robot_name = params["robot_name"]
        render = params["render"]
        mode = params["mode_env"]
        dt = params["dt"]
        init_pos = params["init_pos"]
        goal_pos = params["goal_pos"]
        end_effector_link = params["end_links"][0]
        nr_obst = params["nr_obst"]
        nr_dyn_obst = params["nr_obst_dyn"]
        positions_obstacles = params["positions_obstacles"]
        speed_obstacles = params["speed_obstacles"]

        robots = [
            GenericUrdfReacher(urdf="../pumafabrics/tamed_puma/config/urdfs/"+robot_name+".urdf", mode=mode),
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
        static_obst_dict = {
            "type": "sphere",
            # "geometry": {"position": positions_obstacles[0], "radius": 0.05},
            "geometry": {"trajectory": [str(positions_obstacles[0][0]) + "+t*" + str(speed_obstacles[0][0]),
                                        str(positions_obstacles[0][1]) + "+t*" + str(speed_obstacles[0][1]),
                                        str(positions_obstacles[0][2]) + "+t*" + str(speed_obstacles[0][2])], "radius": 0.05},
            "rgba": [1, 0, 0, 1]
            # todo: IMPORTANT when z=0.5: fabrics becomes unstable/local minima
        }
        obst1 = DynamicSphereObstacle(name="staticObst", content_dict=static_obst_dict)
        if nr_obst>1:
            static_obst_dict = {
                "type": "sphere",
                "geometry": {"position": positions_obstacles[1], "radius": 0.05},
                "rgba": [1, 0, 0, 1]
            }
            obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
        dynamic_obst_dict = {
            "type": "sphere",
            "geometry": {"trajectory": ["-1 + t * 0.1", "-0.6", "0.4"], "radius": 0.05},
        }
        obst0_dyn = DynamicSphereObstacle(name="dynamicObst", content_dict=dynamic_obst_dict)
            #Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "iiwa_link_0",
                "child_link": end_effector_link,
                "desired_position": goal_pos,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 1.,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "iiwa_link_7",
                "child_link": "iiwa_link_ee_x",
                "desired_position": [0.045, 0., 0.],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal2": {
                "weight": 1.,
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
        if nr_dyn_obst == 1:
            obstacles.append(obst0_dyn)
        pos0 = init_pos
        env.reset(pos=np.array(pos0))
        env.add_sensor(full_sensor, [0])
        for obst in obstacles:
            env.add_obstacle(obst)
        for sub_goal in goal.sub_goals():
            env.add_goal(sub_goal)
        env.set_spaces()

        # --- Camera angle ---- #
        env.reconfigure_camera(1.5, 70., -20., (0., 0., 0.5))
        if params["collision_links"] is not None:
            for i_link, link in enumerate(params["collision_links"]):
                env.add_collision_link(
                    robot_index=0,
                    link_index=link,
                    shape_type="sphere",
                    sphere_on_link_index=0,
                    # link_transformation=link_transformation,
                    size=[list(params["collision_radii"].values())[i_link]],#[dict_collision_links['radi'][dict_collision_links['link_nr_urdf'].index(link)]]
                )
        return (env, goal)