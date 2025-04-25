from mpscenes.goals.goal_composition import GoalComposition
def goal_default(robot_name="panda", goal_pos=None, end_effector_link=None, joint_configuration=None):
    if robot_name == "pointrobot":
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
    elif robot_name == "planar":
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
    elif robot_name == "panda":
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
    elif robot_name[0:4] == "iiwa":
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
    elif robot_name[0:8] == "gen3lite":
        goal_dict = {
            "subgoal0": {
                "weight": 3.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "kinova_base_link",
                "child_link": end_effector_link,
                "desired_position": goal_pos,
                "epsilon": 0.03,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 5.,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "dummy_link",
                "child_link": "tool_frame",
                "desired_position": [0.0, 0., 0.13],
                "epsilon": 0.03,
                "type": "staticSubGoal",
            },
            "subgoal2": {
                "weight": 5.,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "dummy_link",
                "child_link": "orientation_helper_link",
                "desired_position": [0.0, 0.10, 0.0],
                "epsilon": 0.03,
                "type": "staticSubGoal",
            },
        }
    elif robot_name[0:6] == "dinova":
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "base_link",
                "child_link": end_effector_link,
                "desired_position": goal_pos,
                "epsilon": 0.03,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 0.,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "arm_dummy_link",
                "child_link": end_effector_link,
                "desired_position": [0., 0., 0.07],
                "epsilon": 0.03,
                "type": "staticSubGoal",
            },
            "subgoal2": {
                "weight": 0.,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "arm_dummy_link",
                "child_link": "arm_orientation_helper_link",
                "desired_position": [0.0, 0.10, 0.0],
                "epsilon": 0.015,
                "type": "staticSubGoal",
            },
            "subgoal3": {
                "weight": 1.0,
                "is_primary_goal": False,
                "indices": [2],
                "desired_position": joint_configuration[0:1],
                "epsilon": 0.015,
                "type": "staticJointSpaceSubGoal",
            }
        }
    else:
        print("please provide a valid robot name!, defaulting to pointmass robot")
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
    return goal