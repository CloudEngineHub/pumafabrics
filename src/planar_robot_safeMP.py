from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.tamedpuma.environments import trial_environments
import numpy as np
import os
from pumafabrics.tamed_puma.tamedpuma.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from pumafabrics.tamed_puma.kinematics.geometry_IL import construct_IL_geometry
from pumafabrics.tamed_puma.utils.plotting_functions import plotting_functions

from pumafabrics.puma_adapted.tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from pumafabrics.puma_adapted.initializer import initialize_framework
from pumafabrics.tamed_puma.utils.normalizations import normalizaton_sim_NN
# TODO hardcoding the indices for subgoal_1 is undesired

class example_planar_stableMP():
    def __init__(self):
        dt = 0.01
    def set_forward_kinematics(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/planar_urdf_2_joints.urdf", "r") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            rootLink="panda_link0",
            end_link="panda_link4",
        )
        return forward_kinematics

    def set_planner(self, goal: GoalComposition, degrees_of_freedom: int = 2, ONLY_GOAL=False, bool_speed_control=True, mode="acc", dt=0.01):
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
        forward_kinematics = self.set_forward_kinematics()
        planner = ParameterizedFabricPlannerExtended(
            degrees_of_freedom,
            forward_kinematics=forward_kinematics,
            time_step=dt
        )
        q = planner.variables.position_variable()
        collision_links = ['panda_link1', 'panda_link4']
        self_collision_pairs = {}
        panda_limits = [
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
            ]
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            collision_links=collision_links,
            self_collision_pairs=self_collision_pairs,
            goal=goal,
            number_obstacles=0,
            #limits=panda_limits,
        )
        planner.concretize_extensive(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
        return planner


    def run_panda_example(self, n_steps=2000, env=None, goal=None, init_pos=np.array([-0.1, -0.7]), goal_pos=[0.8, 1.4], mode="acc", mode_NN="2nd", dt=0.01):
        # --- parameters --- #
        dof = 2
        dim_task = 2
        # mode = "acc"
        # dt = 0.01
        # init_pos = np.array([-0.1, -0.7])
        # goal_pos = [0.8, 1.4]
        # scaling_factor = 1
        scaling_room = {"x": [-3, 3], "y":[-3, 3]}
        if mode == "vel":
            str_mode = "velocity"
        elif mode == "acc":
            str_mode = "acceleration"
        else:
            print("this control mode is not defined")

        # initialize environment and planner
        # (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
        planner = self.set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)
        planner.set_base_geometry()
        action = np.zeros(env.n())
        ob, *_ = env.step(action)

        # construct symbolic pull-back of Imitation learning geometry:
        # forward_kinematics = set_forward_kinematics()
        fk = planner.get_forward_kinematics("panda_link4")
        geometry_safeMP = construct_IL_geometry(planner=planner, dof=dof, dimension_task=dim_task, forwardkinematics=fk,
                                                variables=planner.variables, first_dim=1)
        # kinematics and IK functions
        x_function, xdot_function = geometry_safeMP.fk_functions()

        # create pulled function in configuration space
        h_function = geometry_safeMP.create_function_h_pulled()
        # geometry_safeMP.set_limits(v_min=np.array([-5, -5]), v_max=np.array([5, 5]), acc_min=np.array([-5, -5]), acc_max=np.array([5, 5]))

        # Parameters
        params_name = '2nd_order_2D'
        q_init = ob['robot_0']["joint_state"]["position"]
        qdot_init = ob['robot_0']["joint_state"]["velocity"]
        x_t_init = np.array([np.append(x_function(q_init)[1:3], xdot_function(q_init, qdot_init))])
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

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"]
            qdot = ob_robot["joint_state"]["velocity"]
            q_list[:, w] = q

            # --- End-effector state ---#
            x_ee = x_function(q).full().transpose()[0][1:3]
            x_list[:, w] = x_ee
            xdot_ee = xdot_function(q, qdot).full().transpose()[0]
            x_t = np.array([np.append(x_ee, xdot_ee)])

            # --- translate to axis system of NN ---#
            x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                           dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)

            # --- get action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]
            if mode == "acc":
                action_t_gpu = transition_info["phi"]
                xddot_t_NN  = transition_info["phi"]
            else:
                action_t_gpu = transition_info["desired "+str_mode]
                xddot_t_NN = transition_info["phi"]
            action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, dt=dt, mode_NN=mode_NN)
            xddot_safeMP = normalizations.reverse_transformation(action_gpu=xddot_t_NN, dt=dt, mode_NN="2nd")

            # -- transform to configuration space --#
            qddot_safeMP = geometry_safeMP.get_numerical_h_pulled(q_num=q, qdot_num=qdot, h_NN=xddot_safeMP[0:dim_task])
            # qddot_safeMP = -1*qddot_safeMP #todo: check if it should be reversed
            action_safeMP_pulled = geometry_safeMP.get_action_safeMP_pulled(qdot, qddot_safeMP, mode=mode, dt=dt)

            action = planner.compute_action(
                q=ob_robot["joint_state"]["position"],
                qdot=ob_robot["joint_state"]["velocity"],
                x_goal_0=goal.sub_goals()[0].position(),
                weight_goal_0=goal.sub_goals()[0].weight(),
                # x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
                # radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
                # x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
                # radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
                radius_body_panda_link1=0.2,
                radius_body_panda_link4=0.2,
            )
            print("action: ", action)
            print("action safeMP task: ", action_safeMP)
            print("action safeMP pulled: ", action_safeMP_pulled)
            ob, *_ = env.step(action_safeMP_pulled)

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
        plt.savefig(params.results_path+"images/planar_robot_safeMP_plot")
        make_plots = plotting_functions(results_path=params.results_path)
        make_plots.plotting_x_values(x_list, dt=dt, x_start=x_list[:, 0], x_goal=np.array(goal_pos),
                                     scaling_room=scaling_room)
        env.close()
        return {}


if __name__ == "__main__":
    render = True
    mode_NN = "2nd"
    mode = "acc"
    dt = 0.01
    init_pos = np.array([-0.1, -0.7])
    goal_pos = [0.8, 1.4]

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initialize_environment_planar(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos)
    example_class = example_planar_stableMP()

    res = example_class.run_panda_example(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                                          dt=dt, mode=mode)
