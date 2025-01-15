import os
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.kinematics.geometry_IL import construct_IL_geometry
from pumafabrics.tamed_puma.utils.plot_point_robot import plotting_functions
from functions_stableMP_fabrics.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from agent.utils.normalizations import normalizaton_sim_NN
from functions_stableMP_fabrics.environments import trial_environments
from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework

class example_kuka_stableMP_fabrics_2D_2nd:
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
        with open(absolute_path + "/examples/urdfs/iiwa7.urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            rootLink="iiwa_link_0",
            end_link="iiwa_link_ee",
        )
        planner = ParameterizedFabricPlannerExtended(
            degrees_of_freedom,
            forward_kinematics,
            time_step=dt,
        )
        collision_links = ["iiwa_link_3", "iiwa_link_4", "iiwa_link_5", "iiwa_link_6", "iiwa_link_7"]
        # iiwa_limits = [
        #     [-2.96705973, 2.96705973],
        #     [-2.0943951, 2.0943951],
        #     [-2.96705973, 2.96705973],
        #     [-2.0943951, 2.0943951],
        #     [-2.96705973, 2.96705973],
        #     [-2.0943951, 2.0943951],
        #     [-3.05432619, 3.05432619],
        # ]
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            collision_links=collision_links,
            goal=goal,
            number_obstacles=0,
            number_plane_constraints=0,
            # limits=iiwa_limits,
        )
        planner.concretize_extensive(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
        return planner


    def combine_action(self, M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot = []):
        xddot_combined = -np.dot(planner.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + xddot_speed
        if planner._mode == "vel":
            action_combined = qdot + planner._time_step * xddot_combined
        else:
            action_combined = xddot_combined
        return action_combined

    def run_kuka_example(self, n_steps=2000, env=None, goal=None, init_pos=np.array([0.0, 0.0, 0.]), goal_pos=[-0.24355761, -0.75252747, 0.5], mode="acc", dt=0.01):
        # --- parameters --- #
        dof = 7
        dim_task = 2
        # mode = "acc"
        # dt = 0.01
        # nr_obst = 0
        # init_pos = np.zeros((dof,))
        # goal_pos = [-0.24355761, -0.75252747, 0.5] # [0.1, -0.6, 0.4]
        # scaling_factor = 1
        scaling_room = {"x": [-1, 1], "y":[-1, 1]}
        if mode == "vel":
            str_mode = "velocity"
        elif mode == "acc":
            str_mode = "acceleration"
        else:
            print("this control mode is not defined")

        # (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
        planner = self.set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)
        planner_avoidance = self.set_planner(goal=None, bool_speed_control=True, mode=mode, dt=dt)
        # planner.export_as_c("planner.c")
        action = np.zeros(dof)
        ob, *_ = env.step(action)
        env.add_collision_link(0, 3, shape_type='sphere', size=[0.10])
        env.add_collision_link(0, 4, shape_type='sphere', size=[0.10])
        env.add_collision_link(0, 7, shape_type='sphere', size=[0.10])

        # construct symbolic pull-back of Imitation learning geometry:
        # forward_kinematics = set_forward_kinematics()
        fk = planner.get_forward_kinematics("iiwa_link_7")
        geometry_safeMP = construct_IL_geometry(planner=planner, dof=dof, dimension_task=dim_task, forwardkinematics=fk,
                                                variables=planner.variables, first_dim=0)
        # kinematics and IK functions
        x_function, xdot_function = geometry_safeMP.fk_functions()

        # create pulled function in configuration space
        h_function = geometry_safeMP.create_function_h_pulled()
        geometry_safeMP.set_limits(v_min=-50*np.ones((dof,)), v_max=50*np.ones((dof,)), acc_min=-50*np.ones((dof,)), acc_max=50*np.ones((dof,)))

        # Parameters
        params_name = '2nd_order_2D'
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        qdot_init = ob['robot_0']["joint_state"]["velocity"][0:dof]
        x_t_init = np.array([np.append(x_function(q_init)[0:dim_task], xdot_function(q_init, qdot_init)[0:dim_task])])
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

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            q_list[:, w] = q

            # --- End-effector state ---#
            x_ee = x_function(q).full().transpose()[0][0:dim_task]
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
            action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, dt=dt)
            xddot_safeMP = normalizations.reverse_transformation(action_gpu=xddot_t_NN, dt=dt)

            # -- transform to configuration space --#
            qddot_safeMP = geometry_safeMP.get_numerical_h_pulled(q_num=q, qdot_num=qdot, h_NN=xddot_safeMP[0:dim_task])
            action_safeMP_pulled = geometry_safeMP.get_action_safeMP_pulled(qdot, qddot_safeMP, mode=mode, dt=dt)

            # --- action fabrics --- #
            arguments_dict = dict(
                q=ob_robot["joint_state"]["position"][0:dof],
                qdot=ob_robot["joint_state"]["velocity"][0:dof],
                x_goal_0=ob_robot['FullSensor']['goals'][2+nr_obst]['position'],
                weight_goal_0=ob_robot['FullSensor']['goals'][2+nr_obst]['weight'],
                x_goal_1=ob_robot['FullSensor']['goals'][3+nr_obst]['position'],
                weight_goal_1=0., #ob_robot['FullSensor']['goals'][3+nr_obst]['weight'],
                x_goal_2=ob_robot['FullSensor']['goals'][4+nr_obst]['position'],
                weight_goal_2=0., #ob_robot['FullSensor']['goals'][3+nr_obst]['weight'],
                # x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
                # radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
                # x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
                # radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
                radius_body_links={3: 0.1, 4: 0.1, 5: 0.1, 6:0.1, 7: 0.1},
                constraint_0=np.array([0, 0, 1, 0.0]))
            action_fabrics = planner.compute_action(**arguments_dict)
            M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = planner_avoidance.compute_M_f_action_avoidance(
                **arguments_dict)
            M_safeMP = np.identity(dof,)
            qddot_speed = np.zeros((dof,)) #todo: think about what to do with speed regulation term!!
            action_fabrics_safeMP = self.combine_action(M_avoidance, M_safeMP, f_avoidance, -qddot_safeMP[0:dof], qddot_speed, planner,
                                                   qdot=ob_robot["joint_state"]["velocity"][0:dof])


            ob, *_ = env.step(action_safeMP_pulled)

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
        plt.savefig(params.results_path+"images/kuka_robot_plot_2D")
        make_plots = plotting_functions(results_path=params.results_path)
        make_plots.plotting_x_values(x_list, dt=dt, x_start=x_list[:, 0], x_goal=np.array(goal_pos), scaling_room=scaling_room)
        env.close()
        return {}


if __name__ == "__main__":
    render = True
    dof = 7
    mode = "acc"
    dt = 0.01
    nr_obst = 0
    init_pos = np.zeros((dof,))
    goal_pos = [-0.24355761, -0.75252747, 0.5] # [0.1, -0.6, 0.4]

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initialize_environment_kuka(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos)
    example_class = example_kuka_stableMP_fabrics_2D_2nd()
    res = example_class.run_kuka_example(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode)
