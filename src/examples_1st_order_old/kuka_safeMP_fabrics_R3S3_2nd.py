import os
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from functions_safeMP_fabrics.filters import PDController
from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.kinematics.geometry_IL import construct_IL_geometry
from pumafabrics.tamed_puma.utils.plot_point_robot import plotting_functions
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from agent.utils.normalizations import normalizaton_sim_NN
from functions_safeMP_fabrics.environments import trial_environments
from functions_safeMP_fabrics.kinematics_kuka import KinematicsKuka
from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework

class example_kuka_stableMP_R3S3():
    def __init__(self):
        dt = 0.01
    def define_goal_struct(self, goal_pos):
        goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "iiwa_link_0",
                "child_link": "iiwa_link_ee",
                "desired_position": goal_pos,
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 0.0,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "iiwa_link_6",
                "child_link": "iiwa_link_ee",
                "desired_position": [0.1, 0.0, 0.0],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        goal = GoalComposition(name="goal", content_dict=goal_dict)
        return goal
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
        planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            forward_kinematics,
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
        planner.concretize(mode=mode, time_step=dt, extensive_concretize=True, bool_speed_control=bool_speed_control)
        return planner


    def combine_action(self, M_avoidance, M_attractor, f_avoidance, f_attractor, xddot_speed, planner, qdot = []):
        xddot_combined = -np.dot(planner.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + xddot_speed
        if planner._mode == "vel":
            action_combined = qdot + planner._time_step * xddot_combined
        else:
            action_combined = xddot_combined
        return action_combined

    def run_kuka_example(self, n_steps=2000, env=None, goal=None, init_pos=np.array([0.0, 0.0, 0.]), goal_pos=[-0.24355761, -0.75252747, 0.5], mode="acc", mode_NN = "1st", dt=0.01):
        # --- parameters --- #
        dof = 7
        dim_pos = 3
        dim_task = 7

        vel_limits = np.array([86, 85, 100, 75, 130, 135, 135])*np.pi/180
        collision_radii = {3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}

        # mode = "vel"
        # dt = 0.01
        # nr_obst = 0
        # init_pos = np.zeros((dof,))
        # goal_pos = [-0.24355761, -0.75252747, 0.5]
        # scaling_factor = 1
        scaling_room = {"x": [-1, 1], "y":[-1, 1], "z": [0, 2]}
        if mode == "vel":
            str_mode = "velocity"
        elif mode == "acc":
            str_mode = "acceleration"
        else:
            print("this control mode is not defined")

        # (env, goal) = initalize_environment(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
        goal = self.define_goal_struct(goal_pos=goal_pos)
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
        # fk = planner.get_forward_kinematics("iiwa_link_7")
        fk_full = planner._forward_kinematics.fk(q=planner._variables.position_variable(),
                                                 parent_link="iiwa_link_0",
                                                 child_link="iiwa_link_7",
                                                 positionOnly=False)
        geometry_safeMP = construct_IL_geometry(planner=planner, dof=dof, dimension_task=dim_task, forwardkinematics=fk_full,
                                                variables=planner.variables, first_dim=0, BOOL_pos_orient = True)
        kuka_kinematics = KinematicsKuka()
        # kinematics and IK functions
        x_function, xdot_function = geometry_safeMP.fk_functions()
        rotation_matrix_function = geometry_safeMP.rotation_matrix_function()

        # create pulled function in configuration space
        h_function = geometry_safeMP.create_function_h_pulled()
        vel_pulled_function = geometry_safeMP.create_function_vel_pulled()
        geometry_safeMP.set_limits(v_min=-vel_limits, v_max=vel_limits, acc_min=-50*np.ones((dof,)), acc_max=50*np.ones((dof,)))

        # Parameters
        if mode_NN == "1st":
            params_name = '1st_order_R3S3'
        else:
            params_name = '2nd_order_R3S3'
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        qdot_init = ob['robot_0']["joint_state"]["velocity"][0:dof]

        if params_name[0] == '1':
            x_t_init = kuka_kinematics.forward_kinematics(q_init)
        else:
            xdot_function_analytical = geometry_safeMP.fk_functions_with_orientation()
            xdot_init = xdot_function_analytical(q_init, qdot_init)
            x_orientation = np.array([1., 0., 0., 0])
            x_t_init = np.array([np.append(np.append(x_function(q_init).full()[0:dim_pos], x_orientation), xdot_init)])
            print("not implemented")
        # x_orientation = geometry_safeMP.get_orientation_quaternion(q_init)
        # if params_name[0] == '2':
        #     xdot_function_analytical = geometry_safeMP.fk_functions_with_orientation()
        #     xdot_init = xdot_function_analytical(q_init, qdot_init)
        #     x_orientation = np.array([1., 0., 0., 0])
        #     x_t_init = np.array([np.append(np.append(x_function(q_init).full()[0:dim_pos], x_orientation), xdot_init)])
        # else:
        #     x_t_init = np.array([np.append(x_function(q_init)[0:dim_pos], x_orientation)])

        results_base_directory = './'
        pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=dt)

        # Load parameters
        Params = getattr(importlib.import_module('params.' + params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Translation of goal:
        normalizations = normalizaton_sim_NN(scaling_room=scaling_room, dof_task=dim_task)
        state_goal = np.array((goal._sub_goals[0]._config["desired_position"]))
        orientation_goal = np.array([1., 0., 0., 0.]) #np.array([0.0726, -0.6326, 0.3511, -0.6865]) #np.array([1., 0., 0., 0.])
        goal_normalized = np.append(normalizations.call_normalize_state(state=state_goal[0:dim_pos]), orientation_goal)
        translation = normalizations.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN) #todo: now both goals are the same!!
        translation_gpu = torch.FloatTensor(translation).cuda()

        # Initialize dynamical system
        min_vel = learner.min_vel
        max_vel = learner.max_vel
        x_init_gpu, x_init_cpu = normalizations.transformation_to_NN(x_t=x_t_init, translation_gpu=translation_gpu,
                                                      dt=dt, min_vel=min_vel, max_vel=max_vel)
        if mode_NN == "2nd":
            x_init_gpu[0][-7:] = torch.cuda.FloatTensor([0., 0., 0., 0., 0., 0., 0.]) # todo, replace!!!
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
        filter_nr = 10
        q_list = np.zeros((dof, n_steps))
        qdot_list = np.zeros((dof, n_steps))
        x_list = np.zeros((dim_pos, n_steps))

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            q_list[:, w] = q
            qdot_list[:, w] = qdot

            # --- End-effector state ---#
            pose_t_ee =  kuka_kinematics.forward_kinematics(q)
            x_ee = pose_t_ee[0:3]
            xee_orientation = pose_t_ee[3:]
            # x_ee = x_function(q).full().transpose()[0][0:dim_task]
            # # x_t = geometry_safeMP.joint_state_to_action_state(q=q, qdot=qdot)
            # # xdot_ee = xdot_function(q, qdot).full().transpose()[0]
            # xee_orientation = geometry_safeMP.get_orientation_quaternion(q)
            if params_name[0] == "1":
                x_t = np.array([np.append(x_ee, xee_orientation)])
            else:
                xdot_t = xdot_function_analytical(q, qdot).full().transpose()[0][0:dim_task]
                x_t = np.array([np.append(np.append(x_ee, xee_orientation), xdot_t)])
            x_list[:, w] = x_t[0][0:dim_pos]

            # ADAPTTTTTT:
            #x_t[0][3:3+4] = np.array([1., 0., 0., 0.]) #todo: replace!!!!

            # --- translate to axis system of NN ---#
            x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                           dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)
            if mode_NN == "2nd":
                x_t_gpu[0][-7:] = torch.cuda.FloatTensor([0., 0., 0., 0., 0., 0., 0.])
            # --- get action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]
            if mode_NN == "2nd":
                action_t_gpu = transition_info["phi"]
                xddot_t_NN  = transition_info["phi"]
            else: #mode_NN == 1st
                action_t_gpu = transition_info["desired velocity"]
                xddot_t_NN = transition_info["phi"]
            action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, dt=dt, mode_NN=mode_NN) #todo: remove hardcoded "vel"
            xddot_safeMP = normalizations.reverse_transformation(action_gpu=xddot_t_NN, dt=dt, mode_NN=mode_NN) #todo: remove hardcoded "vel

            # # -- transform to configuration space --#
            action_norm = action_t_gpu.T.cpu().detach().numpy()

            if mode_NN == "1st":
                action_quat_vel = action_safeMP[3:]
                action_safeMP_quat = np.append(action_safeMP[0:3], action_quat_vel)

                euler_vel = geometry_safeMP.quat_vel_to_angular_vel(angle_quaternion=xee_orientation,
                                                                    vel_quaternion=action_quat_vel)
                # action_safeMP_euler = np.append(action_safeMP[0:3], euler_vel / 0.01)
                # action_safeMP_pulled = geometry_safeMP.get_numerical_vel_pulled(q_num=q, h_NN=action_safeMP_euler)

                # action_safeMP_pulled = kuka_kinematics.get_qdot_from_pos_orient(q=q, xdot=action_safeMP_quat)
                # action_safeMP_pulled = kuka_kinematics.get_qdot_from_position(q=q, xdot=action_safeMP[:3])
                action_safeMP_pulled = kuka_kinematics.get_qdot_from_linear_angular_vel(q=q, xdot=action_safeMP_quat)
            else:
                xddot_safeMP_euler = np.append(xddot_safeMP[0:dim_pos], np.array([0., 0., 0.])) #todo, replace with something better!!
                qddot_safeMP = geometry_safeMP.get_numerical_h_pulled(q_num=q, qdot_num=qdot, h_NN=xddot_safeMP_euler[0:dim_task])
                action_safeMP_pulled = geometry_safeMP.get_action_safeMP_pulled(qdot, qddot_safeMP, mode=mode, dt=dt)

            #check when action limit is exceeded, due to xdot or qdot?
            #action_lim = geometry_safeMP.get_action_in_limits(action_old=action_safeMP_pulled, mode=mode)
            if mode == "acc" and mode_NN == "1st":
                # ---- get a decent acceleration based on the velocity command ---#
                action = pdcontroller.control(desired_velocity=action_safeMP_pulled, current_velocity=qdot)
                qddot_safeMP = action
            else:
                action = action_safeMP_pulled #np.clip(action_safeMP_pulled, -vel_limits, vel_limits)

            # ----- Fabrics action ----#
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                x_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
                radius_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['size'],
                x_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position'],
                radius_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size'],
                radius_body_links=collision_radii,
                constraint_0=np.array([0, 0, 1, 0.0]))

            M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = planner_avoidance.compute_M_f_action_avoidance(
                **arguments_dict)
            M_safeMP = np.identity(dof, )
            qddot_speed = np.zeros((dof,))  # todo: think about what to do with speed regulation term!!
            weight_safeMP = 0.2
            action_fabrics_safeMP = self.combine_action(M_avoidance, M_safeMP, f_avoidance, -weight_safeMP*qddot_safeMP[0:dof],
                                                   qddot_speed, planner,
                                                   qdot=ob_robot["joint_state"]["velocity"][0:dof])
            ob, *_ = env.step(action_fabrics_safeMP)

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
        plt.savefig(params.results_path+"images/kuka_robot_plot_R3S3")
        make_plots = plotting_functions(results_path=params.results_path)
        make_plots.plotting_x_values(x_list, dt=dt, x_start=x_list[:, 0], x_goal=np.array(goal_pos), scaling_room=scaling_room)
        env.close()
        return {}


if __name__ == "__main__":
    render = True
    dof = 7
    mode = "acc"
    mode_NN = "1st"
    dt = 0.01
    nr_obst = 2
    init_pos = np.zeros((dof,))
    # goal_pos = [-0.24355761, -0.75252747, 0.5]
    # goal_pos = [0.07311596, -0.5, 0.6]
    # goal_pos = [0., 0., 0.]
    goal_pos = [-0.24355761, -0.75252747, 0.5]

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initialize_environment_kuka(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos, nr_obst=nr_obst)
    example_class = example_kuka_stableMP_R3S3()
    res = example_class.run_kuka_example(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN)
