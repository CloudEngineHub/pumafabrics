import numpy as np

from functions_safeMP_fabrics.filters import PDController
from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.utils.plotting_functions2 import plotting_functions2
from agent.utils.normalizations import normalizaton_sim_NN
from functions_safeMP_fabrics.environments import trial_environments
from functions_safeMP_fabrics.kinematics_kuka import KinematicsKuka
from tools.animation import TrajectoryPlotter
import pybullet as p
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
import copy

class example_kuka_stableMP_R3S3():
    def __init__(self):
        dt = 0.01
    def define_goal_struct(self, goal_pos):
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

    def run_kuka_example(self, n_steps=2000, env=None, goal=None, init_pos=np.array([0.0, 0.0, 0.]), goal_pos=[-0.24355761, -0.35252747, 0.5], mode="acc", mode_NN = "1st", dt=0.01):
        # --- parameters --- #
        dof = 6
        dim_pos = 3
        dim_task = 7
        end_link_name = "iiwa_link_6"
        vel_limits = np.array([86, 85, 100, 75, 130, 135])*np.pi/180
        scaling_room = {"x": [-1, 1], "y":[-1, 1], "z": [0, 2]}
        if mode == "vel":
            str_mode = "velocity"
        elif mode == "acc":
            str_mode = "acceleration"
        else:
            print("this control mode is not defined")

        env.add_collision_link(0, 3, shape_type='sphere', size=[0.10])
        env.add_collision_link(0, 4, shape_type='sphere', size=[0.10])
        # env.add_collision_link(0, 7, shape_type='sphere', size=[0.10])

        goal = self.define_goal_struct(goal_pos=goal_pos)
        action = np.zeros(dof)
        ob, *_ = env.step(np.append(action, np.array([0.])))

        # construct symbolic pull-back of Imitation learning geometry:
        kuka_kinematics = KinematicsKuka(end_link_name=end_link_name)

        # Parameters
        if mode_NN == "1st":
            params_name = '1st_order_R3S3_converge'
        else:
            print("not implemented!!")
            params_name = '2nd_order_R3S3'
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        qdot_init = ob['robot_0']["joint_state"]["velocity"][0:dof]

        if params_name[0] == '1':
            x_t_init = kuka_kinematics.forward_kinematics(q_init, end_link_name=end_link_name)
        else:
            print("not implemented")

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
        translation = normalizations.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN)
        offset_orientation = normalizations.get_offset_quaternion(goal_quat_desired=orientation_goal, goal_NN_quat=goal_NN[3:])#todo: now both goals are the same!!
        x_t_init[3:] = kuka_kinematics.check_quaternion_initial(x_orientation=x_t_init[3:], translation_quat=offset_orientation) #self, x_orientation, translation_quat
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
        filter_nr = 60
        q_list = np.zeros((dof, n_steps))
        qdot_list = np.zeros((dof, n_steps))
        action_list = np.zeros((dof, n_steps))
        quat_vel_list = np.zeros((4, n_steps))
        angular_vel_list = np.zeros((3, n_steps))
        x_list = np.zeros((dim_pos, n_steps))
        quat_list = np.zeros((4, n_steps))
        quat_prev = copy.deepcopy(x_t_init[3:])

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            q_list[:, w] = q
            qdot_list[:, w] = qdot

            # --- End-effector state ---#
            pose_t_ee =  kuka_kinematics.forward_kinematics(q, end_link_name=end_link_name)
            pose_t_ee[3:] = kuka_kinematics.check_quaternion_flipped(quat = pose_t_ee[3:], quat_prev = quat_prev)
            x_ee = pose_t_ee[0:3]
            xee_orientation = pose_t_ee[3:]
            quat_prev = copy.deepcopy(xee_orientation)
            if params_name[0] == "1":
                x_t = np.array([np.append(x_ee, xee_orientation)])
            else:
                print("not implemented!!")
            x_list[:, w] = x_t[0][0:dim_pos]

            # ADAPTTTTTT:
            #x_t[0][3:3+4] = np.array([1., 0., 0., 0.]) #todo: replace!!!!

            # --- translate to axis system of NN ---#
            x_t_gpu, _ = normalizations.transformation_to_NN(x_t=x_t, translation_gpu=translation_gpu,
                                           dt=dt, min_vel=dynamical_system.min_vel, max_vel=dynamical_system.max_vel)
            x_t_gpu[0][3:7] = normalizations.system_quat_to_NN(quat=x_t_gpu[0][3:7], offset=offset_orientation)
            if mode_NN == "2nd":
                x_t_gpu[0][-7:] = torch.cuda.FloatTensor([0., 0., 0., 0., 0., 0., 0.])

            # --- get action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]
            if mode_NN == "2nd":
                print("not implemented!!")
                action_t_gpu = transition_info["phi"]
                xddot_t_NN  = transition_info["phi"]
            else:
                action_t_gpu = transition_info["desired velocity"]
                xddot_t_NN = transition_info["phi"]
            action_safeMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, dt=dt, mode_NN=mode_NN)
            xddot_safeMP = normalizations.reverse_transformation(action_gpu=xddot_t_NN, dt=dt, mode_NN=mode_NN)

            # # -- transform to configuration space --#
            action_norm = action_t_gpu.T.cpu().detach().numpy()

            if mode_NN == "1st":
                action_quat_vel = action_norm[3:]
                action_safeMP_quat = np.append(action_safeMP[0:3], action_quat_vel)

                euler_vel = kuka_kinematics.quat_vel_to_angular_vel(angle_quaternion=xee_orientation,
                                                                    vel_quaternion=action_quat_vel)
                action_safeMP_pulled = kuka_kinematics.get_qdot_from_linear_angular_vel(q=q, xdot=action_safeMP_quat) #action_safeMP)
            else:
                print("not implemented!!")

            #check when action limit is exceeded, due to xdot or qdot?
            #action_lim = geometry_safeMP.get_action_in_limits(action_old=action_safeMP_pulled, mode=mode)
            if mode == "acc" and mode_NN == "1st":
                # ---- get a decent acceleration based on the velocity command ---#
                action = pdcontroller.control(desired_velocity=action_safeMP_pulled, current_velocity=qdot)
            else:
                action = np.clip(action_safeMP_pulled, -vel_limits, vel_limits)

            # test:
            #action = kuka_kinematics.inverse_diff_kinematics(xdot=np.array([0., 0., 0., 0., 0., 1.])).T.cpu().detach().numpy().transpose()[0]

            ob, *_ = env.step(np.append(action, np.array([0.])))

            #pause when high action:
            norm_action = np.linalg.norm(action)
            if norm_action > 20 or w == 50:
                print(" xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ")
                print("time: ", w*dt)
                print("action quat vel: ", action_quat_vel.transpose())
                print("euler vel: ", euler_vel.transpose())
                print("action safeMP: ", action_safeMP_pulled)
                print(" --------------------- ")
                print("previous action quat vel:", quat_vel_list[:, w-1])
                print("previous euler vel:", angular_vel_list[:, w-1])
                print("previous action: ", action_list[:, w-1])
                print(" ---------------------")
                print("angles:", q)
                kkk=1

            # get pybullet angular velocities, and quaternion velocities:
            body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
            num_joints = p.getNumJoints(0)
            _, _, _, _, _, _, linear_velocity, angular_velocity = p.getLinkState(bodyUniqueId=1, linkIndex=7,
                                                                                 computeLinkVelocity=1)
            # print("angular_vel:", angular_velocity)

            # get quaternion velocity to angular velocity:
            # print("quat_vel:", action_quat_vel)

            # save for plotting:
            quat_vel_list[:, w] = action_quat_vel.transpose()
            angular_vel_list[:, w] = euler_vel.transpose()
            action_list[:, w] = np.clip(action_safeMP_pulled, -vel_limits, vel_limits)
            quat_list[:, w] = xee_orientation


            # #check for sign flips!
            # if w>1:
            #     kuka_kinematics.check_sign_flip(v=quat_vel_list[:, w], v_prev=quat_vel_list[:, w-1])

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
        plt.savefig(params.results_path+"images/kuka_robot_plot_R3S3")
        # make_plots = plotting_functions(results_path=params.results_path)
        # make_plots.plotting_x_values(x_list, dt=dt, x_start=x_list[:, 0], x_goal=np.array(goal_pos), scaling_room=scaling_room)
        plotting_class = plotting_functions2(results_path=params.results_path)
        plotting_class.velocities_over_time(quat_vel_list=quat_vel_list,
                                            ang_vel_list=angular_vel_list,
                                            joint_vel_list=qdot_list,
                                            action_list=action_list,
                                            dt=dt)
        plotting_class.pose_over_time(quat_list=quat_list)
        env.close()
        return {}


if __name__ == "__main__":
    render = True
    dof = 6
    mode = "vel"
    mode_NN = "1st"
    dt = 0.01
    nr_obst = 0
    init_pos = np.zeros((dof,))
    goal_pos = [-0.24355761, -0.3252747, 0.5] #[0.09206515, -0.55024947,  0.36955635] # [-0.24355761, -0.75252747, 0.5]

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initialize_environment_kuka(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos, nr_obst=nr_obst, end_effector_link="iiwa_link_6")
    example_class = example_kuka_stableMP_R3S3()
    res = example_class.run_kuka_example(n_steps=400, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN)
