import numpy as np

from functions_safeMP_fabrics.filters import PDController
from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.utils.plotting_functions2 import plotting_functions2
from functions_safeMP_fabrics.environments import trial_environments
from functions_safeMP_fabrics.kinematics_kuka import KinematicsKuka
from functions_safeMP_fabrics.cartesian_impedance_control import CartesianImpedanceController
import matplotlib.pyplot as plt
import importlib
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

    def run_kuka_example(self, n_steps=2000, env=None, goal=None, init_pos=np.array([0.0, 0.0, 0.]), goal_pos=[-0.24355761, -0.75252747, 0.5], mode="acc", mode_NN = "1st", dt=0.01):
        # --- parameters --- #
        dof = 7
        dim_pos = 3
        dim_task = 7
        vel_limits = np.array([86, 85, 100, 75, 130, 135, 135])*np.pi/180
        scaling_room = {"x": [-1, 1], "y":[-1, 1], "z": [0, 2]}
        if mode == "vel":
            str_mode = "velocity"
        elif mode == "acc":
            str_mode = "acceleration"
        else:
            print("this control mode is not defined")

        env.add_collision_link(0, 3, shape_type='sphere', size=[0.10])
        env.add_collision_link(0, 4, shape_type='sphere', size=[0.10])
        env.add_collision_link(0, 7, shape_type='sphere', size=[0.10])

        goal = self.define_goal_struct(goal_pos=goal_pos)
        action = np.zeros(dof)
        ob, *_ = env.step(action)

        # construct symbolic pull-back of Imitation learning geometry:
        kuka_kinematics = KinematicsKuka()

        # Parameters
        if mode_NN == "1st":
            params_name = '1st_order_R3S3'
        else:
            print("not implemented!!")
            params_name = '2nd_order_R3S3'
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]
        qdot_init = ob['robot_0']["joint_state"]["velocity"][0:dof]

        if params_name[0] == '1':
            x_t_init = kuka_kinematics.forward_kinematics(q_init)
        else:
            print("not implemented")

        results_base_directory = './'
        pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=dt)
        cartesian_controller = CartesianImpedanceController(robot=[])

        # Load parameters
        Params = getattr(importlib.import_module('params.' + params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # goal:
        state_goal = np.array((goal._sub_goals[0]._config["desired_position"]))
        orientation_goal = np.array([1., 0., 0., 0.])

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
            pose_t_ee =  kuka_kinematics.forward_kinematics(q)
            pose_t_ee[3:] = kuka_kinematics.check_quaternion_flipped(quat = pose_t_ee[3:], quat_prev = quat_prev)
            x_ee = pose_t_ee[0:3]
            xee_orientation = pose_t_ee[3:]
            quat_prev = copy.deepcopy(xee_orientation)
            if params_name[0] == "1":
                x_t = np.array([np.append(x_ee, xee_orientation)])
            else:
                print("not implemented!!")
            x_list[:, w] = x_t[0][0:dim_pos]

            action_quat_vel = np.zeros(4)
            euler_vel = np.zeros(3)
            action_safeMP_pulled = np.zeros(7)

            # ----- cartesian impedance controller ----#
            Jac_current = kuka_kinematics.call_jacobian(q)
            ee_velocity = np.matmul(Jac_current, qdot)[0].cpu().detach().numpy()
            action, action_quat_vel, euler_vel = cartesian_controller.control_law(position_d=state_goal,
                                                      orientation_d=orientation_goal,
                                                      ee_pose=x_t[0],
                                                      ee_velocity=ee_velocity,
                                                      J=Jac_current,
                                                      qdot=qdot)

            if w<1000:
                action = np.array([1., 1., 1., 1., 1., 1., 0.5])
            else:
                print("action:", action)
            ob, *_ = env.step(action)

            #pause when high action:
            norm_action = np.linalg.norm(action)
            # if norm_action > 10 or w == 50:
            #     print(" xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ")
            #     print("time: ", w*dt)
            #     print("action quat vel: ", action_quat_vel.transpose())
            #     print("euler vel: ", euler_vel.transpose())
            #     print("action safeMP: ", action_safeMP_pulled)
            #     print(" --------------------- ")
            #     print("previous action quat vel:", quat_vel_list[:, w-1])
            #     print("previous euler vel:", angular_vel_list[:, w-1])
            #     print("previous action: ", action_list[:, w-1])
            #     print(" ---------------------")
            #     print("angles:", q)
            #     kkk=1

            # # get pybullet angular velocities, and quaternion velocities:
            # body_ids = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
            # num_joints = p.getNumJoints(0)
            # _, _, _, _, _, _, linear_velocity, angular_velocity = p.getLinkState(bodyUniqueId=1, linkIndex=7,
            #                                                                      computeLinkVelocity=1)

            # save for plotting:
            quat_vel_list[:, w] = action_quat_vel.transpose()
            angular_vel_list[:, w] = euler_vel.transpose()
            action_list[:, w] = np.clip(action, -vel_limits, vel_limits)
            quat_list[:, w] = xee_orientation

        plt.savefig(params.results_path+"images/kuka_robot_plot_R3S3")
        plotting_class = plotting_functions2(results_path=params.results_path)
        plotting_class.velocities_over_time(quat_vel_list=quat_vel_list,
                                            ang_vel_list=angular_vel_list,
                                            joint_vel_list=qdot_list,
                                            action_list=action_list,
                                            dt=dt)
        plotting_class.pose_over_time(quat_list=quat_list)
        plotting_class.position_over_time(q_list=q_list, qdot_list=qdot_list)
        env.close()
        return {}


if __name__ == "__main__":
    render = True
    dof = 7
    mode = "vel"
    mode_NN = "1st"
    dt = 0.001
    nr_obst = 0
    init_pos = np.zeros((dof,))
    goal_pos = [-0.24355761, -0.75252747, 0.5] #[0.09206515, -0.55024947,  0.36955635] # [-0.24355761, -0.75252747, 0.5]
    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initialize_environment_kuka(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos, nr_obst=nr_obst)
    example_class = example_kuka_stableMP_R3S3()
    res = example_class.run_kuka_example(n_steps=2000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN)
