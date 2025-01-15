import numpy as np

from pumafabrics.puma_adapted.tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from pumafabrics.puma_adapted.initializer import initialize_framework
from pumafabrics.tamed_puma.utils.normalizations_2 import normalization_functions
from pumafabrics.tamed_puma.utils.plot_point_robot import plotting_functions
from pumafabrics.tamed_puma.create_environment.environments import trial_environments

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.

class example_point_robot_PUMA():
    def __init__(self, v_min=0, v_max=0, acc_min=0, acc_max=0):
        dt = 0.01
        self.v_min = v_min
        self.v_max = v_max
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.INT_COLLISION_CHECK = 0
        self.BOOL_COLLISION_CHECK = 0
        self.params = {}
        self.params["x_min"] = np.array([-10, -10])
        self.params["x_max"] = np.array([10, 10])

    def get_action_in_limits(self, action_old, mode="acc"):
        if mode == "vel":
            action = np.clip(action_old, self.v_min, self.v_max)
        else:
            action = np.clip(action_old, self.acc_min, self.acc_max)
        return action

    def stop_when_collided(self, q, obst_struct, w):
        for i in range(len(obst_struct)):
            pos_obst = obst_struct[i+3]["position"]
            radius_obst = obst_struct[i+3]["size"]

            distance = np.linalg.norm(q[0:2] - pos_obst[0:2]) - radius_obst
            print("distance:", distance)
            if distance < 0.2 and self.BOOL_COLLISION_CHECK == 0:
                self.INT_COLLISION_CHECK = w
                self.BOOL_COLLISION_CHECK = 1
            elif distance < 0.2 and self.BOOL_COLLISION_CHECK == 1:
                self.BOOL_COLLISION_CHECK = 1
        return {}

    def run_point_robot_urdf(self, n_steps=2000, env=None, goal=None, init_pos=np.array([-5, 5]), goal_pos=[-2.4355761, -7.5252747], mode="acc", mode_NN="2nd", dt=0.01):
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
        # --- parameters --- #

        if mode == "vel":
            str_mode = "velocity"
        elif mode == "acc":
            str_mode = "acceleration"
        else:
            print("this control mode is not defined")

        action_safeMP = np.array([0.0, 0.0, 0.0])
        action_fabrics = np.array([0.0, 0.0, 0.0])
        ob, *_ = env.step(action_safeMP)
        q_list = np.zeros((2, n_steps))

        # Parameters
        params_name = '2nd_order_2D'
        x_t_init = np.array([np.append(ob['robot_0']["joint_state"]["position"][0:2], ob['robot_0']["joint_state"]["velocity"][0:2])]) # initial states
        # simulation_length = 2000
        results_base_directory = '../pumafabrics/puma_adapted/'

        # Load parameters
        Params = getattr(importlib.import_module('pumafabrics.puma_adapted.params.' + params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Translation of goal:
        normalizations = normalization_functions(x_min=self.params["x_min"], x_max=self.params["x_max"], dt=dt, mode_NN=mode_NN, learner=learner)
        state_goal = np.array((goal._sub_goals[0]._config["desired_position"]))
        goal_normalized = normalizations.call_normalize_state(state=state_goal)
        translation = normalizations.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN)
        translation_gpu = torch.FloatTensor(translation).cuda()

        # Initialize dynamical system
        min_vel = learner.min_vel
        max_vel = learner.max_vel
        dof = len(min_vel[0])
        x_init_cpu, x_init_gpu = normalizations.normalize_state_position_to_NN(x_t=x_t_init, translation_cpu=translation)
        dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

        # Initialize trajectory plotter
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        trajectory_plotter = TrajectoryPlotter(fig, x0=x_init_cpu.T, pause_time=1e-5, goal=data['goals training'][0])
        # x_t_NN = torch.FloatTensor(x_t_init_scaled).cuda()

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:2]
            qdot = ob_robot["joint_state"]["velocity"][0:2]
            if self.BOOL_COLLISION_CHECK == 1:
                q_list[:, w] = q_list[:, self.INT_COLLISION_CHECK-1]
            else:
               q_list[:, w] = q
            x_t = np.array([np.append(q, qdot)])

            # --- translate to axis system of NN ---#
            x_t_cpu, x_t_gpu = normalizations.normalize_state_position_to_NN(x_t=x_t, translation_cpu=translation)

            # --- get action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]
            if mode == "acc":
                action_t_gpu = transition_info["desired acceleration"]
            else:
                action_t_gpu = transition_info["desired "+str_mode]

            action_safeMP[0:dof] = normalizations.reverse_transformation(action_gpu=action_t_gpu, mode_NN=mode_NN)
            action = self.get_action_in_limits(action_safeMP, mode=mode)
            self.stop_when_collided(q=q, obst_struct=ob_robot["FullSensor"]["obstacles"], w=w)
            ob, *_, = env.step(action)

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
        plt.savefig("../examples/images/point_robot_PUMA")
        env.close()
        make_plots = plotting_functions(results_path="../examples/images/")
        make_plots.plotting_q_values(q_list, dt=dt, q_start=q_list[:, 0], q_goal=np.array(goal_pos), file_name="point_robot_q_PUMA")
        return q_list

def main(render=True):
    # --- Initial parameters --- #
    mode = "acc"
    mode_NN = "2nd"
    dt = 0.01
    init_pos = np.array([0.0, 0.0])
    goal_pos = [-2.4355761, -7.5252747]

    dof = 2
    v_min = -50 * np.ones((dof+1,))
    v_max = 50 * np.ones((dof+1,))
    acc_min = -50 * np.ones((dof+1,))
    acc_max = 50 * np.ones((dof+1,))

    # --- generate environment
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos)

    example_class = example_point_robot_PUMA(v_min=v_min, v_max=v_max, acc_min=acc_min, acc_max=acc_max)
    res = example_class.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos, dt=dt, mode=mode, mode_NN=mode_NN)
    return {}

if __name__ == "__main__":
    main()