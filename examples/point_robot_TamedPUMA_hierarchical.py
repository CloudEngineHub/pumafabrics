import os
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from mpscenes.goals.goal_composition import GoalComposition

from pumafabrics.tamed_puma.tamedpuma.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from pumafabrics.puma_extension.tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from pumafabrics.puma_extension.initializer import initialize_framework
from pumafabrics.tamed_puma.utils.normalizations_2 import normalization_functions
from pumafabrics.tamed_puma.utils.plot_point_robot import plotting_functions
from pumafabrics.tamed_puma.create_environment.environments import trial_environments

"""
TamedPUMA - hierarchical example for a 3D point mass robot. The planner uses a 2D point
mass to compute actions for a simulated 3D point mass.
Although not included in the paper, this example lets fabrics track a end-effector positional reference provided by PUMA.
"""

class example_point_robot_hierarchical():
    def __init__(self):
        self.params = {}
        self.params["x_min"] = np.array([-10, -10])
        self.params["x_max"] = np.array([10, 10])

    def set_planner(self, goal: GoalComposition, ONLY_GOAL=False, bool_speed_control=True, mode="acc", dt=0.01):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        Params
        ----------
        goal: StaticSubGoal
            The goal to the motion planning problem.
        """
        degrees_of_freedom = 2
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/../pumafabrics/tamed_puma/config/urdfs/point_robot.urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            root_link="world",
            end_links=["base_link_y"],
        )
        collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
        collision_finsler = "2.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
        planner = ParameterizedFabricPlannerExtended(
            degrees_of_freedom,
            forward_kinematics,
            time_step=dt,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler,
            damper_beta =  "0.9 * (ca.tanh(-0.5 * (ca.norm_2(x) - 0.02)) + 1) * 6.5 + 0.01 + ca.fmax(0, sym('a_ex') - sym('a_le'))"
        )
        collision_links = ["base_link_y"]
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
        dof = 2
        planner = self.set_planner(goal, bool_speed_control=True, mode=mode, dt=dt)

        # action_safeMP = np.array([0.0, 0.0, 0.0])
        action_fabrics = np.array([0.0, 0.0, 0.0])
        ob, *_ = env.step(action_fabrics)
        q_list = np.zeros((2, n_steps))

        # Parameters
        params_name = '2nd_order_2D'
        x_t_init = np.array([np.append(ob['robot_0']["joint_state"]["position"][0:2], ob['robot_0']["joint_state"]["velocity"][0:2])]) # initial states
        # simulation_length = 2000
        results_base_directory = '../pumafabrics/puma_extension/'

        # Load parameters
        Params = getattr(importlib.import_module('pumafabrics.puma_extension.params.' + params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Translation of goal:
        normalizations = normalization_functions(x_min=self.params["x_min"], x_max=self.params["x_max"], dt=dt,
                                                 mode_NN=mode_NN, learner=learner)
        state_goal = np.array((goal._sub_goals[0]._config["desired_position"]))
        goal_normalized = normalizations.call_normalize_state(state=state_goal)
        translation = normalizations.get_translation(goal_pos=goal_normalized, goal_pos_NN=goal_NN)
        translation_gpu = torch.FloatTensor(translation).cuda()

        # Initialize dynamical system
        min_vel = learner.min_vel
        max_vel = learner.max_vel
        x_init_cpu, x_init_gpu = normalizations.normalize_state_position_to_NN(x_t=x_t_init, translation_cpu=translation)
        dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)
        # dynamical_system.saturate

        # Initialize trajectory plotter
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        trajectory_plotter = TrajectoryPlotter(fig, x0=x_init_cpu.T, pause_time=1e-5, goal=data['goals training'][0])
        # x_t_NN = torch.FloatTensor(x_t_init_scaled).cuda()

        # Initialize lists

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:2]
            qdot = ob_robot["joint_state"]["velocity"][0:2]
            q_list[:, w] = q
            x_t = np.array([np.append(q, qdot)])

            # --- translate to axis system of NN ---#
            x_t_cpu, x_t_gpu = normalizations.normalize_state_position_to_NN(x_t=x_t, translation_cpu=translation)

            # --- get action by NN --- #
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            x_t_NN = transition_info["desired state"]

            # denormalized position of NN, used as goal for fabrics
            pos_safeMP = normalizations.reverse_transformation_position(x_t_NN[0][0:2])

            # --- get action by fabrics --- #
            arguments_dict = dict(
                q=ob_robot["joint_state"]["position"][0:dof],
                qdot=ob_robot["joint_state"]["velocity"][0:dof],
                x_goal_0=pos_safeMP,
                weight_goal_0=1000, #ob_robot['FullSensor']['goals'][2]['weight'],
                x_obst_0=ob_robot['FullSensor']['obstacles'][3]['position'],
                radius_obst_0=ob_robot['FullSensor']['obstacles'][3]['size'],
                x_obst_1=ob_robot['FullSensor']['obstacles'][4]['position'],
                radius_obst_1=ob_robot['FullSensor']['obstacles'][4]['size'],
                # x_obst_2=ob_robot['FullSensor']['obstacles'][5]['position'],
                # radius_obst_2=ob_robot['FullSensor']['obstacles'][5]['size'],
                radius_body_base_link_y=np.array([0.2])
            )
            action_fabrics[0:dof] = planner.compute_action(**arguments_dict)

            # --- update environment ---#
            ob, *_, = env.step(np.append(action_fabrics, 0))

            # --- Update plot ---#
            trajectory_plotter.update(x_t_gpu.T.cpu().detach().numpy())
        plt.savefig(params.results_path+"images/point_robot_hierarchical")
        env.close()
        make_plots = plotting_functions(results_path=params.results_path)
        make_plots.plotting_q_values(q_list, dt=dt, q_start=q_list[:, 0], q_goal=np.array(goal_pos))
        return q_list

def main(render=True, n_steps=5000):
    # --- Initial parameters --- #
    mode = "acc"
    mode_NN = "2nd"
    dt = 0.01
    init_pos = np.array([0., 0.])
    goal_pos = [-2.4355761, -7.5252747]

    # --- generate environment --- #
    envir_trial = trial_environments()
    (env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos,
                                                              goal_pos=goal_pos)

    # --- run example --- #
    example_class = example_point_robot_hierarchical()
    res = example_class.run_point_robot_urdf(n_steps=n_steps, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos,
                               dt=dt, mode=mode, mode_NN=mode_NN)
    return {}

if __name__ == "__main__":
    main()