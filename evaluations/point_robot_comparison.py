"""
This file is to generate a comparison plot between safeMP and safeMP+fabrics and theorem III.5.
"""

import numpy as np
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from examples.point_robot_PUMA import example_point_robot_PUMA
from examples.point_robot_TamedPUMA_FPM import example_point_robot_TamedPUMA_FPM
from examples.point_robot_TamedPUMA_CPM import example_point_robot_TamedPUMA_CPM
from examples.point_robot_Fabrics import example_point_robot_fabrics
from examples.point_robot_TamedPUMA_hierarchical import example_point_robot_hierarchical
from pumafabrics.tamed_puma.utils.plot_point_robot import plotting_functions

# --- Initial parameters --- #
mode = "acc"
mode_NN = "2nd"
dt = 0.01
init_pos = np.array([0.0, 0.0])
goal_pos = [-2.4355761, -7.5252747]
scaling_room = {"x": [-10, 10], "y":[-10, 10]}
render = False

dof = 2
v_min = -50 * np.ones((dof+1,))
v_max = 50 * np.ones((dof+1,))
acc_min = -50 * np.ones((dof+1,))
acc_max = 50 * np.ones((dof+1,))

envir_trial = trial_environments()

# --- run fabrics (only) example ---#
(env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
example_fabrics= example_point_robot_fabrics()
q_list_fabrics = example_fabrics.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos, dt=dt, mode=mode, mode_NN=mode_NN)

# --- run safe MP (only) example ---#
(env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
example_safeMP = example_point_robot_PUMA(v_min=v_min, v_max=v_max, acc_min=acc_min, acc_max=acc_max)
q_list_safeMP = example_safeMP.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos, dt=dt, mode=mode, mode_NN = mode_NN)

# # --- hierarchical method ---#
# (env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
# example_hierachical = example_point_robot_hierarchical()
# q_list_hierarchical = example_hierachical.run_point_robot_urdf(n_steps=3000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos, dt=dt, mode=mode, mode_NN=mode_NN)

# run safe MP + fabrics example ---#
(env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
example_safeMP_fabrics = example_point_robot_TamedPUMA_FPM()
q_list_safeMP_fabrics = example_safeMP_fabrics.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos, dt=dt, mode=mode, mode_NN=mode_NN)

# run theorem III.5 ---#
(env, goal) = envir_trial.initalize_environment_pointmass(render, mode=mode, dt=dt, init_pos=init_pos, goal_pos=goal_pos)
example_theoremIII_5 = example_point_robot_TamedPUMA_CPM(v_min=v_min, v_max=v_max, acc_min=acc_min, acc_max=acc_max)
q_list_theoremIII_5 = example_theoremIII_5.run_point_robot_urdf(n_steps=1000, env=env, goal=goal, init_pos=init_pos, goal_pos=goal_pos, dt=dt, mode=mode, mode_NN=mode_NN)


#--- plot results ---#
plots_class = plotting_functions()
plots_class.comparison_plot(q_list_fabrics, q_list_safeMP, q_list_safeMP_fabrics, q_list_theoremIII_5, q_goal=goal_pos, q_start=init_pos, scaling_room=scaling_room)