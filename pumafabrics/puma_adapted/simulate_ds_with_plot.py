from pumafabrics.puma_adapted.tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from pumafabrics.puma_adapted.initializer import initialize_framework
import numpy as np

# Parameters
params_name = "2nd_order_R3S3_kinova"
# params_name = "1st_order_2D"
if params_name == '1st_order_2D':
    x_t_init = np.array([[0.5, 0.6], [-0.75, 0.9], [0.9, -0.9], [-0.9, -0.9], [0.9, 0.9], [0.9, 0.3], [-0.9, -0.1],
                         [-0.9, 0.0], [0.4, 0.4], [0.9, -0.1], [-0.9, -0.5], [0.9, -0.5]])  # initial states
elif params_name == '2nd_order_2D':
    x_t_init = np.array([[0.5, 0.6, 0., 0.], [-0.75, 0.9, 0., 0.], [0.9, -0.9, 0., 0.], [-0.9, -0.9, 0., 0.], [0.9, 0.9, 0., 0.],
                         [0.9, 0.3, 0., 0.], [-0.9, -0.1, 0., 0.], [-0.9, 0.0, 0., 0.], [0.4, 0.4, 0., 0.], [0.9, -0.1, 0., 0.],
                         [-0.9, -0.5, 0., 0.], [0.9, -0.5, 0., 0.]])  # initial states
elif params_name[0:12] == '1st_order_3D':
    x_t_init = np.array([[0.5, 0.6, 0.], [-0.75, 0.9, 0.], [0.9, -0.9, 0.], [-0.9, -0.9, 0.], [0.9, 0.9, 0.], [0.9, 0.3, 0.],
                         [-0.9, -0.1, 0.], [-0.9, 0.0, 0.], [0.4, 0.4, 0.], [0.9, -0.1, 0.], [-0.9, -0.5, 0.], [0.9, -0.5, 0.]])
elif params_name[0:14] == "1st_order_R3S3":
    # x_t_init = np.array([[0.5, 0.6, 0.1, 0.0, 0.0, 0.0, 0.0]])
    x_t_init = np.array([[-0.1125, -0.4735, -0.4360,  0.3808, -0.8680,  0.0633, -0.3123], [-0.75, 0.9, 0., 1.0, 0.0, 0.0, 0.0], [0.9, -0.9, 0., 1.0, 0.0, 0.0, 0.0],
                         [-0.9, -0.9, 0., 1.0, 0.0, 0.0, 0.0], [0.9, 0.9, 0., 1.0, 0.0, 0.0, 0.0], [0.9, 0.3, 0., 1.0, 0.0, 0.0, 0.0],
                         [-0.9, -0.1, 0., 1.0, 0.0, 0.0, 0.0], [-0.9, 0.0, 0., 1.0, 0.0, 0.0, 0.0], [0.4, 0.4, 0., 1.0, 0.0, 0.0, 0.0],
                         [0.9, -0.1, 0., 1.0, 0.0, 0.0, 0.0], [-0.9, -0.5, 0., 1.0, 0.0, 0.0, 0.0], [0.9, -0.5, 0., 1.0, 0.0, 0.0, 0.0]])
elif params_name == '2nd_order_3D_euc_boundary':
    x_t_init = np.array([[0.5, 0.6, 0., 0., 0., 0.], [-0.75, 0.9, 0., 0., 0., 0.], [0.9, -0.9, 0., 0., 0., 0.], [-0.9, -0.9, 0., 0., 0., 0.],
                         [0.9, 0.9, 0., 0., 0., 0.], [0.9, 0.3, 0., 0., 0., 0.],
                         [-0.9, -0.1, 0., 0., 0., 0.], [-0.9, 0.0, 0., 0., 0., 0.], [0.4, 0.4, 0., 0., 0., 0.], [0.9, -0.1, 0., 0., 0., 0.],
                         [-0.9, -0.5, 0., 0., 0., 0.], [0.9, -0.5, 0., 0., 0., 0.]])
elif params_name[0:14] == '2nd_order_R3S3':
    x_t_init = np.array([[0.5, 0.6, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [-0.75, 0.9, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0.9, -0.9, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [-0.9, -0.9, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0.9, 0.9, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0.9, 0.3, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [-0.9, -0.1, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [-0.9, 0.0, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0.4, 0.4, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0.9, -0.1, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [-0.9, -0.5, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0.9, -0.5, 0., 1.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

obstacle_struct = {"centers": [[0.5, -0.5]],
                   "axes": [[0.2, 0.3]], "safety_margins": [[1.0]]}
simulation_length = 4000
results_base_directory = './'

# Load parameters
Params = getattr(importlib.import_module('params.' + params_name), 'Params')
params = Params(results_base_directory)
params.results_path += params.selected_primitives_ids + '/'
params.load_model = True

# Initialize framework
learner, _, data = initialize_framework(params, params_name, verbose=False)

# Initialize dynamical system

dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(x_t_init).cuda())

# Initialize trajectory plotter
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
fig.show()
trajectory_plotter = TrajectoryPlotter(fig, x0=x_t_init.T, pause_time=1e-5, goal=data['goals training'][0])

# Simulate dynamical system and plot
for i in range(simulation_length):
    # Do transition
    transition_info = dynamical_system.transition(space='task') #, obstacles=obstacle_struct)
    x_t = transition_info['desired state']
    if params_name[0:3] == "1st":
        action_t = transition_info["desired velocity"]
    else:
        action_t = transition_info["desired acceleration"]
    # Update plot
    trajectory_plotter.update(x_t.T.cpu().detach().numpy())
plt.savefig(params.results_path + "images/results_simulate_ds.png")
plt.show()
