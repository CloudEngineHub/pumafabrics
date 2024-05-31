from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
import numpy as np

# Parameters
params_name = '2nd_order_2D'
x_t_init = np.array([[0.5, 0.6, 0, 0],  [-0.75, 0.9, 0, 0], [0.9, -0.9, 0, 0], [-0.9, -0.9, 0, 0], [0.9, 0.9, 0, 0],
                     [0.9, 0.3, 0, 0], [-0.9, -0.1, 0, 0],
                     [-0.9, 0.0, 0, 0], [0.4, 0.4, 0, 0], [0.9, -0.1, 0, 0], [-0.9, -0.5, 0, 0], [0.9, -0.5, 0, 0]])  # initial states
simulation_length = 2000
results_base_directory = './'

# Load parameters
Params = getattr(importlib.import_module('params.' + params_name), 'Params')
params = Params(results_base_directory)
params.results_path += params.selected_primitives_ids + '/'
params.load_model = True

# Initialize framework
learner, _, data = initialize_framework(params, params_name, verbose=False)

# Initialize dynamical system

dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(x_t_init).cuda(), delta_t=0.1)

# Initialize trajectory plotter
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
fig.show()
trajectory_plotter = TrajectoryPlotter(fig, x0=x_t_init.T, pause_time=1e-5, goal=data['goals training'][0])

x_t = torch.FloatTensor(x_t_init).cuda()

# Simulate dynamical system and plot
for i in range(simulation_length):
    # Do transition
    x_t = dynamical_system.transition(space='task', x_t=x_t)['desired state']

    # Update plot
    trajectory_plotter.update(x_t.T.cpu().detach().numpy())

plt.savefig("simulate_ds_trial_plot")
