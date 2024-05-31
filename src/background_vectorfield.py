from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
import numpy as np
from evaluation.evaluator_init import evaluator_init


class create_background_vectorfield():
    def __init__(self):
        dt = 0.01

    def get_background_vectors(self):
        # Parameters
        params_name = '2nd_order_2D'
        # x_t_init = np.array([[0.5, 0.6], [-0.75, 0.9], [0.9, -0.9], [-0.9, -0.9], [0.9, 0.9], [0.9, 0.3], [-0.9, -0.1],
        #                      [-0.9, 0.0], [0.4, 0.4], [0.9, -0.1], [-0.9, -0.5], [0.9, -0.5]])  # initial states
        # x_t_init = np.array([[0.5, 0.6, 0., 0.], [-0.75, 0.9, 0., 0.], [0.9, -0.9, 0., 0.], [-0.9, -0.9, 0., 0.], [0.9, 0.9, 0., 0.],
        #                      [0.9, 0.3, 0., 0.], [-0.9, -0.1, 0., 0.], [-0.9, 0.0, 0., 0.], [0.4, 0.4, 0., 0.], [0.9, -0.1, 0., 0.],
        #                      [-0.9, -0.5, 0., 0.], [0.9, -0.5, 0., 0.]])  # initial states
        x_0 = np.linspace(-0.5, 1, 5)
        y_0 = np.linspace(-1, 0.5, 5)
        x_grid, y_grid  = np.meshgrid(x_0, y_0)
        xy_0 = np.column_stack((x_grid.flatten(), y_grid.flatten()))
        v_0 = np.zeros((np.size(xy_0, 0), 1))
        x_t_init = np.column_stack((xy_0, v_0, v_0))
        # x_t_init = np.column_stack((xy_0))
        # x_t_init = xy_0
        # x_t_init = np.array([np.linspace(-10, 10, 100)])
        simulation_length = 2000
        results_base_directory = './'

        # Load parameters
        Params = getattr(importlib.import_module('params.' + params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, params_name, verbose=False)
        evaluation_class = evaluator_init(learner=learner, data=data, params=params)

        # Initialize dynamical system

        dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(x_t_init).cuda())


        # Initialize trajectory plotter
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        fig.show()
        # trajectory_plotter = TrajectoryPlotter(fig, x0=x_t_init.T, pause_time=1e-5, goal=data['goals training'][0])

        x_list = []
        y_list = []

        # Simulate dynamical system and plot
        for i in range(simulation_length):
            # Do transition
            x_t = dynamical_system.transition(space='task')['desired state']
            v_t = dynamical_system.transition(space='task')["desired velocity"]
            t_x_t = torch.t(10*x_t)
            x_t_cpu = t_x_t.cpu().detach().numpy()
            t_v_t = torch.t(v_t)
            v_t_cpu = t_v_t.cpu().detach().numpy()

            # # # Update plot
            # trajectory_plotter.update(x_t.T.cpu().detach().numpy())

            # append to lists
            x_list.append(x_t_cpu[0, :])
            y_list.append(x_t_cpu[1, :])
        # plt.show()
        # plt.savefig(params.results_path + "images/results_background_vectorfield.png")
        # x_field = x_t_cpu[0, :]
        # y_field = x_t_cpu[1, :]
        vx_field = [1, 1] #v_t_cpu[0, :]
        vy_field = [1, 1] #v_t_cpu[1, :]
        return x_list, y_list, vx_field, vy_field

    def add_arrow(self, ax, xdata, ydata, position=None, direction='right', size=15, color=None):
        """
        add an arrow to a line.

        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken.
        """

        color = "lightgrey"

        for w in [10, 100]:
            for i in range(len(xdata[0])):
                xinit = xdata[w][i]
                yinit = ydata[w][i]
                xdiff = xdata[w+5][i] - xinit
                ydiff = ydata[w+5][i] - yinit
                ax.arrow(xinit, yinit, xdiff, ydiff, shape='full', lw=0, length_includes_head=True, head_width=0.3, color=color, zorder=0)

        # plot initial starting points as markers
        for i in range(len(xdata[0])):
            ax.scatter(xdata[0][i], ydata[1][i], marker="o", color="lightgrey", zorder=0, s=30)

    def plot_backgroundNN(self, ax, x_field, y_field):
        background_lines = ax.plot(x_field, y_field, color="lightgrey", zorder=0)
        background_lines[0].set_label("Trajectories SMPs")
        self.add_arrow(ax, xdata=x_field, ydata=y_field, color="lightgrey")

    def plot_backgroundNN_figure(self, x_field, y_field, vx_field=0, vy_field=0):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 4)

        # Creating plot
        fig, ax = plt.subplots(figsize=(12, 7))
        self.plot_backgroundNN(ax, x_field, y_field)
        # ax.plot(x_field, y_field)
        # self.add_arrow(ax, xdata=x_field, ydata=y_field, color="lightgrey")
        plt.show()


if __name__ == "__main__":
    background_class = create_background_vectorfield()
    [x_field, y_field, vx_field, vy_field] = background_class.get_background_vectors()
    background_class.plot_backgroundNN_figure(x_field, y_field, vx_field, vy_field)