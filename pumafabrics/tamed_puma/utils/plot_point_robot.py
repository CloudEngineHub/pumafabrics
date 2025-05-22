import numpy as np
import matplotlib.pyplot as plt
from pumafabrics.puma_extension.background_vectorfield import create_background_vectorfield
import os

class plotting_functions():
    def __init__(self, results_path=""):
        self.results_path = results_path
        self.dir_path = "images"

    def generate_background_NN(self):
        # --- run background with NN ---#
        background_class = create_background_vectorfield()
        [x_field, y_field, vx_field, vy_field] = background_class.get_background_vectors()
        return x_field, y_field, background_class

    def plotting_x_values(self, x_list, dt=0.01, x_start=np.array([0, 0]), x_goal=np.array([0, 0]), scaling_room=dict):
        time_x = np.arange(0.0, len(x_list) * dt, dt)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x_list[0, :], x_list[1, :], '--', color="b")
        ax.plot(x_start[0], x_start[1], "o", color='g')
        ax.plot(x_goal[0], x_goal[1], "x", color='r')
        ax.grid()
        ax.set_xlim(scaling_room["x"][0], scaling_room["x"][1])
        ax.set_ylim(scaling_room["y"][0], scaling_room["y"][1])
        ax.set(xlabel="x [m]", ylabel="y [m]", title="Configurations q")
        ax.legend(["x", "start", "end"])
        plt.savefig(self.results_path+"xy_simulation.png")

    def plotting_q_values(self, q_list, dt=0.01, q_start=np.array([0, 0]), q_goal=np.array([0, 0]), file_name="configurations_simulation"):
        time_x = np.arange(0.0, len(q_list) * dt, dt)
        fig, ax = plt.subplots(1, 1)
        ax.plot(q_list[0, :], q_list[1, :], '--', color="b")
        ax.plot(q_start[0], q_start[1], "o", color='g')
        ax.plot(q_goal[0], q_goal[1], "x", color='r')
        ax.grid()
        # ax.set_xlim(scaling_room["x"][0], scaling_room["x"][1])
        # ax.set_ylim(scaling_room["y"][0], scaling_room["y"][1])
        ax.set(xlabel="x [m]", ylabel="y [m]", title="Configurations q")
        ax.legend(["q", "start", "end"])
        if not os.path.isdir(self.dir_path):
            os.makedirs(self.dir_path)
        plt.savefig(self.results_path+file_name+".png")


    def comparison_plot(self, q_list_0, q_list_1, q_list_3, q_list_4, q_list_5=None, q_goal=np.array([0.0, 0.0]), q_start=np.array([0.0, 0.0]), dt=0.01, scaling_room=None):
        time_x = np.arange(0.0, len(q_list_0) * dt, dt)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 4)

        #--- plot safe MP ---#
        line_1, = ax.plot(q_list_1[0, :], q_list_1[1, :], "--", color="b")
        line_1.set_label("PUMA (data-driven)")

        #--- plot fabrics ---#
        line_0, = ax.plot(q_list_0[0, :], q_list_0[1, :], "--", color="k")
        line_0.set_label("Fabrics (geometric)")

        #--- plot safeMP + fabrics
        line_3, = ax.plot(q_list_3[0, :], q_list_3[1, :], color="cyan")
        line_3.set_label("FPM (ours)")

        #--- plot theorem III.5 ---#
        line_4, = ax.plot(q_list_4[0, :], q_list_4[1, :], color='tab:orange')
        line_4.set_label("CPM (ours)")

        # #--- plot hierarchical safe MP + fabrics ---#
        if q_list_5 is not None:
            line_5, = ax.plot(q_list_5[0, :], q_list_5[1, :], color="g")
            line_5.set_label("Hierarchical")

        # initial and final position
        start_point = ax.scatter(q_start[0], q_start[1], marker="o", color="g", zorder=3, s=100)
        end_point = ax.scatter(q_goal[0], q_goal[1], marker="*", color="g", zorder=3, s=100)
        start_point.set_label("Start")
        end_point.set_label("Target")

        # obstacles
        obstacle_1 = plt.Circle((0.0, -6.5), radius=0.9, color=[0.4, 0.2, 0.6, 1.])
        ax.add_artist(obstacle_1)
        obstacle_1.set_label("Obstacles")
        obstacle_1 = plt.Circle((-2.0, -4.0), radius=1.2, color=[0.4, 0.2, 0.6, 1.])
        ax.add_artist(obstacle_1)
        # obstacle_1 = plt.Circle((5.0, -3.0), radius=1.0, color="lightgray")
        # ax.add_artist(obstacle_1)

        # plot background NN in gray:
        x_field, y_field, background_class = self.generate_background_NN()
        background_class.plot_backgroundNN(ax, x_field, y_field)

        ax.grid()
        # ax.set_xlim(scaling_room["x"][0], scaling_room["x"][1])
        # ax.set_ylim(scaling_room["y"][0], scaling_room["y"][1])
        ax.set_xlim(-5.5, 8.2)
        ax.set_ylim(-10.5, 7.2)
        ax.set(xlabel="x [m]", ylabel="y [m]") #, title="Trajectories of the proposed methods on a point-mass example", size=20)
        ax.legend(loc="upper left", fontsize=9.5)
        # ax.legend(["GFs", "SMPs", "Geometric", "Compatible", "Start", "Target", "Obstacles"], loc="upper left")
        plt.suptitle("Trajectories of the proposed methods on a point-mass example", fontsize=13.5)
        plt.savefig("images/comparison_plot.eps", format="eps") #, dpi=100)
        plt.savefig("images/comparison_plot.jpg", format="jpg")  # , dpi=100)
        plt.show()

    def comparison_plot_iiwa(self, results=dict, cases=dict, dt=0.02):
        results_GM = results["GM"]
        xee_list_GM = results_GM["xee_list"][0]
        xee_GM = np.array(xee_list_GM).transpose()

        time_x = np.arange(0.0, len(xee_list_GM) * dt, dt)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 4)

        line_3, = ax.plot(xee_GM[0, :], xee_GM[1, :], color="cyan")
        line_3.set_label("Geometric")

        ax.grid()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set(xlabel="x [m]", ylabel="y [m]") #, title="Trajectories of the proposed methods on a point-mass example", size=20)
        ax.legend(loc="upper left", fontsize=9.5)
        plt.suptitle("Trajectories of the proposed methods on a point-mass example", fontsize=13.5)
        plt.savefig("comparison_plot.eps", format="eps") #, dpi=100)
        plt.savefig("comparison_plot.jpg", format="jpg")  # , dpi=100)
        plt.show()

