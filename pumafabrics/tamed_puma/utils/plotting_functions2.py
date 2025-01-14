import numpy as np
import matplotlib.pyplot as plt
from pumafabrics.puma_adapted.background_vectorfield import create_background_vectorfield

class plotting_functions2():
    def __init__(self, results_path=""):
        self.results_path = results_path

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

    def plotting_q_values(self, q_list, dt=0.01, q_start=np.array([0, 0]), q_goal=np.array([0, 0])):
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
        plt.savefig(self.results_path+"images/configurations_simulation.png")



    def velocities_over_time(self, quat_vel_list, ang_vel_list, joint_vel_list, action_list, dt=0.01):
        time_x = np.arange(0.0, len(quat_vel_list.transpose()) * dt, dt)
        fig, ax = plt.subplots(3, 1)

        ax[0].plot(time_x, quat_vel_list[0, :])
        ax[0].plot(time_x, quat_vel_list[1, :])
        ax[0].plot(time_x, quat_vel_list[2, :])
        ax[0].plot(time_x, quat_vel_list[3, :])
        ax[0].set(xlabel="time [s]", ylabel="quat vel ", title="Quaternion velocities")
        ax[0].grid()

        ax[1].plot(time_x, ang_vel_list[0, :])
        ax[1].plot(time_x, ang_vel_list[1, :])
        ax[1].plot(time_x, ang_vel_list[2, :])
        ax[1].set(xlabel="time [s]", ylabel="ang vel ", title="Angular velocities")
        ax[1].grid()

        ax[2].plot(time_x, joint_vel_list[0, :])
        ax[2].plot(time_x, joint_vel_list[1, :])
        ax[2].plot(time_x, joint_vel_list[2, :])
        ax[2].plot(time_x, joint_vel_list[3, :])
        ax[2].set(xlabel="time [s]", ylabel="$q_{dot}$", title="Joint velocities")

        ax[2].plot(time_x, action_list[0, :], '--')
        ax[2].plot(time_x, action_list[1, :], '--')
        ax[2].plot(time_x, action_list[2, :], '--')
        ax[2].plot(time_x, action_list[3, :], '--')
        ax[2].set(xlabel="time [s]", ylabel="$q_{dot}$", title="Actions (vel)")
        ax[2].grid()

        plt.show()

    def pose_over_time(self, quat_list, dt=0.01):
        time_x = np.arange(0.0, len(quat_list.transpose()) * dt, dt)
        fig, ax = plt.subplots(3, 1)

        ax[0].plot(time_x, quat_list[0, :])
        ax[0].plot(time_x, quat_list[1, :])
        ax[0].plot(time_x, quat_list[2, :])
        ax[0].plot(time_x, quat_list[3, :])
        ax[0].set(xlabel="time [s]", ylabel="quaternion ", title="Quaternions")
        ax[0].grid()

        plt.show()
        # ax.grid()
        # # ax.set_xlim(scaling_room["x"][0], scaling_room["x"][1])
        # # ax.set_ylim(scaling_room["y"][0], scaling_room["y"][1])
        # ax.set(xlabel="x [m]", ylabel="y [m]", title="Configurations q")
        # ax.legend(["q", "start", "end"])
        # plt.savefig(self.results_path+"images/configurations_simulation.png")

    def position_over_time(self, q_list, qdot_list, dt=0.01):
        time_x = np.arange(0.0, len(q_list.transpose()) * dt, dt)
        fig, ax = plt.subplots(3, 1)

        ax[0].plot(time_x, q_list[0, :])
        ax[0].set(xlabel="time [s]", ylabel="position ", title="Position")
        ax[0].grid()

        ax[1].plot(time_x, qdot_list[0, :])
        ax[1].set(xlabel="time [s]", ylabel="qdot", title="Velocity")
        ax[1].grid()
        plt.show()
        plt.close('all')

    def actions_over_time(self, action_list, dt=1):
        time_x = np.arange(0.0, len(action_list.transpose()) * dt, dt)
        fig, ax = plt.subplots(2, 1)


        ax[0].plot(time_x, action_list[0, :])
        ax[0].plot(time_x, action_list[1, :])
        if len(action_list)>2:
            ax[0].plot(time_x, action_list[2, :])
        if len(action_list)>3:
            ax[0].plot(time_x, action_list[3, :])
            ax[0].plot(time_x, action_list[4, :])
            ax[0].plot(time_x, action_list[5, :])
            ax[0].plot(time_x, action_list[6, :])

        plt.show()

