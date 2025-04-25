import numpy as np
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.examples_helpers.Fabrics_example import FabricsExample
import pybullet
"""
Example of the Kinova gen3-lite running Fabrics as a controller.
"""
class example_kinova_fabrics(FabricsExample):
    def __init__(self, file_name="kinova_TamedPUMA_tomato"):
        super().__init__(file_name=file_name)

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_robots(params=self.params)

    def run_kinova_example(self):
        # --- parameters --- #
        goal_pos = self.params["goal_pos"]
        dof = self.params["dof"]
        action = np.zeros(dof)
        self.initialize_environment()
        ob, *_ = self.env.step(action)
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]

        self.initialize_example(q_init)

        xee_list = []
        runtime_arguments = {}

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:self.dof]
            qdot = ob_robot["joint_state"]["velocity"][0:self.dof]

            if self.params["nr_obst"]>0:
                obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                obstacles = []
            # recompute goal position
            goal_pos = [goal_pos[i] + self.params["goal_vel"][i] * self.params["dt"] for i in range(len(goal_pos))]
            pybullet.addUserDebugPoints([goal_pos], [[1, 0, 0]], 5, 0.1)

            runtime_arguments["q"] = q
            runtime_arguments["qdot"] = qdot
            runtime_arguments["positions_obstacles"] = []
            runtime_arguments["goal_pos"] = goal_pos
            runtime_arguments["obstacles"] = obstacles
            self.fabrics_controller.set_defaults_from_observation(ob_robot=ob_robot)

            action, self.GOAL_REACHED, error, self.IN_COLLISION, _ =  self.run(runtime_arguments)
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q[0:dof], self.quat_prev)
            xee_list.append(x_ee[0])

            if self.GOAL_REACHED:
                self.time_to_goal = w*self.params["dt"]
                break

            if self.IN_COLLISION:
                self.time_to_goal = float("nan")
                break

        self.env.close()

        results = {
            "min_distance": self.utils_analysis.get_min_dist(),
            "collision": self.IN_COLLISION,
            "goal_reached": self.GOAL_REACHED,
            "time_to_goal": self.time_to_goal,
            "xee_list": xee_list,
            "solver_times": np.array(self.fabrics_controller.request_solver_times())*1000,
            "solver_time": np.mean(self.fabrics_controller.request_solver_times()),
            "solver_time_std": np.std(self.fabrics_controller.request_solver_times()),
        }
        return results

def main(render=True, n_steps=None):
    example_class = example_kinova_fabrics()
    q_init_list = [
        np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91, 1.)),
    ]
    positions_obstacles_list = [
        [[0.5, 0., 0.55], [0.5, 0., 10.1]],
    ]
    speed_obstacles_list = [
        [[0., 0., 0.], [0., 0., 0.]],
    ]
    goal_pos_list = [
        [0.58, -0.014, 0.115],
    ]
    goal_vel_list = [
        [0., 0., 0.] for _ in range(len(q_init_list))
    ]
    goal_vel_list[0] = [0., 0., 0.]
    id_nr = 0
    example_class.overwrite_defaults(params = example_class.params,
                                     init_pos=q_init_list[id_nr],
                                     positions_obstacles=positions_obstacles_list[id_nr],
                                     speed_obstacles=speed_obstacles_list[id_nr],
                                     goal_pos=goal_pos_list[id_nr],
                                     goal_vel=goal_vel_list[id_nr],
                                     render=render, n_steps=n_steps
                                     )
    example_class.construct_example()
    res = example_class.run_kinova_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
    return {}

if __name__ == "__main__":
    main()