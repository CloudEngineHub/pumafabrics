import numpy as np
import pybullet
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.examples_helpers.PUMA_3D_ModulationIK import PUMA_modulationIK
"""
Example of Kinova gen3-lite running ModulationIK.
"""
class example_kinova_PUMA_modulationIK(PUMA_modulationIK):
    def __init__(self, file_name="kinova_ModulationIK_tomato", path_config="../pumafabrics/tamed_puma/config/"):
        super().__init__(file_name=file_name, path_config=path_config)

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_robots(params=self.params)

    def run_kuka_example(self):
        dof = self.params["dof"]
        goal_pos = self.params["goal_pos"]

        action = np.zeros(dof)
        self.initialize_environment()
        ob, *_ = self.env.step(action)
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]

        self.initialize_example(q_init=q_init)

        xee_list = []
        qdot_diff_list = []
        runtime_arguments = {}

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            if self.params["nr_obst"] > 0:
                positions_obstacles = [ob_robot["FullSensor"]["obstacles"][self.params["nr_obst"]]["position"], ob_robot["FullSensor"]["obstacles"][self.params["nr_obst"]+1]["position"]]
                obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                positions_obstacles = []
                obstacles = []
            goal_pos = [goal_pos[i] + self.params["goal_vel"][i] * self.params["dt"] for i in range(len(goal_pos))]

            runtime_arguments["q"] = q
            runtime_arguments["qdot"] = qdot
            runtime_arguments["positions_obstacles"] = positions_obstacles
            runtime_arguments["goal_pos"] = goal_pos
            runtime_arguments["obstacles"] = obstacles

            action, self.GOAL_REACHED, dist_to_goal, self.IN_COLLISION, x_t_action = self.run(runtime_arguments)
            ob, *_ = self.env.step(action)

            # result analysis:
            pybullet.addUserDebugPoints(list(x_t_action), [[1, 0, 0]], 5, 0.1)
            pybullet.addUserDebugPoints([goal_pos], [[1, 0, 0]], 5, 0.1)
            x_ee, _ = self.utils_analysis._request_ee_state(q, self.quat_prev)
            xee_list.append(x_ee[0])
            qdot_diff_list.append(1)

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
            "qdot_diff_list": qdot_diff_list,
            "solver_times": np.array(self.solver_times)*1000,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        return results

def main(render=True, n_steps=None):
    q_init_list = [
        np.array((-0.13467712, -0.26750494,  0.04957501,  1.53952123,  1.4392644,  -1.57109087))
    ]
    positions_obstacles_list = [
        [[10.0, 0., 0.55], [0.5, 0., 10.1]],
    ]
    goal_pos_list = [
        [0.53858072, -0.04530622,  0.4580668]
    ]

    example_class = example_kinova_PUMA_modulationIK()
    example_class.overwrite_defaults(params=example_class.params, init_pos=q_init_list[0],
                                     goal_pos=goal_pos_list[0],
                                     positions_obstacles=positions_obstacles_list[0],
                                     render=render, n_steps=n_steps)
    example_class.construct_example()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
    return {}

if __name__ == "__main__":
    main()