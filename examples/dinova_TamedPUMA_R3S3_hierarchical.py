import numpy as np
import pybullet
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.examples_helpers.TamedPUMA_hierarchical import TamedPUMAhierarchical
"""
Example of dinova gen3-lite running ModulationIK.
"""
class example_dinova_tamedpuma_R3S3_hierarchical(TamedPUMAhierarchical):
    def __init__(self, file_name="dinova_hierarchical_tomato", path_config="../pumafabrics/tamed_puma/config/", mode_NN="1st"):
        super().__init__(file_name=file_name, path_config=path_config, mode_NN=mode_NN)

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_dinova(params=self.params)

    def run_kuka_example(self):
        dof = self.params["dof"]
        goal_pos = self.params["goal_pos"]
        action = np.zeros(dof)
        self.initialize_environment()
        ob, *_ = self.env.step(action)

        q_init = ob['robot_0']["joint_state"]["position"][0:dof]

        self.initialize_example(q_init)

        # Initialize lists
        xee_list = []
        qdot_diff_list = []
        runtime_arguments = {}

        for w in range(self.params["n_steps"]):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            if self.params["nr_obst"]>0:
                obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                obstacles = []

            # recompute translation to goal pose:
            goal_pos = [goal_pos[i] + self.params["goal_vel"][i]*self.params["dt"] for i in range(len(goal_pos))]
            pybullet.addUserDebugPoints([goal_pos], [[1, 0, 0]], 5, 0.1)

            runtime_arguments["q"] = q
            runtime_arguments["qdot"] = qdot
            runtime_arguments["positions_obstacles"] = []
            runtime_arguments["goal_pos"] = goal_pos
            runtime_arguments["obstacles"] = obstacles

            self.fabrics_controller.set_defaults_from_observation(ob_robot=ob_robot)
            action, self.GOAL_REACHED, dist_to_goal, self.IN_COLLISION, x_t_action = self.run(runtime_arguments)
            pybullet.addUserDebugPoints(list([x_t_action[0:3]]), [[1, 0, 0]], 10, 0.1)

            ob, *_ = self.env.step(action)

            # result analysis:
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
        np.array((0, 0, 0, -0.13467712, -0.26750494,  0.04957501,  1.53952123,  1.4392644,  -1.57109087))
    ]
    positions_obstacles_list = [
        [[0.0, 0., 1.55], [0.5, 0., 3.1]],
    ]
    goal_pos_list = [
        [0.53858072, -0.04530622,  0.8580668]
    ]
    mode_NN = "1st"
    example_class = example_dinova_tamedpuma_R3S3_hierarchical(mode_NN=mode_NN)
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