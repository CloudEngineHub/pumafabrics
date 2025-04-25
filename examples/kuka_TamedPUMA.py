import numpy as np
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
import pybullet
from pumafabrics.tamed_puma.examples_helpers.TamedPUMA import TamedPUMAExample
"""
Example of KUKA iiwa 14 running TamedPUMA as a controller.
"""
class example_kuka_TamedPUMA(TamedPUMAExample):
    def __init__(self, file_name="kuka_TamedPUMA_tomato"):
        super().__init__(file_name)

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_robots(params=self.params)

    def run_kuka_example(self):
        # --- parameters --- #
        n_steps = self.params["n_steps"]
        goal_pos = self.params["goal_pos"]
        dof = self.params["dof"]
        action = np.zeros(dof)
        self.initialize_environment()
        ob, *_ = self.env.step(action)
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]

        self.initialize_example(q_init)

        # Initialize lists
        xee_list = []
        qdot_diff_list = []
        runtime_arguments = {}

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            if self.params["nr_obst"]>0:
                obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                obstacles = []

            runtime_arguments["q"] = q
            runtime_arguments["qdot"] = qdot
            runtime_arguments["positions_obstacles"] = []
            runtime_arguments["goal_pos"] = goal_pos
            runtime_arguments["obstacles"] = obstacles

            goal_pos = [goal_pos[i] + self.params["goal_vel"][i]*self.params["dt"] for i in range(len(goal_pos))]
            pybullet.addUserDebugPoints([goal_pos], [[1, 0, 0]], 5, 0.1)
            self.fabrics_controller.set_defaults_from_observation(ob_robot=ob_robot)

            action, self.GOAL_REACHED, dist_to_goal, self.IN_COLLISION, _, qddot_PUMA = self.run(runtime_arguments)
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, self.quat_prev)
            xee_list.append(x_ee[0])
            qdot_diff_list.append(np.mean(np.absolute(qddot_PUMA   - action)))
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
        np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
    ]
    positions_obstacles_list = (
        [[0.5, 0.15, 0.05], [0.5, 0.15, 0.2]],
    )
    speed_obstacles_list = [
        [[0., 0., 0.], [0., 0., 0.]],
    ]
    goal_pos_list = [
        [0.58, -0.214, 0.115],
    ]
    goal_vel_list = [
        [0., 0., 0.] for _ in range(len(q_init_list))
    ]
    goal_vel_list[0] = [-0.01, 0., 0.]
    network_yaml = "kuka_TamedPUMA_tomato"
    example_class = example_kuka_TamedPUMA(file_name=network_yaml)
    index = 0
    example_class.overwrite_defaults(params=example_class.params, init_pos=q_init_list[index], positions_obstacles=positions_obstacles_list[index],
                                     render=render, speed_obstacles=speed_obstacles_list[index], n_steps=n_steps,
                                     goal_pos=goal_pos_list[index], goal_vel=goal_vel_list[index])
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