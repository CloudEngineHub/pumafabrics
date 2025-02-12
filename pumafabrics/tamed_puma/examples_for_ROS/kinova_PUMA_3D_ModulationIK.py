import os
import numpy as np
import warnings

# from pumafabrics.puma_adapted.datasets.dingo_kinova.reformat_dataset import root_link_name
from pumafabrics.tamed_puma.utils.filters import PDController
from pumafabrics.tamed_puma.utils.normalizations_2 import normalization_functions
from pumafabrics.tamed_puma.create_environment.environments import trial_environments
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
import importlib
from pumafabrics.puma_adapted.initializer import initialize_framework
import copy
import yaml
from pumafabrics.tamed_puma.modulation_ik.Modulation_ik import IKGomp
import time
from pumafabrics.tamed_puma.tamedpuma.example_generic import ExampleGeneric

class example_kuka_PUMA_modulationIK(ExampleGeneric):
    def __init__(self, file_name="kinova_ModulationIK_tomato", path_config="../pumafabrics/tamed_puma/config/"):
        super(ExampleGeneric, self).__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(current_dir, path_config)
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.solver_times = []
        with open(config_dir + file_name + ".yaml", "r") as setup_stream:
             self.params = yaml.safe_load(setup_stream)
        self.dof = self.params["dof"]
        self.robot_name = self.params["robot_name"]
        warnings.filterwarnings("ignore")

    def initialize_environment(self):
        envir_trial = trial_environments()
        (self.env, self.goal) = envir_trial.initialize_environment_kuka(params=self.params)

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/../config/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )

    def construct_example(self, with_environment=True, results_base_directory='../pumafabrics/puma_adapted/'):
        # Construct classes:
        # results_base_directory = '../pumafabrics/puma_adapted/'
        
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"],
                                              robot_name=self.params["robot_name"],
                                              root_link_name=self.params["root_link"],
                                              end_link_name=self.params["end_links"][0])
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])
        self.construct_fk()
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"],
                                            kinematics=self.kuka_kinematics)

        # Parameters
        if self.params["mode_NN"] == "1st":
            self.params_name = self.params["params_name_1st"]
        else:
            self.params_name = self.params["params_name_2nd"]

        # Load parameters
        Params = getattr(importlib.import_module('pumafabrics.puma_adapted.params.' + self.params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        self.learner, _, data = initialize_framework(params, self.params_name, verbose=False)
        self.goal_NN = data['goals training'][0]
        print("self.goal_NN in construction:", self.goal_NN)

        # Initialize GOMP
        self.gomp_class  = IKGomp(  q_home=self.params["init_pos"],
                                    end_link_name = self.params["end_links"][0],
                                    robot_name = self.params["robot_name"],
                                    root_link_name=self.params["root_link"])
        self.gomp_class.construct_ik(nr_obst=self.params["nr_obst"], collision_links=self.params["collision_links"])

        # Normalization class
        self.normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"])


    def initialize_example(self, q_init):
        self.offset_orientation = np.array(self.params["orientation_goal"])

        # Translation of goal:
        goal_pos = self.params["goal_pos"]
        self.translation_gpu, self.translation_cpu = self.normalizations.translation_goal(state_goal = np.array(goal_pos), goal_NN=self.goal_NN)
        print("goal_pos:", goal_pos)
        # initial state:
        x_t_init = self.gomp_class.get_initial_pose(q_init=q_init, offset_orientation=self.offset_orientation)
        x_t, _ = self.gomp_class.get_current_pose(q=q_init, quat_prev=x_t_init[3:])
        x_init_gpu = self.normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=self.translation_cpu, offset_orientation=self.offset_orientation)
        self.dynamical_system = self.learner.init_dynamical_system(initial_states=x_init_gpu[:, :3].clone(), delta_t=1)

        self.quat_prev = copy.deepcopy(x_t_init[3:])

    def run(self, runtime_arguments):
        q = runtime_arguments["q"]
        qdot = runtime_arguments["qdot"]
        goal_pos = runtime_arguments["goal_pos"]
        print("goal_pos:", goal_pos)
        positions_obstacles = runtime_arguments["positions_obstacles"]

        # recompute translation to goal pose:
        self.translation_gpu, self.translation_cpu = self.normalizations.translation_goal(state_goal=np.array(goal_pos), goal_NN=self.goal_NN)

        # --- end-effector states and normalized states --- #
        x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, self.quat_prev)
        # print("x_t:", x_t)
        x_t, xee_orientation = self.gomp_class.get_current_pose(q, quat_prev=self.quat_prev)
        # print("x_t gomp:", x_t)
        # print("self.goal_NN:", self.goal_NN)

        self.quat_prev = copy.deepcopy(xee_orientation)
        vel_ee, Jac_current = self.kuka_kinematics.get_state_velocity(q=q, qdot=qdot)
        x_t_gpu = self.normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=self.translation_cpu, offset_orientation=self.offset_orientation)

        # --- action by NN --- #
        time0 = time.perf_counter()
        centers_obstacles = [[100, 100, 100], [100, 100, 100]]
        for i in range(self.params["nr_obst"]):
            centers_obstacles[i][0:3] = self.params["positions_obstacles"][i]
        obstacles_struct = {"centers": centers_obstacles,
                           "axes": [[0.3, 0.3, 0.3]], "safety_margins": [[1., 1., 1.]]}
        transition_info = self.dynamical_system.transition(space='task', x_t=x_t_gpu[:, :3].clone(), obstacles=obstacles_struct)
        x_t_NN = transition_info["desired state"]
        if self.params["mode_NN"] == "2nd":
            print("not implemented!!")
        else:
            action_t_gpu = transition_info["desired velocity"]
            action_cpu = action_t_gpu.T.cpu().detach().numpy()
        x_t_action = self.normalizations.reverse_transformation_position(position_gpu=x_t_NN) #, offset_orientation=offset_orientation)
        action_safeMP = self.normalizations.reverse_transformation(action_gpu=action_t_gpu)
        
        q_d, solver_flag = self.gomp_class.call_ik(x_t_action[0][0:3], self.params["orientation_goal"],
                                                  positions_obsts=positions_obstacles,
                                                  q_init_guess=q,
                                                  q_home=q)
        xee_IK, _ = self.gomp_class.get_current_pose(q=q_d, quat_prev=self.quat_prev)
        # print("solver_flag:", solver_flag)
        print("q_d:", q_d)
        print("q:", q)
        action = self.pdcontroller.control(desired_velocity=q_d, current_velocity=q)
        print("action unclipped: ", action)
        self.solver_times.append(time.perf_counter() - time0)
        action = np.clip(action, -1*np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))

        self.check_goal_reached(x_ee=x_t[0][0:3], x_goal=goal_pos)
        dist_to_goal = self.return_distance_goal_reached()

        # result analysis:
        # x_ee, _ = self.utils_analysis._request_ee_state(q, self.quat_prev)
        #self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=self.obstacles, parent_link=self.params["root_link"])
        self.GOAL_REACHED, _ = self.utils_analysis.check_goal_reaching(q, self.quat_prev, x_goal=goal_pos)

        # if self.GOAL_REACHED:
        #     self.time_to_goal = w*self.params["dt"]
        
        return action, self.GOAL_REACHED, dist_to_goal




def main(render=True):
    q_init_list = [
        # with goal changing:
        np.array((1.14515927e-04, 3.48547333e-01, 1.57849233e+00, 2.58859258e-04, -1.11692976e-03, 1.42772066e-03))
    ]
    positions_obstacles_list = [
        # # with goal changing:
        [[10.0, 0., 0.55], [0.5, 0., 10.1]],
    ]
    goal_pos_list = [
        # # #changing goal pose:
        [0.53858072, -0.04530622,  0.4580668]
    ]

    example_class = example_kuka_PUMA_modulationIK()
    example_class.overwrite_defaults(params=example_class.params, init_pos=q_init_list[0],
                                     goal_pos=goal_pos_list[0],
                                     positions_obstacles=positions_obstacles_list[0],
                                     render=render)
    example_class.construct_example()
    example_class.initialize_example(q_init=q_init_list[0])
    example_class.run()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])
    return {}

if __name__ == "__main__":
    main()