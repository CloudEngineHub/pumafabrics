import os
import yaml
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.tamedpuma.parametrized_planner_extended import ParameterizedFabricPlannerExtended
from pumafabrics.tamed_puma.utils.normalizations_2 import normalization_functions
from pumafabrics.tamed_puma.tamedpuma.environments import trial_environments
from pumafabrics.tamed_puma.kinematics.kinematics_kuka import KinematicsKuka
from pumafabrics.tamed_puma.tamedpuma.energy_regulator import energy_regulation
import importlib
from pumafabrics.tamed_puma.nullspace_control.nullspace_controller import CartesianImpedanceController
from pumafabrics.puma_adapted.initializer import initialize_framework
from pumafabrics.tamed_puma.utils.analysis_utils import UtilsAnalysis
from pumafabrics.tamed_puma.utils.filters import PDController
import copy
import time

class example_kuka_stableMP_fabrics_20():
    def __init__(self): #, bool_energy_regulator=False, bool_combined=True, robot_name="iiwa14"):
        self.GOAL_REACHED = False
        self.IN_COLLISION = False
        self.time_to_goal = float("nan")
        self.solver_times = []
        with open("../pumafabrics/tamed_puma/config/kuka_stableMP_fabrics_2nd.yaml", "r") as setup_stream:
            self.params = yaml.safe_load(setup_stream)
        self.robot_name = self.params["robot_name"]

    def overwrite_defaults(self, render=None, init_pos=None, goal_pos=None, nr_obst=None, bool_energy_regulator=None, bool_combined=None, positions_obstacles=None, orientation_goal=None, params_name_1st=None):
        if render is not None:
            self.params["render"] = render
        if init_pos is not None:
            self.params["init_pos"] = init_pos
        if goal_pos is not None:
            self.params["goal_pos"] = goal_pos
        if orientation_goal is not None:
            self.params["orientation_goal"] = orientation_goal
        if nr_obst is not None:
            self.params["nr_obst"] = nr_obst
        if bool_energy_regulator is not None:
            self.params["bool_energy_regulator"] = bool_energy_regulator
        if bool_combined is not None:
            self.params["bool_combined"] = bool_combined
        if positions_obstacles is not None:
            self.params["positions_obstacles"] = positions_obstacles
        if params_name_1st is not None:
            self.params["params_name_1st"] = params_name_1st

    def check_goal_reached(self, x_ee, x_goal):
        dist = np.linalg.norm(x_ee - x_goal)
        if dist<0.02:
            self.GOAL_REACHED = True
            return True
        else:
            return False

    def initialize_environment(self):
        envir_trial = trial_environments()
        self.params["nr_obst"]=2
        (self.env, self.goal) = envir_trial.initialize_environment_kuka(params=self.params)
        self.params["nr_obst"]=1000

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )
    def set_planner(self, goal: GoalComposition): #, degrees_of_freedom: int = 7, mode="acc", dt=0.01, bool_speed_control=True):
        """
        Initializes the fabric planner for the panda robot.
        """
        if goal is not None:
            goal  = self.goal
        self.construct_fk()
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/examples/urdfs/"+self.robot_name+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        forward_kinematics = GenericURDFFk(
            urdf,
            root_link="iiwa_link_0",
            end_links=["iiwa_link_7"],
        )
        planner = ParameterizedFabricPlannerExtended(
            self.params["dof"],
            self.forward_kinematics,
            time_step=self.params["dt"],
            # collision_geometry=self.params["collision_geometry"],
            # collision_finsler=self.params["collision_finsler"],
        )
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            collision_links=self.params["collision_links"],
            goal=goal,
            number_obstacles=100, #self.params["nr_obst"],
            number_plane_constraints=0,
            limits=self.params["iiwa_limits"],
        )
        planner.concretize_extensive(mode=self.params["mode"], time_step=self.params["dt"], extensive_concretize=self.params["bool_extensive_concretize"], bool_speed_control=self.params["bool_speed_control"])
        return planner, forward_kinematics

    def compute_action_fabrics(self, q, ob_robot):
        nr_obst = self.params["nr_obst"]
        if nr_obst>19:
            x_obsts = [[10, 10, 10] for i in range(nr_obst)]
            x_obsts[0] = list(ob_robot['FullSensor']['obstacles'][2]['position'])
            x_obsts[1] = list(ob_robot['FullSensor']['obstacles'][3]['position'])
            radius_obsts = [[0.5] for i in range(nr_obst)]
            radius_obsts[0] = list(ob_robot['FullSensor']['obstacles'][2]['size'])
            radius_obsts[1] = list(ob_robot['FullSensor']['obstacles'][3]['size'])
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                x_obsts=x_obsts,
                radius_obsts=radius_obsts, #[radius_obsts.append([np.array([0.1])) for i in range(nr_obst-2)],
                radius_body_links=self.params["collision_radii"],
                constraint_0=np.array([0, 0, 1, 0.0]))
        elif nr_obst>0:
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                x_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
                radius_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['size'],
                x_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position'],
                radius_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size'],
                radius_body_links=self.params["collision_radii"],
                constraint_0=np.array([0, 0, 1, 0.0]))
        else:
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                radius_body_links=self.params["collision_radii"],
                constraint_0=np.array([0, 0, 1, 0.0]))
        M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = self.planner_avoidance.compute_M_f_action_avoidance(
            **arguments_dict)
        qddot_speed = np.zeros((self.params["dof"],))  # todo: think about what to do with speed regulation term!!
        return action_avoidance, M_avoidance, f_avoidance, qddot_speed

    def combine_action(self, M_avoidance, M_attractor, f_avoidance, f_attractor, qddot_speed, qdot = []):
        xddot_combined = -np.dot(self.planner_avoidance.Minv(M_avoidance + M_attractor), f_avoidance + f_attractor) + qddot_speed
        if self.planner_avoidance._mode == "vel":
            action_combined = qdot + self.planner_avoidance._time_step * xddot_combined
        else:
            action_combined = xddot_combined
        return action_combined

    def integrate_to_vel(self, qdot, action_acc, dt):
        qdot_action = action_acc *dt +qdot
        return qdot_action

    def vel_NN_rescale(self, transition_info, offset_orientation, xee_orientation, normalizations, kuka_kinematics):
        action_t_gpu = transition_info["desired velocity"]
        action_stableMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, mode_NN="1st") #because we use velocity action!
        action_quat_vel = action_stableMP[3:]
        action_quat_vel_sys = kuka_kinematics.quat_vel_with_offset(quat_vel_NN=action_quat_vel,
                                                                   quat_offset=offset_orientation)
        xdot_pos_quat = np.append(action_stableMP[:3], action_quat_vel_sys)

        # --- if necessary, also get rpy velocities corresponding to quat vel ---#
        vel_rpy = kuka_kinematics.quat_vel_to_angular_vel(angle_quaternion=xee_orientation,
                                                            vel_quaternion=xdot_pos_quat[3:7]) / self.params["dt"]  # action_quat_vel
        return xdot_pos_quat, vel_rpy

    def acc_NN_rescale(self, transition_info, offset_orientation, xee_orientation, normalizations, kuka_kinematics):
        action_t_gpu = transition_info["desired acceleration"]
        action_stableMP = normalizations.reverse_transformation(action_gpu=action_t_gpu, mode_NN="2nd") #because we use velocity action!
        action_quat_acc = action_stableMP[3:]
        action_quat_acc_sys = kuka_kinematics.quat_vel_with_offset(quat_vel_NN=action_quat_acc,
                                                                   quat_offset=offset_orientation)
        xddot_pos_quat = np.append(action_stableMP[:3], action_quat_acc_sys)
        return xddot_pos_quat

    def construct_example(self):
        self.initialize_environment()
        self.planner_avoidance, self.fk = self.set_planner(goal=None)
        self.kuka_kinematics = KinematicsKuka(dt=self.params["dt"], end_link_name=self.params["end_links"][0], robot_name=self.params["robot_name"])
        self.utils_analysis = UtilsAnalysis(forward_kinematics=self.forward_kinematics,
                                            collision_links=self.params["collision_links"],
                                            collision_radii=self.params["collision_radii"])
        self.pdcontroller = PDController(Kp=1.0, Kd=0.1, dt=self.params["dt"])
        self.controller_nullspace = CartesianImpedanceController(robot_name=self.params["robot_name"])

    def run_kuka_example(self): #, n_steps=2000, goal_pos=[-0.24355761, -0.75252747, 0.5], mode="acc", mode_NN = "1st", dt=0.01, mode_env=None):
        # --- parameters --- #
        n_steps = self.params["n_steps"]
        orientation_goal = np.array(self.params["orientation_goal"])
        offset_orientation = np.array(self.params["orientation_goal"])
        goal_pos = self.params["goal_pos"]
        dof = self.params["dof"]
        action = np.zeros(dof)
        ob, *_ = self.env.step(action)

        # Construct classes:
        results_base_directory = './'

        # Parameters
        if self.params["mode_NN"] == "1st":
            self.params_name = self.params["params_name_1st"]
        else:
            self.params_name = self.params["params_name_2nd"]
        print("self.params_name:", self.params_name)
        q_init = ob['robot_0']["joint_state"]["position"][0:dof]

        # Load parameters
        Params = getattr(importlib.import_module('params.' + self.params_name), 'Params')
        params = Params(results_base_directory)
        params.results_path += params.selected_primitives_ids + '/'
        params.load_model = True

        # Initialize framework
        learner, _, data = initialize_framework(params, self.params_name, verbose=False)
        goal_NN = data['goals training'][0]

        # Normalization class
        normalizations = normalization_functions(x_min=data["x min"], x_max=data["x max"], dof_task=self.params["dim_task"], dt=self.params["dt"], mode_NN=self.params["mode_NN"], learner=learner)

        # Translation of goal:
        translation_gpu, translation_cpu = normalizations.translation_goal(state_goal = np.append(goal_pos, orientation_goal), goal_NN=goal_NN)

        # initial state:
        x_t_init = self.kuka_kinematics.get_initial_state_task(q_init=q_init, qdot_init=np.zeros((dof, 1)), offset_orientation=offset_orientation, mode_NN=self.params["mode_NN"])
        x_init_gpu = normalizations.normalize_state_to_NN(x_t=[x_t_init], translation_cpu=translation_cpu, offset_orientation=offset_orientation)
        dynamical_system = learner.init_dynamical_system(initial_states=x_init_gpu, delta_t=1)

        # energization:
        energy_regulation_class = energy_regulation(dim_task=7, mode_NN=self.params["mode_NN"], dof=dof, dynamical_system=dynamical_system)
        energy_regulation_class.relationship_dq_dx(offset_orientation, translation_cpu, self.kuka_kinematics, normalizations, self.fk)

        # Initialize lists
        xee_list = []
        qdot_diff_list = []
        quat_prev = copy.deepcopy(x_t_init[3:7])
        Jac_dot_prev = np.zeros((7, 7))
        Jac_prev = np.zeros((7, 7))
        time_list = []

        for w in range(n_steps):
            # --- state from observation --- #
            ob_robot = ob['robot_0']
            q = ob_robot["joint_state"]["position"][0:dof]
            qdot = ob_robot["joint_state"]["velocity"][0:dof]
            if self.params["nr_obst"]>0 and self.params["nr_obst"]<20:
                self.obstacles = list(ob["robot_0"]["FullSensor"]["obstacles"].values())
            else:
                self.obstacles = []

            # --- end-effector states and normalized states --- #
            x_t, xee_orientation, _ = self.kuka_kinematics.get_state_task(q, quat_prev, mode_NN=self.params["mode_NN"], qdot=qdot)
            quat_prev = copy.deepcopy(xee_orientation)
            x_t_gpu = normalizations.normalize_state_to_NN(x_t=x_t, translation_cpu=translation_cpu, offset_orientation=offset_orientation)

            # --- action by NN --- #
            time0 = time.perf_counter()
            transition_info = dynamical_system.transition(space='task', x_t=x_t_gpu)
            time00 = time.perf_counter()
            time_list.append(time00 - time0)

            # # -- transform to configuration space --#
            # --- rescale velocities and pose (correct offset and normalization) ---#
            xdot_pos_quat, euler_vel = self.vel_NN_rescale(transition_info, offset_orientation, xee_orientation, normalizations, self.kuka_kinematics)
            xddot_pos_quat = self.acc_NN_rescale(transition_info, offset_orientation, xee_orientation, normalizations, self.kuka_kinematics)
            x_t_action = normalizations.reverse_transformation_pos_quat(state_gpu=transition_info["desired state"], offset_orientation=offset_orientation)

            # ---- velocity action_stableMP: option 1 ---- #
            qdot_stableMP_pulled = self.kuka_kinematics.inverse_diff_kinematics_quat(xdot=xdot_pos_quat,
                                                                                    angle_quaternion=xee_orientation).numpy()[0]
            #### --------------- directly from acceleration!! -----#
            qddot_stableMP, Jac_prev, Jac_dot_prev = self.kuka_kinematics.inverse_2nd_kinematics_quat(q=q, qdot=qdot_stableMP_pulled, xddot=xddot_pos_quat, angle_quaternion=xee_orientation, Jac_prev=Jac_prev)
            qddot_stableMP = qddot_stableMP.numpy()[0]
            action_nullspace = self.controller_nullspace._nullspace_control(q=q, qdot=qdot)
            qddot_stableMP = qddot_stableMP + action_nullspace
            #qddot_stableMP = self.pdcontroller.control(desired_velocity=qdot_stableMP_pulled, current_velocity=qdot)

            if self.params["bool_combined"] == True:
                # ----- Fabrics action ----#
                action_avoidance, M_avoidance, f_avoidance, qddot_speed = self.compute_action_fabrics(q=q, ob_robot=ob_robot)

                if self.params["bool_energy_regulator"] == True:
                    # ---- get action by NN via theorem III.5 in https://arxiv.org/pdf/2309.07368.pdf ---#
                    action_combined = energy_regulation_class.compute_action_theorem_III5(q=q, qdot=qdot,
                                                                                          qddot_attractor = qddot_stableMP,
                                                                                          action_avoidance=action_avoidance,
                                                                                          M_avoidance=M_avoidance,
                                                                                          transition_info=transition_info)
                else:
                    # --- get action by a simpler combination, sum of dissipative systems ---#
                    action_combined = qddot_stableMP + action_avoidance
            else: #otherwise only apply action by stable MP
                action_combined = qddot_stableMP

            if self.params["mode_env"] is not None:
                if self.params["mode_env"] == "vel": # todo: fix nicely or mode == "acc"): #mode_NN=="2nd":
                    action = self.integrate_to_vel(qdot=qdot, action_acc=action_combined, dt=self.params["dt"])
                    action = np.clip(action, -1*np.array(self.params["vel_limits"]), np.array(self.params["vel_limits"]))
                else:
                    action = action_combined
            else:
                action = action_combined
            self.solver_times.append(time.perf_counter() - time0)
            ob, *_ = self.env.step(action)

            # result analysis:
            x_ee, _ = self.utils_analysis._request_ee_state(q, quat_prev)
            xee_list.append(x_ee[0])
            qdot_diff_list.append(np.mean(np.absolute(qddot_stableMP   - action_combined)))
            self.IN_COLLISION = self.utils_analysis.check_distance_collision(q=q, obstacles=self.obstacles)
            self.GOAL_REACHED, error = self.utils_analysis.check_goal_reaching(q, quat_prev, x_goal=goal_pos)
            if self.GOAL_REACHED:
                self.time_to_goal = w*self.params["dt"]
                break

            if self.IN_COLLISION:
                self.time_to_goal = float("nan")
                break
        self.env.close()

        print("time network average:", np.array(time_list).mean())
        print("standard deviation", np.array(time_list).std())

        results = {
            "min_distance": self.utils_analysis.get_min_dist(),
            "collision": self.IN_COLLISION,
            "goal_reached": self.GOAL_REACHED,
            "time_to_goal": self.time_to_goal,
            "xee_list": xee_list,
            "qdot_diff_list": qdot_diff_list,
            "solver_times": self.solver_times,
            "solver_time": np.mean(self.solver_times),
            "solver_time_std": np.std(self.solver_times),
        }
        return results


if __name__ == "__main__":
    q_init_list = [
        np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
        np.array((0.531, 1.36, 0.070, -1.065, 0.294, -1.2, -0.242)),
        np.array((-0.702, 0.355, -0.016, -1.212, 0.012, -0.502, -0.010)),
        np.array((0.531, 1.16, 0.070, -1.665, 0.294, -1.2, -0.242)),
        np.array((0.07, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
        np.array((0.531, 0.836, 0.070, -1.665, 0.294, -0.877, -0.242)),
        np.array((0.51, 0.67, -0.17, -1.73, 0.25, -0.86, -0.11)),
        np.array((0.91, 0.79, -0.22, -1.33, 1.20, -1.76, -1.06)),
        np.array((0.83, 0.53, -0.11, -0.95, 1.05, -1.24, -1.45)),
        np.array((0.87, 0.14, -0.37, -1.81, 0.46, -1.63, -0.91)),
    ]
    positions_obstacles_list = [
        [[0.5, 0., 0.55], [0.5, 0., 10.1]],
        [[0.5, 0.15, 0.05], [0.5, 0.15, 0.2]],
        [[0.5, -0.35, 0.5], [0.24, 0.45, 10.2]],
        [[0.45, 0.02, 0.2], [0.6, 0.02, 0.2]],
        [[0.5, -0.0, 0.5], [0.3, -0.1, 10.5]],
        [[0.5, -0.05, 0.3], [0.5, 0.2, 10.25]],
        [[0.5, -0.0, 0.2], [0.5, 0.2, 10.4]],
        [[0.5, -0.0, 0.28], [0.5, 0.2, 10.4]],
        [[0.5, 0.25, 0.55], [0.5, 0.2, 10.4]],
        [[0.5, 0.1, 0.45], [0.5, 0.2, 10.4]],
    ]
    init_pos = [0.5312149701934061, 0.8355097803551061, 0.0700492926199493, -1.6651880968294615, 0.2936679665237496, -0.8774234085561443, -0.24231138029250487]
    example_class = example_kuka_stableMP_fabrics_20()
    example_class.overwrite_defaults(init_pos=q_init_list[3], positions_obstacles=positions_obstacles_list[3], render=True, nr_obst=100)
    example_class.construct_example()
    res = example_class.run_kuka_example()

    print(" -------------------- results -----------------------")
    print("min_distance:", res["min_distance"])
    print("collision occurred:", res["collision"])
    print("goal reached:", res["goal_reached"])
    print("time_to_goal:", res["time_to_goal"])
    print("solver time: mean: ", res["solver_time"], " , std: ", res["solver_time_std"])