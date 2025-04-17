import os, time
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from mpscenes.goals.goal_composition import GoalComposition
from pumafabrics.tamed_puma.tamedpuma.parametrized_planner_extended import ParameterizedFabricPlannerExtended
import pytorch_kinematics as pk
import torch

class FabricsController:
    def __init__(self, params):
        self.update_params(params)
        self.solver_times = []

    def update_params(self, params):
        self.params = params

    def construct_fk(self):
        absolute_path = os.path.dirname(os.path.abspath(__file__))
        with open(absolute_path + "/../config/urdfs/"+self.params["robot_name"]+".urdf", "r", encoding="utf-8") as file:
            urdf = file.read()
        self.forward_kinematics = GenericURDFFk(
            urdf,
            root_link=self.params["root_link"],
            end_links=self.params["end_links"],
        )

    def set_planner(self, goal: GoalComposition, nr_plane_constraints=1):
        """
        Initializes the fabric planner.
        """
        self.construct_fk()
        planner = ParameterizedFabricPlannerExtended(
            self.params["dof"],
            self.forward_kinematics,
            time_step=self.params["dt"],
        )
        planner.set_components(
            collision_links=self.params["collision_links"],
            goal=goal,
            number_obstacles=self.params["nr_obst"],
            number_plane_constraints=nr_plane_constraints, #todo
            limits=self.params["iiwa_limits"],
        )
        planner.concretize_extensive(mode=self.params["mode"], time_step=self.params["dt"], extensive_concretize=self.params["bool_extensive_concretize"], bool_speed_control=self.params["bool_speed_control"])
        return planner, self.forward_kinematics

    def set_avoidance_planner(self, goal=None):
        self.planner_avoidance, self.fk = self.set_planner(goal=goal, nr_plane_constraints=0)
        return self.planner_avoidance, self.fk

    def set_full_planner(self, goal:GoalComposition):
        self.planner_full, self.fk = self.set_planner(goal=goal)
        # rotation matrix for the goal orientation:
        self.rot_matrix = pk.quaternion_to_matrix(torch.FloatTensor(self.params["orientation_goal"]).cuda()).cpu().detach().numpy()
        return self.planner_full, self.fk

    def compute_action_full(self, q, ob_robot, obstacles: list, nr_obst=0, goal_pos=None, weight_goal_0=None, weight_goal_3=1., x_goal_3=0.):
        time0 = time.perf_counter()
        x_goal_1_x = ob_robot['FullSensor']['goals'][nr_obst+3]['position']
        x_goal_2_z = ob_robot['FullSensor']['goals'][nr_obst+4]['position']
        p_orient_rot_x = self.rot_matrix @ x_goal_1_x
        p_orient_rot_z = self.rot_matrix @ x_goal_2_z

        if goal_pos is None:
            goal_pos = ob_robot['FullSensor']['goals'][2 + nr_obst]['position']
        if weight_goal_0 is None:
            weight_goal_0 = ob_robot['FullSensor']['goals'][2 + nr_obst]['weight']

        arguments_dict = dict(
            q=q,
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0 = goal_pos[0:3],
            weight_goal_0 = weight_goal_0,
            x_goal_1 = p_orient_rot_x,
            weight_goal_1 = 0. , #ob_robot['FullSensor']['goals'][3+nr_obst]['weight'],
            x_goal_2=p_orient_rot_z,
            weight_goal_2=0, #ob_robot['FullSensor']['goals'][4 + nr_obst]['weight'],
            x_goal_3=x_goal_3,
            weight_goal_3 = 0, #weight_goal_3,
            x_obsts=[obstacles[i]["position"] for i in range(len(obstacles))],
            radius_obsts=[obstacles[i]["size"] for i in range(len(obstacles))],
            # radius_body_links=self.params["collision_radii"],
            constraint_0=[0, 0, 1, 0],
        )
        for i, collision_link in enumerate(self.params["collision_links"]):
            arguments_dict["radius_body_"+collision_link] = list(self.params["collision_radii"].values())[i]
        action = self.planner_full.compute_action(
            **arguments_dict)
        self.solver_times.append(time.perf_counter() - time0)
        return action, [], [], []

    def compute_action_avoidance(self, q, ob_robot):
        nr_obst = self.params["nr_obst"]
        if nr_obst>0:
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                x_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
                radius_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['size'],
                x_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['position'],
                radius_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst + 1]['size'],
                constraint_0=np.array([0, 0, 1, 0.0]))
            for i, collision_link in enumerate(self.params["collision_links"]):
                arguments_dict["radius_body_" + collision_link] = list(self.params["collision_radii"].values())[i]
        else:
            arguments_dict = dict(
                q=q,
                qdot=ob_robot["joint_state"]["velocity"],
                constraint_0=np.array([0, 0, 1, 0.0]))
            for i, collision_link in enumerate(self.params["collision_links"]):
                arguments_dict["radius_body_" + collision_link] = list(self.params["collision_radii"].values())[i]

        M_avoidance, f_avoidance, action_avoidance, xddot_speed_avoidance = self.planner_avoidance.compute_M_f_action_avoidance(
            **arguments_dict)
        qddot_speed = np.zeros((self.params["dof"],))  # todo: think about what to do with speed regulation term!!
        return action_avoidance, M_avoidance, f_avoidance, qddot_speed

    def request_solver_times(self):
        return self.solver_times