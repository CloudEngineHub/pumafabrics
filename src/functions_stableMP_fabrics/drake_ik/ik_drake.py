import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    Capsule,
    Cylinder,
    DiagramBuilder,
    InverseKinematics,
    RigidTransform,
    Role,
    RollPitchYaw,
    RotationMatrix,
    Solve,
)
from src.functions_stableMP_fabrics.drake_ik.manipulation_scenarios import ManipulationScenarios
from src.functions_stableMP_fabrics.drake_ik.pybullet_environments import PybulletEnvironments

class iiwa_example_drake():
    def __init__(self, render = True, dof = 7, mode = "acc", dt = 0.01, nr_obst = 1, obst0_pos = [0.25, 0.0, 0.5]):
        self.render = render
        self.dof = dof
        self.mode = mode
        self.dt = dt
        self.nr_obst = nr_obst #nr_obst
        self.obst0_pos = obst0_pos
        self.rot_matrix_default = np.array([[ 0.48549717,  0.57340063,  0.65992743],
                                            [-0.00599432, -0.75265884,  0.65838342],
                                            [ 0.87421769, -0.3235991 , -0.36197659]])
        self.desired_translation_default = [ 0.49255212, -0.42401989,  0.77684586]
        # self.rot_matrix_default = np.array([[-0.009203543268808336, 0.24662186422698304, -0.9690680837157453],
        #                        [0.9999576464987401, 0.0022698911362843137, -0.008919237799848434],
        #                        [0.0, -0.9691091288804563, -0.24663230996883403]])
        self.desired_translation_default = [ 0.49, -0.42,  0.77] #[0.49, -0.42, 0.77]

    def define_desired_pose(self, translation=None, Rot_matrix=None):
        """
        Define desired EEF pose
        """
        if Rot_matrix is None:
            Rot_matrix = self.rot_matrix_default
        desired_rotation = RotationMatrix(Rot_matrix)   # I guess you have to use their data type
        if translation is None:
            translation = self.desired_translation_default
        desired_translation = np.array(translation, dtype=float).T
        return desired_translation, desired_rotation

    def define_system(self, q0=None):
        """
        Define system
        """
        builder = DiagramBuilder()
        self.plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

        # add objects (robot + obstacles):
        manip_scenario = ManipulationScenarios()
        iiwa = manip_scenario.AddIiwa(self.plant, "with_box_collision", q0)
        if self.nr_obst>0:
            box = manip_scenario.AddShape(self.plant, Box(0.1, 0.1, 1.0), "box")
            self.plant.WeldFrames(
                self.plant.world_frame(),
                self.plant.GetFrameByName("box", box),
                RigidTransform(self.obst0_pos),
            )
        self.plant.Finalize()

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(context)

        q0 = self.plant.GetPositions(self.plant_context)
        print("q0", q0)
        self.gripper_frame = self.plant.GetFrameByName("iiwa_link_7", iiwa)
        return q0

    def build_solver(self, q0, desired_translation, desired_rotation):
        """
        Build IK solver
        """
        ik = InverseKinematics(self.plant, self.plant_context)
        ik.AddPositionConstraint(
                    self.gripper_frame,
                    [0, 0, 0],
                    self.plant.world_frame(),
                    desired_translation-0.001,
                    desired_translation+0.001,
        )
        ik.AddOrientationConstraint(
            self.gripper_frame,
            RotationMatrix(),
            self.plant.world_frame(),
            desired_rotation,
            0.01,
        )
        ik.AddMinimumDistanceLowerBoundConstraint(0.001, 0.1)

        prog = ik.get_mutable_prog()
        q = ik.q()
        prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
        prog.SetInitialGuess(q, q0)
        return ik

    def call_ik_solver(self, ik):
        """
        Call IK solver
        """
        result = Solve(ik.prog())
        if result.is_success():
            print("IK success")
        else:
            print("IK failure")

        """
        Access results
        """
        q_ik_result = result.GetSolution()
        print(q_ik_result)
        return q_ik_result

    def update_pos_obstacles(self, pos_obstacles):
        """ In case of dynamic obstacles, update the position of the obstacles """
        self.obst0_pos = pos_obstacles[0]

    def visualize_pose_pybullet(self, desired_translation, q_goal):
        """
        check if position makes physical sense in Pybullet:
        """
        goal_pos = desired_translation.tolist()

        envir_trial = PybulletEnvironments()
        (env, goal) = envir_trial.initialize_environment_kuka(self.render, mode=self.mode, dt=self.dt, init_pos=q_goal,
                                                              goal_pos=goal_pos, nr_obst=self.nr_obst,
                                                              obst0_pos = self.obst0_pos)
        action = np.zeros(self.dof)
        for _ in range(3000):
            ob, *_ = env.step(action)

    def main(self, translation=None, rot_matrix=None, q0=None):
        """ Define desired EEF pose """
        desired_translation, desired_rotation = self.define_desired_pose(translation, rot_matrix)

        """ Define system """
        q0 = self.define_system(q0)

        """ Define solver """
        ik = self.build_solver(q0, desired_translation, desired_rotation)

        """ Call IK solver """
        q_ik_result = self.call_ik_solver(ik)

        """ Visualize in Pybullet """
        # self.visualize_pose_pybullet(desired_translation, q_ik_result)
        return q_ik_result

if __name__ == "__main__":
    iiwa_example_class = iiwa_example_drake()
    q_desired = iiwa_example_class.main()