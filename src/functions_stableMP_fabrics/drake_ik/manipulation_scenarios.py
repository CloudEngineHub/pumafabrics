import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    Cylinder,
    DiagramBuilder,
    InverseKinematics,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    RigidTransform,
    Role,
    RollPitchYaw,
    RotationMatrix,
    Solve,
    Parser,
    StartMeshcat,
    RevoluteJoint,
    # for add_shape:
    UnitInertia,
    SpatialInertia,
    Box,
    Sphere,
    Capsule,
    ProximityProperties,
    AddContactMaterial,
    CoulombFriction,
    AddCompliantHydroelasticProperties,
)

class ManipulationScenarios():
    def __init__(self):
        dt = 1
        self.q0_default = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]

    def AddIiwa(self, plant, collision_model="no_collision", q0=None):
        parser = Parser(plant)
        # parser.AddModel() -> should work with urdf files
        # https://drake.mit.edu/pydrake/pydrake.multibody.parsing.html
        iiwa = parser.AddModelsFromUrl(
            f"package://drake/manipulation/models/iiwa_description/iiwa7/iiwa7_{collision_model}.sdf"
        )[0]
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

        # Set default positions:
        # q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]
        if q0 is None:
            q0 = self.q0_default
        index = 0
        for joint_index in plant.GetJointIndices(iiwa):
            joint = plant.get_mutable_joint(joint_index)
            if isinstance(joint, RevoluteJoint):
                joint.set_default_angle(q0[index])
                index += 1

        return iiwa

    def AddShape(self, plant, shape, name, mass=1, mu=1, color=[0.5, 0.5, 0.9, 1.0]):
        instance = plant.AddModelInstance(name)
        # TODO: Add a method to UnitInertia that accepts a geometry shape (unless
        # that dependency is somehow gross) and does this.
        if isinstance(shape, Box):
            inertia = UnitInertia.SolidBox(
                shape.width(), shape.depth(), shape.height()
            )
        elif isinstance(shape, Cylinder):
            inertia = UnitInertia.SolidCylinder(
                shape.radius(), shape.length(), [0, 0, 1]
            )
        elif isinstance(shape, Sphere):
            inertia = UnitInertia.SolidSphere(shape.radius())
        elif isinstance(shape, Capsule):
            inertia = UnitInertia.SolidCapsule(
                shape.radius(), shape.length(), [0, 0, 1]
            )
        else:
            raise RuntimeError(
                f"need to write the unit inertia for shapes of type {shape}"
            )
        body = plant.AddRigidBody(
            name,
            instance,
            SpatialInertia(
                mass=mass, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=inertia
            ),
        )
        if plant.geometry_source_is_registered():
            proximity_properties = ProximityProperties()
            AddContactMaterial(
                1e4, 1e7, CoulombFriction(mu, mu), proximity_properties
            )
            AddCompliantHydroelasticProperties(0.01, 1e8, proximity_properties)
            plant.RegisterCollisionGeometry(
                body, RigidTransform(), shape, name, proximity_properties
            )

            plant.RegisterVisualGeometry(
                body, RigidTransform(), shape, name, color
            )

        return instance