import numpy as np
import casadi as ca
import quaternionic
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.helpers.variables import Variables
from fabrics.diffGeometry.geometry import Geometry
from scipy.spatial.transform import Rotation as R
import math
import spatial_casadi as sc

class construct_IL_geometry(object):
    def __init__(self, planner, dof: int, dimension_task: int, forwardkinematics: ca.SX, variables: dict, first_dim = 0, BOOL_pos_orient = False):
        # dimensions
        self._dof = dof
        self._dimension_task = dimension_task
        self.first_dim = first_dim
        self.BOOL_pos_orient = BOOL_pos_orient

        # store variables
        self._variables = variables
        self._q = variables._state_variables["q"]
        self._qdot = variables._state_variables["qdot"]

    def construct_symbolic_jacobians(self):
        www = 1

    def quat_vel_to_angular_vel(self, angle_quaternion, vel_quaternion):
        """
        Quaternion velocities to angular velocities,
        Slight modification of book:
        https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/RD2016script.pdf
        """
        H = self.map_angular_quat(angle_quaternion=angle_quaternion)
        angular_vel = 2*np.dot(H, vel_quaternion)
        return angular_vel

    def map_angular_quat(self, angle_quaternion):
        q0 = angle_quaternion[0]
        q1 = angle_quaternion[1]
        q2 = angle_quaternion[2]
        q3 = angle_quaternion[3]
        H = np.array([[-q1, q0, -q3, q2],
                   [-q2, q3, q0, -q1],
                   [-q3, -q2, q1, q0]
                   ])
        # H = np.array([[-q1, q0,  q3, -q2],
        #            [-q2, -q3, q0,  q1],
        #            [-q3,  q2, -q1, q0]
        #            ])
        return H