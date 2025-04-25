"""
Adapted by:
    Rodrigo Perez-Dattari <r.j.perezdattari@tudelft.nl>
"""
import numpy as np
import os
from roboticstoolbox.robot.ERobot import ERobot

class iiwa(ERobot):
    """
    Class that imports a KUKA iiwa URDF model

    ``iiwa()`` is a class which imports a KUKA iiwa robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self, model):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            base_dir + '/../config/urdfs/%s.urdf' % model
        )

        super().__init__(
            links, name=name, manufacturer='KUKA', gripper_links=links[9]
        )

        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0, 0]))


if __name__ == "__main__":
    robot = iiwa('iiwa14')
    print(robot)

    for link in robot.grippers[0].links:
        print(link)
