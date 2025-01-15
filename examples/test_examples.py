import pytest
import warnings
"""
This script contains tests of the pumafabrics examples, but NOT of the evaluation scripts. 
"""

def blueprint_test(test_main):
    """
    Blueprint for environment tests.
    An environment main always has the one argument:
        - render: bool

    The function verifies if the main returns a list of observations.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        history = test_main(render=False)
    assert isinstance(history, dict)

def test_kuka_fabrics():
    from examples.kuka_Fabrics import main
    blueprint_test(main)

def test_kuka_ModulationIK():
    from examples.kuka_PUMA_3D_ModulationIK import main
    blueprint_test(main)

def test_kuka_TamedPUMA():
    from examples.kuka_TamedPUMA import main
    blueprint_test(main)

def test_pointrobot_fabrics():
    from examples.point_robot_Fabrics import main
    blueprint_test(main)

def test_pointrobot_PUMA():
    from examples.point_robot_PUMA import main
    blueprint_test(main)

def test_pointrobot_TamedPUMA_FPM():
    from examples.point_robot_TamedPUMA_FPM import main
    blueprint_test(main)

def test_pointrobot_TamedPUMA_CPM():
    from examples.point_robot_TamedPUMA_CPM import main
    blueprint_test(main)

def test_pointrobot_TamedPUMA_hierarchical():
    from examples.point_robot_TamedPUMA_hierarchical import main
    blueprint_test(main)

def test_mobile_manipulator_fabrics():
    from examples.mobile_manipulator_Fabrics import main
    blueprint_test(main)