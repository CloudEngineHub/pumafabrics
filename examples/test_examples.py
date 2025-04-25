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
        history = test_main(render=False, n_steps=100)
    assert isinstance(history, dict)

def test_kuka_fabrics():
    from examples.kuka_Fabrics import main
    blueprint_test(main)

def test_kinova_fabrics():
    from examples.kinova_Fabrics import main
    blueprint_test(main)

def test_dinova_fabrics():
    from examples.dinova_Fabrics import main
    blueprint_test(main)

def test_kuka_ModulationIK():
    from examples.kuka_PUMA_3D_ModulationIK import main
    blueprint_test(main)

def test_kinova_ModulationIK():
    from examples.kinova_PUMA_3D_ModulationIK import main
    blueprint_test(main)

def test_kuka_TamedPUMA():
    from examples.kuka_TamedPUMA import main
    blueprint_test(main)

def test_kinova_TamedPUMA_3D_hierarchical():
    from examples.kinova_TamedPUMA_3D_hierarchical import main
    blueprint_test(main)

def test_dinova_TamedPUMA_3D_hierarchical():
    from examples.dinova_TamedPUMA_3D_hierarchical import main
    blueprint_test(main)

def test_kinova_TamedPUMA_1stR3S3_hierarchical():
    from examples.kinova_TamedPUMA_1stR3S3_hierarchical import main
    blueprint_test(main)

def test_dinova_TamedPUMA_1stR3S3_hierarchical():
    from examples.dinova_TamedPUMA_1stR3S3_hierarchical import main
    blueprint_test(main)

def test_kinova_TamedPUMA_2ndR3S3_hierarchical():
    from examples.kinova_TamedPUMA_1stR3S3_hierarchical import main
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