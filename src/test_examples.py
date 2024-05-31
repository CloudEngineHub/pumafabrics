import pytest
import warnings
def blueprint_test(test_main):
    """
    Blueprint for environment tests.
    An environment main always has the four arguments:
        - n_steps: int
        - render: bool

    The function verifies if the main returns a list of observations.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # parameters:
        history = test_main(n_steps=100, render=False)
    assert isinstance(history, dict)

def test_point_robot_examples():
    from point_robot_safeMP import example_point_robot_safeMP
    blueprint_test(example_point_robot_safeMP)
