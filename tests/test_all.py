import pytest

from jigsaw_puzzle_solver import assemble_images, fragment_image

"""
@pytest.fixture:
Fixtures are functions, which will run before each test function to which it is applied.
Fixtures are used to feed some data to the tests such as DB connections, URLs or some input data.
Thus, instead of running the same code for every test, we can attach fixture function to the tests.
"""


@pytest.fixture
def input_value():
    input = 39
    return input


@pytest.mark.mandatory
def full_functionality_test(input_value):
    print("hi")
    assert input_value % 3 == 0


