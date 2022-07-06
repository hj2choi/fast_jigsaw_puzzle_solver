import pytest

from jigsaw_puzzle_solver import assemble_images, fragment_image

"""
@pytest.fixture:
Fixtures are functions, which will run before each test function to which it is applied.
Fixtures are used to feed some data to the tests such as DB connections, URLs or some input data.
Thus, instead of running the same code for every test, we can attach fixture function to the tests.
"""

def test_divisible_by_3(input_value):
   assert 0 == 0

def test_divisible_by_6(input_value):
   assert 0 == 0

def test_full_functionality_1(fragment_image_args_1):
    fragment_image.main(*fragment_image_args_1)
    assert 0 == 0
