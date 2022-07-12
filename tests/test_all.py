""" test_all.py
tests the full fragmentation and assembly functionality.


@pytest.fixture:
Fixtures are functions, which will run before each test function to which it is applied.
Fixtures are used to feed some data to the tests such as DB connections, URLs or some input data.
Thus, instead of running the same code for every test, we can attach fixture function to the tests.
"""
import os

import pytest
import cv2

from jigsaw_puzzle_solver import fragment_image, assemble_images

def validate_reconstructed_img(original, reconstructed):
    """
    compare two images regardless of their orientation.

    Args:
        original (cv2 image)
        reconstructed (cv2 image)
    """
    #print(original)
    #print(reconstructed)
    for _ in range(4):
        reconstructed = cv2.rotate(reconstructed, cv2.cv2.ROTATE_90_CLOCKWISE)
        for _ in range(2):
            reconstructed = cv2.flip(reconstructed, 0)
            print("comparing",original.shape,reconstructed.shape)
            if original.shape == reconstructed.shape:
                diff = cv2.subtract(original, reconstructed)
                b, g, r = cv2.split(diff)
                print(cv2.countNonZero(b), cv2.countNonZero(g), cv2.countNonZero(r))
                if cv2.countNonZero(b) == 0 and \
                cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                    return True
    return False


def test_suite_1(fragment_image_args_1, assemble_images_args_1):
    # test suite 1
    fragment_image.main(*fragment_image_args_1)
    assemble_images.main(*assemble_images_args_1)

    original_image_path = fragment_image_args_1[0].image
    reconstructed_image_path = os.path.join(assemble_images_args_1[1]["config"]["output_dir"],
                                            assemble_images_args_1[0].out_filename+".png")
    print(original_image_path)
    print(reconstructed_image_path)
    assert validate_reconstructed_img(cv2.imread(original_image_path),
                                      cv2.imread(reconstructed_image_path))
