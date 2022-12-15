""" test_all.py
tests the full fragmentation and assembly functionality.


@pytest.fixture:
Fixtures are functions, which will run before each test function to which it is applied.
Fixtures are used to feed some data to the tests such as DB connections, URLs or some input data.
Thus, instead of running the same code for every test, we can attach fixture function to the tests.
"""
import os
import math
import shutil
from unittest.mock import patch

import numpy as np
import pytest
import cv2

from jigsaw_puzzle_solver import fragment_image, assemble_images
from jigsaw_puzzle_solver.assembler import assembler as asm

TESTS_FRAGMENTS_DIR = os.path.join("tests", "temp_fragments")
TESTS_OUTPUT_DIR = os.path.join("tests", "temp_out")


def validate_reconstructed_img(original, reconstructed):
    """
    compares two images regardless of their orientation.
    crops original image to fit reconstructed one.

    Args:
        original (cv2 image)
        reconstructed (cv2 image)
    """
    for _ in range(4):
        reconstructed = cv2.rotate(reconstructed, cv2.ROTATE_90_CLOCKWISE)
        for _ in range(2):
            reconstructed = cv2.flip(reconstructed, 0)
            original_fitted = original[:reconstructed.shape[0], :reconstructed.shape[1]]
            if original_fitted.shape == reconstructed.shape and \
            math.sqrt(np.mean((original_fitted - reconstructed) ** 2)) < 1.0:
                return True
    return False


def setup():
    return


def teardown():
    shutil.rmtree(TESTS_FRAGMENTS_DIR)
    shutil.rmtree(TESTS_OUTPUT_DIR)


@patch("cv2.imshow", return_value=None)
@pytest.mark.parametrize("img_path, prefix, cols, rows, anim, anim_mst", [
    ("tests/images/test1.jpg", "integrated1", 6, 7, False, False),  # standard case, jpg file and rectangles
    ("tests/images/test5.png", "integrated2", 2, 3, False, False),  # standard case, png file
    ("tests/images/test3.jpg", "integrated3", 3, 3, False, False),  # standard case, jpg file and squares
    ("tests/images/test2.png", "integrated4", 1, 6, False, False),  # edge case
    ("tests/images/test2.png", "integrated5", 1, 1, False, False),  # edge case, no slicing at all
    ("tests/images/test4.png", "integrated6", 11, 12, False, False),  # standard, computation heavy
    ("tests/images/test1.jpg", "integrated7", 2, 3, True, False),  # draw animation
    ("tests/images/test1.jpg", "integrated10", 2, 3, True, True),  # draw animation with MST on top
    ("tests/images/test1.jpg", "integrated10", 3, 4, True, True),  # reuse fragment image prefix names
    ("tests/images/test2.png", "integrated10", 2, 2, True, True)  # reuse fragment image prefix names
])
def test_integrated_correct_cases(mock_imshow, img_path, prefix, cols, rows, anim, anim_mst):
    # integrated test, check for successful reconstruction
    fragment_image.main(img_path, cols, rows, prefix,
                        verbose=True, fragments_dir=TESTS_FRAGMENTS_DIR)
    assemble_images.main(prefix, cols, rows, imgs_dir=TESTS_FRAGMENTS_DIR, out_dir=TESTS_OUTPUT_DIR,
                         verbose=True, show_anim=anim, anim_interval=1, show_mst_on_anim=anim_mst)
    assert validate_reconstructed_img(cv2.imread(img_path),
                                      cv2.imread(os.path.join(TESTS_OUTPUT_DIR, prefix+".png")))


@patch("cv2.imshow", return_value=None)
@pytest.mark.parametrize("img_path, prefix, cols, rows", [
    ("tests/images/test5.png", "integrated_incorrect1", 1, 10),  # incorrect reconstruction
    ("tests/images/test5.png", "integrated_incorrect2", 10, 1),  # incorrect reconstruction 2
])
def test_integrated_incorrect_cases(mock_imshow, img_path, prefix, cols, rows):
    # integrated test, only check for safe program exit
    fragment_image.main(img_path, cols, rows, prefix,
                        verbose=True, fragments_dir=TESTS_FRAGMENTS_DIR)
    assemble_images.main(prefix, cols, rows, imgs_dir=TESTS_FRAGMENTS_DIR, out_dir=TESTS_OUTPUT_DIR,
                         verbose=True, show_anim=True, anim_interval=1, show_mst_on_anim=True)
    # an assert statement is deliberately omitted here.


