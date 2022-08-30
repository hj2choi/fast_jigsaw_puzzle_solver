""" test_all.py
tests the full fragmentation and assembly functionality.


@pytest.fixture:
Fixtures are functions, which will run before each test function to which it is applied.
Fixtures are used to feed some data to the tests such as DB connections, URLs or some input data.
Thus, instead of running the same code for every test, we can attach fixture function to the tests.
"""
import os
import math

import numpy as np
import pytest
import cv2

from jigsaw_puzzle_solver import fragment_image, assemble_images


def validate_reconstructed_img(original, reconstructed):
    """
    compares two images regardless of their orientation.
    crops original image to fit reconstructed one.

    Args:
        original (cv2 image)
        reconstructed (cv2 image)
    """
    for _ in range(4):
        reconstructed = cv2.rotate(reconstructed, cv2.cv2.ROTATE_90_CLOCKWISE)
        for _ in range(2):
            reconstructed = cv2.flip(reconstructed, 0)
            original_fitted = original[:reconstructed.shape[0], :reconstructed.shape[1]]
            if original_fitted.shape == reconstructed.shape and \
            math.sqrt(np.mean((original_fitted - reconstructed) ** 2)) < 1.0:
                return True
    return False


def test_integrated_1(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - standard case, jpg file and rectangle slices
    img_path = "tests/images/test1.jpg"
    prefix = "integrated_test_1"
    os.system(gen_fragment_image_cmd(img_path, 6, 7, prefix))
    os.system(gen_assemble_images_cmd(6, 7, prefix))
    assert validate_reconstructed_img(cv2.imread(img_path),
                                      cv2.imread("images_out/" + prefix + ".png"))


def test_integrated_2(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - standard case, png file and square slices
    img_path = "tests/images/test2.png"
    prefix = "integrated_test_2"
    os.system(gen_fragment_image_cmd(img_path, 4, 4, prefix))
    os.system(gen_assemble_images_cmd(4, 4, prefix))
    assert validate_reconstructed_img(cv2.imread(img_path),
                                      cv2.imread("images_out/" + prefix + ".png"))


def test_integrated_3(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - edge case
    img_path = "tests/images/test3.jpg"
    prefix = "integrated_test_3"
    os.system(gen_fragment_image_cmd(img_path, 1, 6, prefix))
    os.system(gen_assemble_images_cmd(6, 1, prefix))
    assert validate_reconstructed_img(cv2.imread(img_path),
                                      cv2.imread("images_out/" + prefix + ".png"))


def test_integrated_4(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - standard case, computation heavy
    img_path = "tests/images/test4.png"
    prefix = "integrated_test_4"
    os.system(gen_fragment_image_cmd(img_path, 11, 12, prefix))
    os.system(gen_assemble_images_cmd(11, 12, prefix))
    assert validate_reconstructed_img(cv2.imread(img_path),
                                      cv2.imread("images_out/" + prefix + ".png"))


def test_integrated_5(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - error case: row col mismatch
    # it should run succesfully
    img_path = "tests/images/test5.png"
    prefix = "integrated_test_5"
    os.system(gen_fragment_image_cmd(img_path, 2, 3, prefix))
    os.system(gen_assemble_images_cmd(8, 5, prefix))
    assert validate_reconstructed_img(cv2.imread(img_path),
                                      cv2.imread("images_out/" + prefix + ".png"))


def test_integrated_6(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - error case: row col mismatch
    # only check for safe program exit
    img_path = "tests/images/test5.png"
    prefix = "integrated_test_6"
    os.system(gen_fragment_image_cmd(img_path, 2, 3, prefix))
    ret = os.system(gen_assemble_images_cmd(1, 2, prefix))
    assert ret == 0


def test_integrated_7(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - error case: unsuccesful reconstruction
    # only check for safe program exit
    img_path = "tests/images/test5.png"
    prefix = "integrated_test_7"
    os.system(gen_fragment_image_cmd(img_path, 2, 8, prefix))
    ret = os.system(gen_assemble_images_cmd(2, 8, prefix))
    assert ret == 0


def test_integrated_8(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - reuse same prefix
    # it should run successfully
    img_path = "tests/images/test1.jpg"
    img_path_2 = "tests/images/test2.png"
    prefix = "integrated_test_8"
    os.system(gen_fragment_image_cmd(img_path, 2, 2, prefix))
    os.system(gen_fragment_image_cmd(img_path_2, 3, 3, prefix))
    os.system(gen_assemble_images_cmd(3, 3, prefix))
    assert validate_reconstructed_img(cv2.imread(img_path_2),
                                      cv2.imread("images_out/" + prefix + ".png"))


def test_integrated_9(gen_fragment_image_cmd, gen_assemble_images_cmd):
    # integrated test - reuse same prefix
    # it should run successfully
    img_path = "tests/images/test1.jpg"
    img_path_2 = "tests/images/test2.png"
    prefix = "integrated_test_9"
    os.system(gen_fragment_image_cmd(img_path, 3, 3, prefix))
    os.system(gen_fragment_image_cmd(img_path_2, 2, 2, prefix))
    os.system(gen_assemble_images_cmd(2, 2, prefix))
    assert validate_reconstructed_img(cv2.imread(img_path_2),
                                      cv2.imread("images_out/" + prefix + ".png"))
