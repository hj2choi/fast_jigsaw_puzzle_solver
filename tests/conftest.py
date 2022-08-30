import pytest
from argparse import ArgumentParser
from configparser import ConfigParser


@pytest.fixture
def gen_fragment_image_cmd():
    # generate fragment_image.py system command statement
    return lambda img_path, rows, cols, prefix: ' '.join(
                    ["python ./jigsaw_puzzle_solver/fragment_image.py",
                    "-c tests/config/tests_config.ini",
                    img_path, str(rows), str(cols), prefix])

@pytest.fixture
def gen_assemble_images_cmd():
    # generate assemble_images.py system command statement
    return lambda rows, cols, prefix: ' '.join(
                    ["python ./jigsaw_puzzle_solver/assemble_images.py",
                    "-c tests/config/tests_config.ini",
                    prefix, str(rows), str(cols), prefix])
