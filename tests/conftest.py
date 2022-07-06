import pytest
from argparse import ArgumentParser
from configparser import ConfigParser

@pytest.fixture
def input_value():
   input = 39
   return input


@pytest.fixture
def fragment_image_args_1():
    CP = ConfigParser()
    CP.read_dict({
        "config": {
            "fragments_dir": "image_fragments",
            "debug": False
        }
    })

    AP = ArgumentParser()
    AP.add_argument('image', type=str, help='Path to input image')
    AP.add_argument('cols', type=int, help='Number of column slices')
    AP.add_argument('rows', type=int, help='Number of row slices')
    AP.add_argument('out_prefix', type=str, help='filename prefix to fragmented image files')
    AP.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    AP.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store_true', help='configuration ini file')
    PARSED_ARGS = AP.parse_args([
        "sample_images/testimg_1.jpg",
        "3",
        "2",
        "test_suite_1"
    ])

    return PARSED_ARGS, CP



@pytest.fixture
def fragment_image_args_1():
    CP = ConfigParser()
    CP.read_dict({
        "config": {
            "fragments_dir": "image_fragments",
            "debug": False
        }
    })

    AP = ArgumentParser()
    AP.add_argument('image', type=str, help='Path to input image')
    AP.add_argument('cols', type=int, help='Number of column slices')
    AP.add_argument('rows', type=int, help='Number of row slices')
    AP.add_argument('out_prefix', type=str, help='filename prefix to fragmented image files')
    AP.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    AP.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store_true', help='configuration ini file')
    PARSED_ARGS = AP.parse_args([
        "sample_images/testimg_1.jpg",
        "3",
        "2",
        "test_suite_1"
    ])

    return PARSED_ARGS, CP
