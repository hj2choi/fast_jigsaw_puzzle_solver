import pytest
from argparse import ArgumentParser
from configparser import ConfigParser


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
        "4",
        "2",
        "test_suite_1"
    ])

    return PARSED_ARGS, CP


@pytest.fixture
def assemble_images_args_1():
    CP = ConfigParser()
    CP.read_dict({
        "config": {
            "fragments_dir": "image_fragments",
            "output_dir": "images_out",
            "debug": False,
            "show_assembly_animation": False,
            "animation_interval_millis": 1
        }
    })

    AP = ArgumentParser()
    AP.add_argument('in_prefix', type=str, help='prefix to image fragments')
    AP.add_argument('cols', type=int, help='Number of column slices')
    AP.add_argument('rows', type=int, help='Number of row slices')
    AP.add_argument('out_filename', type=str, help='filename for reconstructed image')
    AP.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    AP.add_argument('--show_animation', '-a', required=False, action='store_true',
                    help='show image reconstruction animation')
    AP.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store_true', help='configuration ini file')
    PARSED_ARGS = AP.parse_args([
        "test_suite_1",
        "4",
        "2",
        "test_suite_1_out"
    ])

    return PARSED_ARGS, CP
