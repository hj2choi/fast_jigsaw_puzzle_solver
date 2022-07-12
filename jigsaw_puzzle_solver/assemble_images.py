""" assemble_images.py
Prerequisite: fragmented images (via fragment_image.py)
This script takes in in_prefix, cols and rows and out_prefix.

Reads all fragmented images with filename starting with $in_prefix
and reconstructs them back to original image.

e.g)
python assemble_images.py test_fragments 3 2 test_out
"""

import os
import sys
import time
from argparse import ArgumentParser
from configparser import ConfigParser

from .assembler import assembler as asm

DEFAULT_CONFIG = {
    "config": {
        "output_dir": "image_fragments",
        "debug": False,
        "show_assembly_animation": True,
        "animation_interval_millis": 200
    }
}


def main(args, cfg):
    """
    main jigsaw puzzle solver routine
    """
    s_time = time.time()
    images_dir = cfg.get("config", "fragments_dir")
    output_directory = cfg.get("config", "output_dir")
    show_assembly_animation = cfg.getboolean("config", "show_assembly_animation") or \
        args.show_animation
    verbose = cfg.getboolean("config", "debug") or args.verbose
    animation_interval = int(cfg.get("config", "animation_interval_millis"))

    # initialize
    assembler = asm.ImageAssembler.load_from_filepath(
        images_dir, args.in_prefix, args.cols, args.rows)
    print("assemble_image.py:", len(assembler.raw_imgs), "files loaded", flush=True)
    if assembler.max_rows * assembler.max_cols != len(assembler.raw_imgs):
        print("WARNING: incorrect slicing dimension.")
    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # block stdout

    # main merge algorithm
    assembler.assemble()

    # save result to output directory, and show animation.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    assembler.save_assembled_image(output_directory + "/" + args.out_filename)
    sys.stdout = sys.__stdout__  # restore stdout
    print("total elapsed time:", time.time() - s_time, "seconds", flush=True)
    if show_assembly_animation:
        assembler.start_assembly_animation(args.show_spanning_tree, animation_interval)


if __name__ == '__main__':
    AP = ArgumentParser()
    AP.add_argument('in_prefix', type=str, help='prefix to image fragments')
    AP.add_argument('cols', type=int, help='Number of column slices')
    AP.add_argument('rows', type=int, help='Number of row slices')
    AP.add_argument('out_filename', type=str, help='filename for reconstructed image')
    AP.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    AP.add_argument('--show_animation', '-a', required=False, action='store_true',
                    help='show image reconstruction animation')
    AP.add_argument('--show_spanning_tree', '-t', required=False, action='store_true',
                    help='show spanning tree on top of the animation (use with -a option)')
    AP.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store_true', help='configuration ini file')
    PARSED_ARGS = AP.parse_args()

    CP = ConfigParser()
    CP.read_dict(DEFAULT_CONFIG)
    CP.read(PARSED_ARGS.config_file)

    main(PARSED_ARGS, CP)
