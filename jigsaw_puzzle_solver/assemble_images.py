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

from assembler import assembler as asm

DEFAULT_CONFIG = {
    "config": {
        "fragments_dir": "image_fragments",
        "output_dir": "images_out",
        "debug": False,
        "show_assembly_animation": True,
        "animation_interval_millis": 200
    }
}


def main(imgs_prefix, cols=0, rows=0, imgs_dir="image_fragments", out_dir="images_out", verbose=False,
         show_anim=True, anim_interval=200, show_mst_on_anim=False):
    """
    main jigsaw puzzle solver routine.
    """
    s_time = time.time()
    # initialize
    assembler = asm.ImageAssembler.load_from_filepath(imgs_dir, imgs_prefix, cols, rows)
    print("assemble_image.py:", len(assembler.raw_imgs), "files loaded", flush=True)
    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # block stdout

    # main merge algorithm
    assembler.assemble()

    # save result to output directory, and show animation.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    assembler.save_assembled_image(out_dir + "/" + imgs_prefix)
    sys.stdout = sys.__stdout__  # restore stdout
    print("total elapsed time:", time.time() - s_time, "seconds", flush=True)
    if show_anim:
        assembler.start_assembly_animation(show_mst_on_anim, anim_interval)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('img_prefix', type=str, help='prefix to image fragments')
    ap.add_argument('--cols', '-x', type=int, required=False, default=0, help='maximum columns size')
    ap.add_argument('--rows', '-y', type=int, required=False, default=0, help='maximum rows size')
    ap.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    ap.add_argument('--show_animation', '-a', required=False, action='store_true',
                    help='show image reconstruction animation')
    ap.add_argument('--show_spanning_tree', '-t', required=False, action='store_true',
                    help='show minimum spanning tree on top of the animation (-a option requried)')
    ap.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store', nargs=1, help='configuration ini file')
    args = ap.parse_args()

    cp = ConfigParser()
    cp.read_dict(DEFAULT_CONFIG)
    cp.read(args.config_file)

    main(args.img_prefix, args.cols, args.rows, cp.get("config", "fragments_dir"), cp.get("config", "output_dir"),
         cp.getboolean("config", "debug") or args.verbose,
         cp.getboolean("config", "show_assembly_animation") or args.show_animation,
         int(cp.get("config", "animation_interval_millis")), args.show_spanning_tree
         )
