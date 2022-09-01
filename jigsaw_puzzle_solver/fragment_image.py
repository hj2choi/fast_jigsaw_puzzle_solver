""" fragment_image.py
This script takes in an image, cols and rows and out_prefix.

splits image into cols * rows number of image slices.
Then, randomly flips (horizontal&vertical) and rotates them.
resulting images are saved to {$config.fragments_dir}/{$out_prefix}_{rand_hash}.png

e.g)
python fragment_image.py sample_images/testimg_1.jpg 3 2 test_fragments
"""

import hashlib
import os
import random
import sys
import time
from argparse import ArgumentParser
from configparser import ConfigParser

import cv2
import numpy as np

DEFAULT_CONFIG = {
    "config": {
        "fragments_dir": "image_fragments",
        "debug": True
    }
}
RANDOM_SEED = 32  # for reproducibility


def rand_flip_x(img):
    """
    flips image horizontally with 50% chance.
    """
    if random.random() > 0.5:
        print("flip_x", end=" ")
        return np.flip(img, 1)
    return img


def rand_flip_y(img):
    """
    flips image vertically with 50% chance.
    """
    if random.random() > 0.5:
        print("flip_y", end=" ")
        return np.flip(img, 0)
    return img


def rand_rotate90(img, clockwise=False):
    """
    rotates image 90 degrees with 50% chance.
    """
    if random.random() > 0.5:
        print("rotate", end=" ")
        return np.rot90(img, 1) if clockwise else np.rot90(img, 3)
    return img


def process_image_segment(img, file_prefix, sequence_no):
    """
    randomly transform (x,y flip + rotate) a given image
    and write to random unique filename starting with {file_prefix}.

    Args:
        img (npArray):          image segment of shape (h, w, RGB)
        file_prefix (str):      output filename prefix
        sequence_no (int):      sequence number for unique identifier.
    """
    uuid_filename = hashlib.md5(str.encode(file_prefix + str(sequence_no))).hexdigest()
    print("image fragment", uuid_filename, end=": ")
    img = rand_rotate90(rand_flip_y(rand_flip_x(img)))
    cv2.imwrite(file_prefix + "_" + uuid_filename + ".png", img)
    print("")


def main(args, cfg):
    """
    1. read and slice image into y*x pieces as specified in args
    2. for each image slice: apply random set of transformations.
    3. save fragmented images to random filenames.
    """
    s_time = time.time()
    random.seed(RANDOM_SEED)  # for reproducibility

    output_dir = cfg.get("config", "fragments_dir")
    verbose = cfg.getboolean("config", "debug") or args.verbose

    # read image
    img = cv2.imread(args.image)
    if img is None:
        print("ERROR: cannot read file")
        return
    print("fragment_image.py: loaded image from", args.image)

    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # block stdout

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # remove all image fragment files sharing the same designated prefix.
    for file in filter(lambda fname: fname.startswith(args.out_prefix) and
                        fname.endswith(".png"), os.listdir(output_dir)):
        os.remove(os.path.join(output_dir, file))

    # slice image into uniform shapes and process each slices
    h, w = len(img) // args.rows, len(img[0]) // args.cols  # height and width
    for i in range(args.rows):
        for j in range(args.cols):
            top, left = int(i * h), int(j * w)  # row start, col start
            process_image_segment(img[top: top + h, left: left + w],
                                  os.path.join(output_dir, args.out_prefix),
                                  i*args.cols+j)

    sys.stdout = sys.__stdout__  # restore stdout
    print("fragmented image into", args.rows * args.cols, "slices. ")
    print(time.time() - s_time, "seconds elapsed")


if __name__ == '__main__':
    AP = ArgumentParser()
    AP.add_argument('image', type=str, help='Path to input image')
    AP.add_argument('cols', type=int, help='Number of column slices')
    AP.add_argument('rows', type=int, help='Number of row slices')
    AP.add_argument('out_prefix', type=str, help='filename prefix to fragmented image files')
    AP.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    AP.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store', nargs=1, help='configuration ini file')
    PARSED_ARGS = AP.parse_args()

    CP = ConfigParser()
    CP.read_dict(DEFAULT_CONFIG)
    CP.read(PARSED_ARGS.config_file)

    main(PARSED_ARGS, CP)
