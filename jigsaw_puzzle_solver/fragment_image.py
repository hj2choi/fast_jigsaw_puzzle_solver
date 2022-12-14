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


def main(img_path, col_slice, row_slice, fragments_prefix,
         verbose=False, fragments_dir="image_fragments"):
    """
    1. read and slice image into y*x pieces as specified in args
    2. for each image slice: apply random set of transformations.
    3. save fragmented images to random filenames.
    """
    s_time = time.time()
    random.seed(RANDOM_SEED)  # for reproducibility

    # read image
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: cannot read file")
        return
    print("fragment_image.py: loaded image from", img_path)

    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # block stdout

    if not os.path.exists(fragments_dir):
        os.makedirs(fragments_dir)
    # remove all image fragment files sharing the same designated prefix.
    for file in filter(lambda fname: fname.startswith(fragments_prefix) and
                       fname.endswith(".png"), os.listdir(fragments_dir)):
        os.remove(os.path.join(fragments_dir, file))

    # slice image into uniform shapes and process each slices
    h, w = len(img) // row_slice, len(img[0]) // col_slice  # height and width
    for i in range(row_slice):
        for j in range(col_slice):
            top, left = int(i * h), int(j * w)  # row start, col start
            process_image_segment(img[top: top + h, left: left + w],
                                  os.path.join(fragments_dir, fragments_prefix),
                                  i*col_slice+j)

    sys.stdout = sys.__stdout__  # restore stdout
    print("fragmented image into", row_slice * col_slice, "slices. ")
    print(time.time() - s_time, "seconds elapsed")


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('img_path', type=str, help='Path to an input image')
    ap.add_argument('col_slice', type=int, help='Number of column slices')
    ap.add_argument('row_slice', type=int, help='Number of row slices')
    ap.add_argument('fragments_prefix', type=str, help='filename prefix to fragmented image files')
    ap.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    ap.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store', nargs=1, help='configuration ini file')
    args = ap.parse_args()

    cp = ConfigParser()
    cp.read_dict(DEFAULT_CONFIG)
    cp.read(args.config_file)

    main(args.img_path, args.col_slice, args.row_slice, args.fragments_prefix,
         cp.getboolean("config", "debug") or args.verbose, cp.get("config", "fragments_dir"))
