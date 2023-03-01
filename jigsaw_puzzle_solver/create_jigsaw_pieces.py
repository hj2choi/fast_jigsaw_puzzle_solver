""" create_jigsaw_pieces.py
This script takes in an image, number of rows and columns, and prefix to jigsaw piece filenames.

Divides the input image into pieces, jumbles them (8 random orientations), and saves them as a jigsaw puzzle.
The resulting images are saved in the specified output directory with a random unique filename with specified prefix.
The number of pieces is equal to the product of rows and columns.

e.g)
python create_jigsaw_pieces.py sample_images/testimg_1.jpg 3 2 jigsaw_pieces_filename_prefix
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
        "jigsaw_pieces_dir": "jigsaw_pieces",
        "debug": True
    }
}
RANDOM_SEED = 32  # for reproducibility


def rand_flip_horizontal(img):
    """
    flips image horizontally with 50% chance.
    """
    if random.random() > 0.5:
        print("flip_x", end=" ")
        return np.flip(img, 1)
    return img


def rand_flip_vertical(img):
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


def jumble_jigsaw_piece(img, file_prefix, sequence_no):
    """
    Applies random transformations to a given image piece and saves it to a random unique filename with given prefix.

    Args:
        img (npArray):          jigsaw piece image of shape (h, w, RGB)
        file_prefix (str):      output filename prefix
        piece_no (int):         piece number for unique identifier.
    """
    uuid_filename = hashlib.md5(str.encode(file_prefix + str(sequence_no))).hexdigest()
    print("jigsaw piece", uuid_filename, end=": ")
    img = rand_rotate90(rand_flip_vertical(rand_flip_horizontal(img)))
    cv2.imwrite(file_prefix + "_" + uuid_filename + ".png", img)
    print("")


def main(img_path, num_cols, num_rows, jigsaw_piece_prefix,
         verbose=False, jigsaw_pieces_dir="jigsaw_pieces"):
    """
    1. read and slice image into equal-sized jigsaw pieces as specified in args
    2. for each jigsaw piece: apply random set of transformations.
    3. save jigsaw pieces to random unique filenames.
    """
    s_time = time.time()
    random.seed(RANDOM_SEED)  # for reproducibility

    # read image
    img = cv2.imread(img_path)
    if img is None:
        print("ERROR: cannot read file")
        return
    print("create_jigsaw_pieces.py: loaded image from", img_path)

    if not verbose:
        sys.stdout = open(os.devnull, 'w')  # block stdout

    if not os.path.exists(jigsaw_pieces_dir):
        os.makedirs(jigsaw_pieces_dir)
    # remove all jigsaw piece files sharing the same designated prefix.
    for file in filter(lambda fname: fname.startswith(jigsaw_piece_prefix) and
                                     fname.endswith(".png"), os.listdir(jigsaw_pieces_dir)):
        os.remove(os.path.join(jigsaw_pieces_dir, file))

    # slice image into equal-sized jigsaw pieces and process each piece
    h, w = len(img) // num_rows, len(img[0]) // num_cols  # height and width
    for i in range(num_rows):
        for j in range(num_cols):
            top, left = int(i * h), int(j * w)  # row start, col start
            jumble_jigsaw_piece(img[top: top + h, left: left + w],
                                os.path.join(jigsaw_pieces_dir, jigsaw_piece_prefix),
                                i * num_cols + j)

    sys.stdout = sys.__stdout__  # restore stdout
    print("created", num_rows * num_cols, "jigsaw puzzle pieces.")
    print(time.time() - s_time, "seconds elapsed")


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('img_path', type=str, help='Path to an input image')
    ap.add_argument('col_slice', type=int, help='Number of columns')
    ap.add_argument('row_slice', type=int, help='Number of rows')
    ap.add_argument('jigsaw_piece_prefix', type=str, help='filename prefix to jigsaw piece image files')
    ap.add_argument('--verbose', '-v', required=False, action='store_true',
                    help='increase output verbosity')
    ap.add_argument('--config_file', '-c', required=False, default="./config/config.ini",
                    action='store', nargs=1, help='configuration ini file')
    args = ap.parse_args()

    cp = ConfigParser()
    cp.read_dict(DEFAULT_CONFIG)
    cp.read(args.config_file)

    main(args.img_path, args.col_slice, args.row_slice, args.jigsaw_piece_prefix,
         cp.getboolean("config", "debug") or args.verbose, cp.get("config", "jigsaw_pieces_dir"))
