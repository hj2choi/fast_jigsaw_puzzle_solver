import hashlib
import os
import random
import sys
import time
from configparser import ConfigParser

import cv2
import numpy as np

RANDOM_SEED = 32  # for reproducibility
incremental_id = 0  # Unique ID for each image fragment (to be hashed)


def rand_flip_x(img):
    if random.random() > 0.5:
        print("flip_x", end=" ")
        return np.flip(img, 1)
    return img


def rand_flip_y(img):
    if random.random() > 0.5:
        print("flip_y", end=" ")
        return np.flip(img, 0)
    return img


def rand_rotate90(img, clockwise=False):
    if random.random() > 0.5:
        print("rotate", end=" ")
        return np.rot90(img)
    return img


def process_image_segment(img, filepath_prefix):
    """
        randomly transform (mirror->flip->rotate) a given image segment and write to random filename.
        incremental unique id is used to prevent filenames from clashing

        @Parameters
        img (npArray):              image segment of shape (h, w, RGB)
        filename_prefix (str):      output filename prefix
    """
    global incremental_id
    randomized_name = hashlib.md5(str.encode(filepath_prefix + str(incremental_id))).hexdigest()
    print("image fragment", randomized_name, end=": ")
    img = rand_rotate90(rand_flip_y(rand_flip_x(img)))
    cv2.imwrite(filepath_prefix + "_" + randomized_name + ".png", img)
    print("")
    incremental_id += 1


def resolve_ambiguous_filename(input_filepath):
    """
        resolve filename if filename doesn't match exactly.
    """
    if not os.path.exists(input_filepath):
        pwd = input_filepath.split("/")
        filename = pwd[-1]
        path_str = "/".join(pwd[:-1]) if len(pwd) > 1 else "."
        for path, dir, files in os.walk("./"):
            for file in files:
                if file.startswith(filename) and len(file.split(".")[0]) == len(filename):
                    # print("did you mean", path + "/" + file,"?")
                    return path_str + "/" + file
    return input_filepath


def main(args, cfg):
    """
        2. read and slice image into y*x pieces as specified in args
        3. for each image slice: apply random set of transformations and save them to random filename
    """
    s_time = time.time()
    random.seed(RANDOM_SEED)  # for reproducibility
    output_dir = cfg.get("config", "fragments_dir")
    verbose = cfg.getboolean("config", "debug") or len(args) >= 5 and args[4] == "-v"  # verbose option for debugging
    source_filename = args[0]
    cut_cols = int(args[1])
    cut_rows = int(args[2])
    output_prefix = args[3]

    # read image
    img = cv2.imread(resolve_ambiguous_filename(source_filename))
    if img is None:
        print("ERROR: cannot resolve filename")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("fragment_image.py: loaded image from", source_filename)
    if not verbose or cut_rows * cut_cols > 12:
        sys.stdout = open(os.devnull, 'w')  # block stdout

    # slice image into uniform shapes and process each slices
    h, w = len(img) // cut_rows, len(img[0]) // cut_cols  # height and width
    for i in range(cut_rows):
        for j in range(cut_cols):
            rs, cs = int(i * h), int(j * w)  # row start, col start
            process_image_segment(img[rs: rs + h, cs: cs + w], output_dir + "/" + output_prefix)
    sys.stdout = sys.__stdout__  # restore stdout
    print("fragmented image into", cut_rows * cut_cols, "slices. ")
    print(time.time() - s_time, "seconds elapsed")


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config.ini")
    main(sys.argv[1:], config)
