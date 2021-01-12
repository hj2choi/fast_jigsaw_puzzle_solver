import sys, os, time, random, hashlib
from configparser import ConfigParser
import numpy as np
import cv2

"""
wrapper functions for randomized transformation. initial random seed is set for reproducibility
"""
def rand_mirror(img):
    return np.flip(img, 1) if random.random() > 0.5 else img

def rand_flip(img):
    return np.flip(img, 0) if random.random() > 0.5 else img

def rand_rotate90(img, clockwise = False):
    return np.rot90(img) if random.random() > 0.5 else img

"""
randomly transform (mirror->flip->rotate) a given image segment and write to random filename.

@Parameters
img (npArray):              image segment of shape (h, w, RGB)
filename_prefix (str):      output filename prefix
"""
uid = 0
def process_image_segment(img, filepath_prefix):
    global uid
    img = rand_rotate90(rand_flip(rand_mirror(img)))
    hash = hashlib.md5(str.encode(filepath_prefix + str(uid))).hexdigest()
    cv2.imwrite(filepath_prefix + "_" + hash + ".png", img)
    print("processed segment:", img.shape, hash)
    uid += 1


def resolve_ambiguous_filename(input_filepath):
    if not os.path.exists(input_filepath):
        pwd = input_filepath.split("/")
        filename = pwd[-1]
        pathstring = "/".join(pwd[:-1]) if len(pwd)>1 else "."
        for path, dir, files in os.walk("./"):
            for file in files:
                if file.startswith(filename) and len(file.split(".")[0]) == len(filename):
                    #print("did you mean",(pathstring+"/"+file),"?\n")
                    print("did you mean",path+"/"+file,"?\n")
                    return pathstring+"/"+file
    return input_filepath

"""
main function takes 3 steps:
1. read image and trim image size in the manner suitable for cropping.
2. divide image into segments as specified in the input
3. for each image segments apply random set of transformations and save them with random filename
"""
def main(args, config):
    random.seed(24) # for reproducibility
    OUTPUT_DIR = config.get("config", "images_dir")
    VERBOSE = config.getboolean("config", "debug") or len(args) >= 5 and args[4] == "-v" # enable VERBOSE option for debugging
    if not VERBOSE:
        sys.stdout = open(os.devnull, 'w') # block print() functionality
    SOURCE_FILENAME = args[0]
    CUT_COLS = int(args[1])
    CUT_ROWS = int(args[2])
    OUTPUT_PREFIX = args[3]

    img = cv2.imread(resolve_ambiguous_filename(SOURCE_FILENAME))
    if img is None:
        print("ERRROR: cannot resolve filename")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    img = img[:len(img) - len(img) % CUT_ROWS, :len(img[0]) - len(img[0]) % CUT_COLS]
    print("image trimmed to shape", img.shape)

    # divide image into segments and process each segment
    h, w = len(img) // CUT_ROWS, len(img[0]) // CUT_COLS # height and width
    for i in range(CUT_ROWS):
        for j in range(CUT_COLS):
            rs, cs = int(i * h), int(j * w) # row start, col start
            process_image_segment(img[rs: rs + h, cs: cs + w], OUTPUT_DIR + "/" + OUTPUT_PREFIX)


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config.ini")
    main(sys.argv[1:], config)
