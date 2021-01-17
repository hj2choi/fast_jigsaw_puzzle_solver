import sys, os, time, random, hashlib
from configparser import ConfigParser
import numpy as np
import cv2

"""
wrapper functions for randomized transformation. initial random seed is set for reproducibility
"""
def rand_mirror(img):
    if random.random() > 0.5:
        print("mirror", end=" ")
        return np.flip(img, 1)
    return img

def rand_flip(img):
    if random.random() > 0.5:
        print("flip", end=" ")
        return np.flip(img, 0)
    return img

def rand_rotate90(img, clockwise = False):
    if random.random() > 0.5:
        print("rotate", end=" ")
        return np.rot90(img)
    return img

"""
randomly transform (mirror->flip->rotate) a given image segment and write to random filename.
incremental unique id is used to prevent filenames from clashing

@Parameters
img (npArray):              image segment of shape (h, w, RGB)
filename_prefix (str):      output filename prefix
"""
uid = 0
def process_image_segment(img, filepath_prefix):
    global uid
    hash = hashlib.md5(str.encode(filepath_prefix + str(uid))).hexdigest()
    print("image slice:", hash, end=" => ")
    img = rand_rotate90(rand_flip(rand_mirror(img)))
    cv2.imwrite(filepath_prefix + "_" + hash + ".png", img)
    print("")
    uid += 1

"""
resolve filename if filename doesn't match exactly.
"""
def resolve_ambiguous_filename(input_filepath):
    if not os.path.exists(input_filepath):
        pwd = input_filepath.split("/")
        filename = pwd[-1]
        pathstring = "/".join(pwd[:-1]) if len(pwd)>1 else "."
        for path, dir, files in os.walk("./"):
            for file in files:
                if file.startswith(filename) and len(file.split(".")[0]) == len(filename):
                    #print("did you mean",path+"/"+file,"?")
                    return pathstring+"/"+file
    return input_filepath

"""
main function takes 3 steps:
1. read image and trim image size for uniform sliciing.
2. divide image into sliced fragments as specified in the input
3. for each image fragments apply random set of transformations and save them with random filename
"""
def main(args, config):
    s_time = time.time()
    random.seed(24) # for reproducibility
    OUTPUT_DIR = config.get("config", "images_dir")
    VERBOSE = config.getboolean("config", "debug") or len(args) >= 5 and args[4] == "-v" # enable VERBOSE option for debugging
    SOURCE_FILENAME = args[0]
    CUT_COLS = int(args[1])
    CUT_ROWS = int(args[2])
    OUTPUT_PREFIX = args[3]

    # read image and trim
    img = cv2.imread(resolve_ambiguous_filename(SOURCE_FILENAME))
    if img is None:
        print("ERRROR: cannot resolve filename")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    img = img[:len(img) - len(img) % CUT_ROWS, :len(img[0]) - len(img[0]) % CUT_COLS]
    print("cut_image.py: image loaded and trimmed (h =",img.shape[0],", w =",img.shape[1],")")
    if not VERBOSE and CUT_ROWS*CUT_COLS > 8:
        sys.stdout = open(os.devnull, 'w') # block stdout

    # sllice image into uniform shapes and process each segment
    h, w = len(img) // CUT_ROWS, len(img[0]) // CUT_COLS # height and width
    for i in range(CUT_ROWS):
        for j in range(CUT_COLS):
            rs, cs = int(i * h), int(j * w) # row start, col start
            process_image_segment(img[rs: rs + h, cs: cs + w], OUTPUT_DIR + "/" + OUTPUT_PREFIX)
    sys.stdout = sys.__stdout__ # resotre stdout
    print("fragmented image into",CUT_ROWS*CUT_COLS,"slices. ")
    print(time.time()-s_time,"seconds elapsed")


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config.ini")
    main(sys.argv[1:], config)
